"""SB3 2.4.1 PPO extension; train() adapted under SB3's MIT license.

One optimizer step combines PPO and forward KL. The lambda=0 path calls
upstream PPO.train directly. No upstream source or optimizer hooks are patched.
"""
from collections import namedtuple
import math

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance

KDSamples = namedtuple("KDSamples", (*RolloutBufferSamples._fields, "teacher_probs"))


def kd_weight(t, lambda_start=10.0, anneal_steps=100000):
    u = min(max(t / anneal_steps, 0), 1)
    return lambda_start * 0.5 * (1 + math.cos(math.pi * u))


def forward_kl(teacher, log_student):
    # xlogy(0,0)=0, so sparse teachers produce finite values and gradients.
    return (th.special.xlogy(teacher, teacher) - teacher * log_student).sum(dim=-1).mean()


class KDRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.teacher_probs = None

    def set_teacher(self, probabilities):
        arr = np.asarray(probabilities, dtype=np.float32)
        if (arr.shape != (self.buffer_size, self.n_envs, 7) or not np.isfinite(arr).all()
                or (arr < 0).any() or not np.allclose(arr.sum(-1), 1, atol=1e-6)):
            raise ValueError("Invalid or misaligned rollout teacher")
        self.teacher_probs = self.swap_and_flatten(arr)

    def _get_samples(self, batch_inds, env=None):
        samples = super()._get_samples(batch_inds, env)
        teacher = None if self.teacher_probs is None else self.to_torch(self.teacher_probs[batch_inds])
        return KDSamples(*samples, teacher)


class _RawCollector(BaseCallback):
    def __init__(self):
        super().__init__()
        self.states = []

    def _on_step(self):
        self.states.append(self.training_env.get_attr("last_transition_raw"))
        return True


class RA4RLPPO(PPO):
    def __init__(self, *args, teacher=None, lambda_start=10.0, anneal_steps=100000, **kwargs):
        self.teacher = teacher
        self.lambda_start, self.anneal_steps = lambda_start, anneal_steps
        self.current_lambda = 0.0
        kwargs.setdefault("rollout_buffer_class", KDRolloutBuffer)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["teacher"]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.current_lambda = kd_weight(self.num_timesteps + n_rollout_steps * env.num_envs,
                                        self.lambda_start, self.anneal_steps)
        if self.current_lambda == 0:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if self.teacher is None:
            raise ValueError("RA4RL requires a retrieval teacher while lambda>0")
        collector = _RawCollector()
        collector.init_callback(self)
        joined = CallbackList([collector, callback])
        # Existing user callback was initialized by learn(); do not reinitialize it each rollout.
        joined.model = self
        ok = super().collect_rollouts(env, joined, rollout_buffer, n_rollout_steps)
        if ok:
            states = [state for step in collector.states for state in step]
            probs = self.teacher(states)
            rollout_buffer.set_teacher(probs.reshape(n_rollout_steps, env.num_envs, 7))
        return ok

    def train(self):
        if self.current_lambda == 0:
            super().train()
            self.logger.record("train/kd_loss", 0.0)
            self.logger.record("train/lambda", 0.0)
            return
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)
        losses = {k: [] for k in ("entropy_loss", "policy_gradient_loss", "value_loss", "clip_fraction",
                                  "kd_loss", "grad_norm")}
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - data.old_log_prob)
                policy_loss = -th.min(advantages * ratio,
                                      advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                values_pred = values if clip_range_vf is None else data.old_values + th.clamp(
                    values - data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(data.returns, values_pred)
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                log_student = self.policy.get_distribution(data.observations).distribution.logits
                kd_loss = forward_kl(data.teacher_probs, log_student)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = loss + self.current_lambda * kd_loss
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                if not th.isfinite(loss):
                    raise FloatingPointError("Nonfinite combined PPO/KD loss")
                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                for key, value in (("entropy_loss", entropy_loss.item()),
                                   ("policy_gradient_loss", policy_loss.item()),
                                   ("value_loss", value_loss.item()), ("clip_fraction", clip_fraction),
                                   ("kd_loss", kd_loss.item()), ("grad_norm", float(grad_norm))):
                    losses[key].append(value)
            self._n_updates += 1
            if not continue_training:
                break
        for key, values in losses.items():
            self.logger.record("train/" + key, np.mean(values))
        self.logger.record("train/lambda", self.current_lambda)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
