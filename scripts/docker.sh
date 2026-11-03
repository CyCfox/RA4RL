#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .runtime/cache outputs data
task_runtime_image="${RA4RL_IMAGE:-pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime}"
task_runtime_image_id="$(docker image inspect "$task_runtime_image" --format '{{.Id}}')"
task_docker_flags=()
if [[ "${RA4RL_DETACH:-0}" == "1" ]]; then
  task_docker_flags+=(--detach)
fi
exec docker run "${task_docker_flags[@]}" --rm --gpus "device=${RA4RL_GPU:-0}" \
  -u "$(id -u):$(id -g)" --name "ra4rl_${RA4RL_JOB:-dev}_cyc" \
  -e HOME=/workspace/my_project/.runtime \
  -e PYTHONPATH=/workspace/my_project/src \
  -e PATH=/workspace/my_project/.runtime/venv/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e ANONYMIZED_TELEMETRY=False \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e SDL_VIDEODRIVER=dummy \
  -e OPENCODE_API_KEY \
  -e RA4RL_IMAGE_ID="$task_runtime_image_id" -e RA4RL_IMAGE="$task_runtime_image" \
  -v "$PWD:/workspace/my_project" -v "${RA4RL_MODELS:-/media/data/chenyuchen/models}:/workspace/models:ro" \
  -w /workspace/my_project \
  "$task_runtime_image" "$@"
