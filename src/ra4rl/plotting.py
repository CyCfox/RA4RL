"""Publication-style curves from recorded fixed-scenario evaluations.

Bands describe episode dispersion within each evaluation, never confidence
intervals or variability across training seeds. No smoothing or synthetic points.
"""
import csv
import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from .data.store import sha256
from .evaluation_log import episode_count

PANELS = (
    ("success_rate", "Success Rate"),
    ("return", "Mean Episode Return"),
    ("episode_steps", "Mean Episode Length (steps)"),
)


def load_curve(run):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    records = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No evaluations to plot: {run}")
    steps = np.array([r["train_env_steps"] for r in records], dtype=np.int64)
    if (steps <= 0).any() or (np.diff(steps) <= 0).any():
        raise ValueError("Evaluation steps must be positive and strictly increasing")
    for key, _ in PANELS:
        means = np.array([r[key + "_mean"] for r in records])
        stds = np.array([r[key + "_std"] for r in records])
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds < 0).any():
            raise ValueError(f"Invalid plotted statistics: {key}")
    return dict(run=run, cfg=cfg, manifest=manifest, records=records, steps=steps)


def plot_runs(runs, output=None, labels=None):
    curves = [load_curve(run) for run in runs]
    if not curves:
        raise ValueError("At least one run is required")
    if labels is not None and len(labels) != len(curves):
        raise ValueError("One label is required per run")
    # Overlay only comparable environments, episode limits, and evaluation scenes.
    protocol = [(c["cfg"]["env_id"], c["cfg"]["max_episode_steps"], c["cfg"]["eval_seeds"])
                for c in curves]
    if any(item != protocol[0] for item in protocol):
        raise ValueError("Cannot overlay different environments, horizons, or evaluation seeds")
    if output is None:
        if len(curves) != 1:
            raise ValueError("Specify output for a comparison plot")
        output = curves[0]["run"] / "plots"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = [f"{c['manifest']['algorithm'].upper()} (seed {c['cfg']['train_seed']})" for c in curves]
    colors = ["#347ff0", "#ff7f19", "#9c44f5", "#159b76"]
    max_steps = max(int(c["steps"][-1]) for c in curves)
    max_episode = max(c["cfg"]["max_episode_steps"] for c in curves)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 12, "axes.linewidth": 1.0,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig = Figure(figsize=(14.5, 4.25))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        for axis, (key, ylabel) in zip(axes, PANELS):
            cap = max_episode if key == "episode_steps" else 1.0
            for i, (curve, label) in enumerate(zip(curves, labels)):
                records = curve["records"]
                means = np.array([r[key + "_mean"] for r in records])
                stds = np.array([r[key + "_std"] for r in records])
                color = colors[i % len(colors)]
                axis.fill_between(curve["steps"], np.clip(means - stds, 0, cap),
                                  np.clip(means + stds, 0, cap), color=color, alpha=.16, linewidth=0)
                axis.plot(curve["steps"], means, color=color, linewidth=1.8, label=label,
                          marker="o" if len(curve["steps"]) == 1 else None, markersize=4)
            axis.set(xlabel="Environment Steps", ylabel=ylabel, xlim=(0, max_steps),
                     ylim=(0, cap * (1.02 if key != "episode_steps" else 1.04)))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
            axis.tick_params(direction="out")
            axis.grid(alpha=.15, linewidth=.5)
        fig.suptitle(curves[0]["cfg"]["env_id"], fontsize=12, y=.98)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .045), ncol=len(curves),
                   frameon=True, fancybox=False)
        fig.text(.5, .015, "Shading: ±1 population SD across evaluation episodes (not across training seeds)",
                 ha="center", va="bottom", fontsize=8, color="#555555")
        fig.subplots_adjust(left=.06, right=.985, bottom=.25, top=.88, wspace=.3)
        paths = []
        for extension in ("png", "pdf", "svg"):
            path = output / f"training_curves.{extension}"
            fig.savefig(path, dpi=240, facecolor="white")
            paths.append(str(path))
        fig.clear()
    table = output / "curve_data.csv"
    fields = ["run", "algorithm", "train_seed", "train_env_steps", "evaluation_episodes"]
    fields += [key + suffix for key, _ in PANELS for suffix in ("_mean", "_std")]
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            for row in curve["records"]:
                writer.writerow(dict(run=str(curve["run"]), algorithm=curve["manifest"]["algorithm"],
                                     train_seed=curve["cfg"]["train_seed"], train_env_steps=row["train_env_steps"],
                                     evaluation_episodes=episode_count(row),
                                     **{k: row[k] for k in fields[5:]}))
    metadata = dict(band="mean +/- 1 population SD (ddof=0) across evaluation episodes; not a CI or seed SD",
                    band_clipped_to_metric_bounds=True, smoothing="none", synthetic_points=False,
                    x_axis="actual train_env_steps", labels=labels,
                    runs=[dict(path=str(c["run"]), evaluations_sha256=sha256(c["run"] / "evaluations.jsonl"),
                               points=len(c["records"]), final_step=int(c["steps"][-1])) for c in curves])
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2))
    return paths + [str(table), str(output / "plot_metadata.json")]
