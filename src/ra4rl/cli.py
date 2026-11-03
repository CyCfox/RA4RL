import argparse
import asyncio

from .config import ENV_IDS, load_config


def main():
    parser = argparse.ArgumentParser(description="RA4RL: defaults to small smoke profile")
    parser.add_argument("command", choices=["collect", "index", "train", "smoke", "analyze", "plot"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--env-id", choices=ENV_IDS)
    parser.add_argument("--formal", action="store_true", help="Explicitly select full experiment parameters")
    parser.add_argument("--algorithm", choices=["ppo", "ra4rl"], default="ppo")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db")
    parser.add_argument("--index")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args()
    cfg = load_config(args.config, args.env_id, smoke=not args.formal)
    if args.command == "collect":
        if not args.formal:
            parser.error("Live LLM collection is explicit: use --formal; use smoke for offline fixtures")
        from .data.collect import collect
        asyncio.run(collect(cfg, args.output))
    elif args.command == "index":
        from .retrieval.index import build_index
        build_index(args.db, args.output, cfg, args.device)
    elif args.command == "train":
        from .training import train
        train(cfg, args.algorithm, args.output, args.db, args.index, args.device, args.resume)
    elif args.command == "smoke":
        if args.formal:
            parser.error("smoke cannot use --formal")
        from .smoke import run_smoke
        run_smoke(args.output, args.device)
    elif args.command == "plot":
        from .plotting import plot_runs
        plot_runs(args.runs or [], args.output, args.labels)
    else:
        from .analysis import analyze
        analyze(args.runs, args.output)


if __name__ == "__main__":
    main()
