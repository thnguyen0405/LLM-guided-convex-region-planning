#!/usr/bin/env python3
"""Run the planners and show a native Matplotlib replay, or replay saved JSON."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from path_finder import MapGeometry, Planner, PlannerConfig
from path_finder.planners import PLANNERS
from path_finder.sampler import BiasSampler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "examples"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", type=Path, default=DATA / "flappy" / "map.json")
    p.add_argument("--replay", type=Path, help="Replay an existing planner JSON without rerunning the planner")
    p.add_argument("--regions", type=Path, help="Optional regions.json or graph.json for visualization")
    p.add_argument("--planner", choices=[*PLANNERS, "rrt_connect", "all"], default="brrt")
    p.add_argument("--sampling", choices=["uniform", "informed", "guild"], default="uniform")
    p.add_argument("--guidance", choices=["none", "corridor", "region_prior", "sequence_guided"], default="none")
    p.add_argument("--guidance-file", type=Path, help="Legacy C++ sampling-artifact schema, NOT raw scores.json")
    p.add_argument("--density-config", type=Path, help="Optional legacy density-decay JSON; disabled by default")
    p.add_argument("--clip-guidance-to-informed", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-time", type=float, default=3.0, help="Seconds; 0 uses iteration/node budgets only")
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--stop-on-first", action="store_true", help="Optional early stop; source continues until its budget")
    p.add_argument("--cost-mode", choices=["source", "geometric"], default="source")
    p.add_argument("--trace-limit", type=int, default=100000)
    p.add_argument("--output", type=Path, help="Output folder; defaults to outputs/<map name>")
    display = p.add_mutually_exclusive_group()
    display.add_argument("--show", "--open", dest="show", action="store_true",
                         help="Show native Matplotlib windows (default; --open is a compatibility alias)")
    display.add_argument("--no-show", dest="show", action="store_false",
                         help="Export JSON, PNG and SVG without opening windows")
    p.set_defaults(show=True)
    p.add_argument("--backend", choices=["MacOSX", "QtAgg", "TkAgg", "Agg"],
                   help="Optional Matplotlib backend override; --no-show always uses Agg")
    return p


def run(args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    elif args.backend:
        matplotlib.use(args.backend)
    from path_finder.visualization import show_replays, write_plots

    if args.output is None:
        if args.replay:
            args.output = args.replay.resolve().parent
        else:
            map_name = args.map.parent.name if args.map.stem == "map" else args.map.stem
            args.output = ROOT / "outputs" / map_name
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replay:
        result = json.loads(args.replay.read_text())
        if result.get("schema") != "python-sampling-planner/v1":
            raise ValueError("--replay requires a python-sampling-planner/v1 result JSON")
        name = result["config"]["planner"]
        if name not in PLANNERS:
            raise ValueError("Unknown planner in replay JSON")
        write_plots(result, args.output / name)
        print(f"Replaying recorded run: {args.replay.resolve()}")
        print(f"  Snapshot: {(args.output / name).with_suffix('.png').resolve()}")
        if args.show:
            show_replays([result])
        return 0 if result["success"] else 2

    geometry = MapGeometry.load(args.map)
    region_file = args.regions or args.map.with_name("regions.json")
    regions = []
    if region_file.exists():
        raw = json.loads(region_file.read_text())
        regions = raw.get("regions", raw.get("vertices", []))
    elif args.regions:
        raise ValueError(f"Region file does not exist: {region_file}")
    if args.guidance == "none" and args.guidance_file:
        raise ValueError("Use --guidance with --guidance-file")
    density = json.loads(args.density_config.read_text()) if args.density_config else None
    selected = PLANNERS if args.planner == "all" else ("brrt" if args.planner == "rrt_connect" else args.planner,)
    summaries, results = [], []
    for name in selected:
        # --all selects sampling variants only where the source implements them.
        sampling = args.sampling
        if args.planner == "all" and (name in ("rrt", "brrt") or (name == "brrt_star" and sampling == "guild")):
            sampling = "uniform"
        config = PlannerConfig(name, args.step, args.radius, args.search_time, args.max_nodes,
                               args.iterations, args.seed, sampling, args.stop_on_first,
                               args.cost_mode, args.trace_limit)
        sampler = BiasSampler.from_file(geometry, args.guidance_file, seed=args.seed, mode=sampling,
            guidance=args.guidance, clip_to_informed=args.clip_guidance_to_informed, density=density)
        result = Planner(geometry, config, sampler).plan()
        result["map"] = geometry.data
        result["map_name"] = geometry.data.get("name") or (
            args.map.parent.name if args.map.stem == "map" else args.map.stem
        ).replace("_", " ").title()
        result["regions"] = regions
        result["inputs"] = dict(map_file=str(args.map.resolve()),
            map_sha256=hashlib.sha256(args.map.read_bytes()).hexdigest(),
            regions_file=str(region_file.resolve()) if region_file.exists() else None,
            guidance_file=str(args.guidance_file.resolve()) if args.guidance_file else None)
        result["limitations"] = ["Planar disk robot only; no sensor or motor integration.",
            "No class-2 labels, event LLM, Bayes or Thompson sampling in this baseline port.",
            "Algorithm translation has not been certified bit-for-bit against the ROS executable.",
            "See PORTING_NOTES.md for source quirks and explicit changes."]
        file = args.output / name
        file.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_plots(result, file)
        results.append(result)
        summary = {k: result[k] for k in ("success", "path_valid", "iterations", "nodes_added",
            "planning_time", "time_to_first_solution", "algorithm_cost", "path_length", "stop_reason")}
        summary.update(planner=name, sampling=sampling, seed=args.seed)
        summaries.append(summary)
        print(f"{name:11s} success={result['success']} nodes={result['nodes_added']} "
              f"iterations={result['iterations']} time={result['planning_time']:.3f}s "
              f"length={result['path_length']}")
        print(f"  Snapshot: {file.with_suffix('.png').resolve()}")
        print(f"  Replay JSON: {file.with_suffix('.json').resolve()}")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    if args.show and results:
        show_replays(results)
    return 0 if all(r["success"] for r in summaries) else 2


def main():
    args = parser().parse_args()
    try:
        return run(args)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Install with: "
              "python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
