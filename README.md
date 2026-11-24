# Convex Region Graph

Python prepares convex-region graphs, top-k sequences and class-1 labels.
A small LLM through Ollama directly assigns each region/connection one query-specific
priority score in [0,1] with a short explanation. There is no separate label-weight stage.

Maps: `flappy`, `hole`, `narrow`, `room`.

## Scoring

```bash
cd /Users/nguyen/BK/SEM7/DACN/DACN-main/convex-region-graph
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Prepare labels without calling Ollama:

```bash
python score_maps.py --map flappy --prepare-only
```

Start [Ollama](https://ollama.com/) with an installed model, then score or resume:

```bash
python score_maps.py --map flappy --model qwen3:4b-instruct --resume
```

Outputs in `convex-region-graph/outputs/<map>/`:

- `query_labeled_data_llm_01.json`: graph features, sequences and Python labels.
- `query_priority_llm_01.json`: LLM priority scores and explanations, created when scoring runs.

Replace `flappy` with another map. Python validates and saves model scores unchanged.
Definitions and prompt: [scoring details](convex-region-graph/QUERY_SCORING.md).

## Planner demo

```bash
cd /Users/nguyen/BK/SEM7/DACN/DACN-main/sampling-based-path-finding-main-python
source .venv/bin/activate
python demo.py --replay outputs/flappy/brrt.json
```

Each map has one saved BRRT/RRT-Connect demo under `outputs/<map>/`:
`brrt.json`, `brrt.png`, `brrt.svg`, `summary.json`.
See [demo setup and commands](sampling-based-path-finding-main-python/README.md).

## Status

The planner demo runs separately from LLM scoring. Class-2 labels, online
Bayes/Thompson guidance, fine-tuning and robot integration are not implemented.
Some memberships remain unavailable until the required inscribed geometry or
short-portal calibration is supplied. Scores do not certify collision-free paths
and never remove regions/connections. Original ROS/C++ sources remain available.
