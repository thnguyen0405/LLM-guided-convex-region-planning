# Sampling-based Path Finding — Python

2D RRT/RRT*/RRT#/BRRT/BRRT* with a Matplotlib demo. Requires Python 3.10+ and a desktop GUI.

From the project root:

```bash
cd "/Users/nguyen/BK/SEM7/DACN/DACN-main/sampling-based-path-finding-main-python"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py --planner brrt --stop-on-first
```

Default: Flappy, BRRT/RRT-Connect. Controls: Play/Pause, Next event, Reset, Final result and sliders.

- Replay: `python demo.py --replay outputs/flappy/brrt.json`
- Compare: `python demo.py --planner all --search-time 0 --iterations 1200 --no-show`
- Tests: `python -m unittest discover -s tests -v`
- Options: `python demo.py --help`

Maps: `flappy`, `hole`, `narrow`, `room`; choose with `--map examples/NAME/map.json`.
JSON/PNG/SVG results and summary: `outputs/<map>/`; use `--output DIR` to override.
Saved BRRT demos are available for all four maps. The heading shows the map name.

Baseline only: no online LLM, class-2 labels, Bayes/Thompson or robot control.
Region colors show tree-node presence. See [source notices](THIRD_PARTY_NOTICES.md) for upstream attribution.
