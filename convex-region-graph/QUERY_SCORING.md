# Query-specific LLM priority scoring

Python computes graph features, top-k region sequences and class-1 memberships.
The LLM receives the retained graph, start/goal, sequences, features and labels,
and directly returns one priority score in [0,1] with a short reason per region
or undirected connection. No separate per-label weights are generated.
Python only validates and stores the model score; it does not calculate or normalize it.

## Run

From `convex-region-graph/`, activate `.venv` and run:

```bash
python score_maps.py --map flappy --prepare-only
python score_maps.py --map flappy --resume
```

Maps: `flappy`, `hole`, `narrow`, `room`. For inference, start Ollama and use an
installed model name via `--model` (default `qwen3:4b-instruct`).
Options include `--graph`, `--config`, `--k`, `--prompt`, `--rubric`,
`--prepared-output`, `--output`, `--base-url` and `--num-ctx` (default 32768).
Large graphs may require more context and device memory.

Outputs: `outputs/<map>/query_labeled_data_llm_01.json` and, after inference,
`query_priority_llm_01.json`. Existing graph exports may also be retained.
Old experimental outputs were removed during cleanup; no scores were converted.

## Sequences and labels

Only connections retained after clearance filtering are used. Each directed
edge cost is the distance from the first region centroid to the safe-portal
midpoint plus the distance from that midpoint to the second centroid.
Yen/Dijkstra searches simple paths for all containing start/goal-region pairs
and returns a global top-k (default 3), or fewer when fewer exist. A shared
endpoint region allows a one-region, zero-cost sequence. No connection produces
an explicit disconnected status. Sequences are graph routes, not certified trajectories.

Membership thresholds and activation rules are in `structural_scoring_config.json`.
Increasing membership is sigmoid((feature-threshold)/softness); decreasing is
one minus that value. Active defaults to >=0.5 and very to >=0.9.
Missing memberships are null with a reason; known zero is inactive, not missing.

| Label | Python definition |
|---|---|
| open | Region inscribed-ball radius / robot radius; increasing (5, 1) |
| corridor | Degree 2 and inscribed-ellipsoid axis ratio; increasing (3, 0.5) |
| junction | Degree >=3 and fraction of actual returned sequences visiting the region; increasing (0.3, 0.1) |
| dead_end | Degree 1 excluding every endpoint-containing region |
| pocket | Endpoint-free side of a bridge in a query-endpoint component |
| narrow | Full-dimensional intersection inradius / robot radius; decreasing (2, 0.3) |
| portal_1d | Region intersection dimension is 1 under configured tolerances |
| short_safe_portal | Safe-segment length decreasing membership for 1D intersections; calibration defaults null |
| critical | Actual returned-sequence connection fraction; increasing (0.8, 0.05) |
| alternative_route | On a simple start-goal route after deleting the best sequence's undirected edges |

Pocket and alternative-route tests require reciprocal edges. Alternative routes
use a block-cut forest to exclude branches that only appear reachable.
Top-k frequency is not all-path necessity: mandatory connection flags use a
separate edge-removal reachability test. Fractions divide by actual returned count.

Exact region inradii, inscribed ellipsoid ratios and full-dimensional intersection
inradii are used only when supplied. Covariance ratio, mean clearance and portal
half-length are not substitutes. A 1D portal has no positive 2D inradius.
`short_safe_portal` requires explicit positive threshold and softness in map units;
no calibration is invented. A point robot has undefined radius-normalized widths.

## Scoring contract

`query_priority_rubric.json` defines the common anchors: 0 lowest priority,
0.5 neutral/mixed, 1 highest priority. These are not probabilities or feasibility
judgments. `query_scoring_prompt.txt` asks the model to explain each score using
actual graph context and supplied labels. Same-label targets may score differently.
There is no fixed additive formula or numeric bonus/penalty system.

Results use `query-priority-scores/v4`, with `score_source: llm_direct_query_priority`.
Each response contains exactly status, score and reason (at most 320 characters).
Python rejects booleans, non-finite/out-of-range scores, malformed responses and
truncated responses. Insufficient evidence yields null. Disconnected queries
receive null without model calls. Checkpoints save after each target; resume
requires matching input, prompt, rubric, model name, server and inference settings.
Model tags are mutable; changing the installed model warrants a fresh output.

Review flags are heuristic text checks, not factual verification, and never
modify scores or labels. Model score quality still requires evaluation.
No low score removes graph elements or overrides collision checks. No class-2
state, online feedback, Bayesian/Thompson selection or planner integration is added.

## Tests

```bash
python -m unittest discover -v
```
