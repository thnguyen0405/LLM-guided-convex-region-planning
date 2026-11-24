"""BiasSampler port: uniform, informed, GUILD and legacy JSON guidance.

Derived from the sibling sampler.h (Hongkai Ye, 2022); see notices.
No LLM inference, class-2 labels or Bayes/Thompson logic is introduced here.
"""

import json
import math
import random
from pathlib import Path

from .geometry import cross, distance, edges, inside, lerp, point3, polygon_area, read_polygon


def informed_set(cost, a, b):
    center = lerp(a, b, 0.5)
    d = distance(a, b)
    u = tuple((y - x) / d for x, y in zip(a, b)) if d else (1.0, 0.0, 0.0)
    # Same planar frame as the source calInformedSet, including degenerate foci.
    v = (-u[1], u[0], 0.0)
    radii = (cost / 2, math.sqrt(max(0.0, cost * cost - d * d)) / 2)
    return center, u, v, radii


class BiasSampler:
    def __init__(self, geometry, seed=42, mode="uniform", guidance="none", data=None,
                 clip_to_informed=False, density=None):
        if mode not in ("uniform", "informed", "guild"):
            raise ValueError("Unknown sampling mode")
        if guidance not in ("none", "corridor", "region_prior", "sequence_guided"):
            raise ValueError("Unknown guidance mode")
        self.geometry, self.mode, self.guidance = geometry, mode, guidance
        self.rng = random.Random(seed)
        self.ellipse, self.guild_sets = None, None
        self.clip = clip_to_informed
        self.regions, self.portals = [], []
        self.guided_probability, self.portal_probability = 0.8, 0.2
        self.nodes_added, self.density_log = 0, []
        self.density = dict(density or {})
        self.protected = set(self.density.get("protected_region_ids", []))
        self._validate_density()
        self.last_source = "uniform"
        if guidance != "none":
            if data is None:
                raise ValueError("Guidance requires an explicit valid JSON file; no silent fallback")
            self._load(data)

    @classmethod
    def from_file(cls, geometry, path=None, **kwargs):
        data = json.loads(Path(path).read_text()) if path else None
        return cls(geometry, data=data, **kwargs)

    def _load(self, data):
        seen = set()
        for raw in data["regions"]:
            rid = raw["id"]
            if not isinstance(rid, str) or not rid or rid in seen:
                raise ValueError("Guidance IDs must be nonempty and unique")
            seen.add(rid)
            poly = read_polygon(raw["polygon"])
            signs = [cross(a, b, poly[(i + 2) % len(poly)]) for i, (a, b) in enumerate(edges(poly))]
            if any(x > 1e-12 for x in signs) and any(x < -1e-12 for x in signs):
                raise ValueError("Guidance polygon must be convex and ordered")
            score = float(raw.get("score", 1))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Legacy guidance scores must be in [0,1], not integer 0-10")
            self.regions.append(dict(id=rid, polygon=poly, score=score, base=score,
                                     area=polygon_area(poly), nodes=0))
        if not self.regions or not any(r["score"] > 0 for r in self.regions):
            raise ValueError("Guidance needs regions and a positive score")
        if self.guidance in ("corridor", "sequence_guided"):
            sequence = data["sequence"]
            if sequence != [r["id"] for r in self.regions]:
                raise ValueError("Regions must match sequence one-for-one in order")
        if self.guidance in ("region_prior", "sequence_guided"):
            if data["start_region"] not in seen or data["goal_region"] not in seen:
                raise ValueError("Unknown guidance endpoint region")
        if self.guidance == "sequence_guided":
            if data["start_region"] != sequence[0] or data["goal_region"] != sequence[-1]:
                raise ValueError("Sequence endpoint IDs do not match")
            self.guided_probability = float(data.get("guided_probability", 0.8))
            self.portal_probability = float(data.get("portal_probability_within_guidance", 0.2))
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (self.guided_probability, self.portal_probability)):
                raise ValueError("Guidance probabilities must be in [0,1]")
            for i, p in enumerate(data.get("portals", [])):
                if i + 1 >= len(sequence) or (p["source"], p["target"]) != (sequence[i], sequence[i + 1]):
                    raise ValueError("Portal must join consecutive sequence regions")
                segment = p["segment"]
                if len(segment) != 2 or any(len(x) != 2 for x in segment):
                    raise ValueError("Portal segment needs two [x,y] endpoints")
                a, b = map(point3, segment)
                if distance(a, b) <= 1e-12:
                    raise ValueError("Portal must have positive length")
                self.portals.append((a, b))
            if self.portals and len(self.portals) != len(sequence) - 1:
                raise ValueError("Portal count must match sequence")
        self.validate_endpoints(self.geometry.start, self.geometry.goal)

    def validate_endpoints(self, start, goal):
        if self.guidance in ("corridor", "sequence_guided"):
            ok = inside(start, self.regions[0]["polygon"]) and inside(goal, self.regions[-1]["polygon"])
        elif self.guidance == "region_prior":
            ok = all(any(inside(p, r["polygon"]) for r in self.regions) for p in (start, goal))
        else:
            return
        if not ok:
            raise ValueError("Start/goal does not match guidance polygons")

    def _validate_density(self):
        defaults = dict(enabled=False, update_interval=100, lambda_=0.4,
                        protected_lambda=0.1, rho_ref=2.0, min_score=0.05)
        defaults.update(self.density)
        self.density = defaults
        d = self.density
        if (not isinstance(d["update_interval"], int) or d["update_interval"] <= 0
                or any(not math.isfinite(d[k]) or d[k] < 0 for k in ("lambda_", "protected_lambda"))
                or not math.isfinite(d["rho_ref"]) or d["rho_ref"] <= 0
                or not math.isfinite(d["min_score"]) or not 0 < d["min_score"] <= 1):
            raise ValueError("Invalid legacy density-decay configuration")

    def begin_run(self, goal):
        self.ellipse, self.guild_sets = None, None
        self.nodes_added, self.density_log = 0, []
        self.protected = set(self.density.get("protected_region_ids", []))
        for r in self.regions:
            r["nodes"], r["score"] = 0, r["base"]
            if inside(goal, r["polygon"]):
                self.protected.add(r["id"])

    def record_added_node(self, state, iteration):
        if not self.density["enabled"] or self.guidance != "region_prior":
            return
        self.nodes_added += 1
        containing = [r for r in self.regions if inside(state, r["polygon"])]
        if containing:
            min(containing, key=lambda r: r["id"])["nodes"] += 1
        if self.nodes_added % self.density["update_interval"]:
            return
        before = sum(r["score"] for r in self.regions)
        probabilities = [r["score"] / before for r in self.regions]
        d = self.density
        for r in self.regions:
            density = r["nodes"] / r["area"]
            penalty = d["protected_lambda"] if r["id"] in self.protected else d["lambda_"]
            r["score"] = max(d["min_score"], r["base"] - penalty * density / (density + d["rho_ref"]))
        after = sum(r["score"] for r in self.regions)
        for r, p in zip(self.regions, probabilities):
            self.density_log.append(dict(iteration=iteration, nodes_added=self.nodes_added,
                region_id=r["id"], region_nodes=r["nodes"], area=r["area"],
                density=r["nodes"] / r["area"], base_score=r["base"], adjusted_score=r["score"],
                probability_before=p, probability_after=r["score"] / after))

    def uniform(self):
        g = self.geometry
        return (self.rng.uniform(g.minimum[0], g.maximum[0]),
                self.rng.uniform(g.minimum[1], g.maximum[1]), g.start[2])

    def _polygon_sample(self, region):
        poly = region["polygon"]
        triangles = [(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1)]
        tri = self.rng.choices(triangles, weights=[abs(cross(*t)) / 2 for t in triangles])[0]
        root, v = math.sqrt(self.rng.random()), self.rng.random()
        x, y = (sum(w * p[k] for w, p in zip((1 - root, root * (1 - v), root * v), tri)) for k in (0, 1))
        return x, y, self.geometry.start[2]

    def _prior(self):
        return self._polygon_sample(self.rng.choices(self.regions, weights=[r["score"] for r in self.regions])[0])

    def _ball(self):
        p = [self.rng.gauss(0, 1) for _ in range(3)]
        norm = math.sqrt(sum(x * x for x in p))
        if norm == 0:
            return self._ball()
        # Preserve source's 3-ball draw then planar projection, NOT uniform 2-disk.
        r = self.rng.random() ** 0.33333
        return tuple(r * x / norm for x in p)

    def _transform(self, p, ellipse):
        center, u, v, radii = ellipse
        return tuple(center[k] + radii[0] * p[0] * u[k] + radii[1] * p[1] * v[k] for k in range(3))

    def _inside_ellipse(self, p):
        center, u, v, radii = self.ellipse
        delta = tuple(x - y for x, y in zip(p, center))
        coords = [sum(x * y for x, y in zip(delta, axis)) for axis in (u, v)]
        total = 0.0
        for x, radius in zip(coords, radii):
            if radius <= 1e-12:
                if abs(x) > 1e-10:
                    return False
            else:
                total += (x / radius) ** 2
        return total <= 1 + 1e-12

    def sample(self):
        self.last_source = self.guidance if self.guidance != "none" else self.mode
        if self.guidance == "sequence_guided":
            if self.rng.random() >= self.guided_probability:
                self.last_source = "uniform"
                return self.uniform()
            if self.portals and self.rng.random() < self.portal_probability:
                self.last_source = "portal"
                a, b = self.rng.choice(self.portals)
                return (*lerp(a, b, self.rng.random())[:2], self.geometry.start[2])
            return self._prior()
        if self.guidance == "corridor":
            region = self.rng.choices(self.regions, weights=[r["area"] for r in self.regions])[0]
            return self._polygon_sample(region)
        if self.guidance == "region_prior":
            for _ in range(256):
                p = self._prior()
                if not self.clip or self.ellipse is None or self._inside_ellipse(p):
                    return p
            return (math.nan, math.nan, math.nan)
        if self.mode == "informed" and self.ellipse is not None:
            return self._transform(self._ball(), self.ellipse)
        if self.mode == "guild" and self.guild_sets is not None:
            p = self._ball()
            return self._transform(p, self.guild_sets[0 if p[0] > 0 else 1])
        self.last_source = "uniform"
        return self.uniform()

    def update_solution(self, cost, start, goal, nodes):
        if self.mode == "informed":
            self.ellipse = informed_set(cost, start.x, goal.x)
        elif self.mode == "guild":
            # The source rejection loop can be infinite on a straight solution.
            candidates = [n for n in nodes if math.isfinite(n.cost)
                and not (n.cost == distance(start.x, n.x)
                         and n.cost + distance(n.x, goal.x) >= goal.cost)]
            if candidates:
                beacon = self.rng.choice(candidates)
                self.guild_sets = (informed_set(beacon.cost, start.x, beacon.x),
                                   informed_set(max(0, goal.cost - beacon.cost), beacon.x, goal.x))
            else:
                self.guild_sets = (informed_set(cost, start.x, goal.x),) * 2
