"""Behavioural checks for source quirks, path safety, budgets and rewiring."""

import json
import math
from pathlib import Path
import unittest

from path_finder import MapGeometry, Planner, PlannerConfig
from path_finder.geometry import distance
from path_finder.planners import Node, PLANNERS
from path_finder.sampler import BiasSampler


ROOT = Path(__file__).resolve().parents[1]


def empty_map(start=(1, 1), goal=(9, 1)):
    return MapGeometry(dict(boundary=[[0, 0], [10, 0], [10, 10], [0, 10]],
                            obstacles=[], start=start, goal=goal, robot_radius=0.1))


class PreservedSampler(BiasSampler):
    """Supply a fixed sample stream to isolate the source planner's decisions."""
    def __init__(self, geometry, samples):
        super().__init__(geometry)
        self.samples = samples
        self.index = 0

    def sample(self):
        result = self.samples[self.index % len(self.samples)]
        self.index += 1
        return result


class PlannerTests(unittest.TestCase):
    def config(self, name="brrt", **kwargs):
        return PlannerConfig(planner=name, search_time=0, max_iterations=600, **kwargs)

    def test_all_source_planners_solve_flappy_and_validate_every_segment(self):
        geometry = MapGeometry.load(ROOT / "examples/flappy/map.json")
        for name in PLANNERS:
            with self.subTest(planner=name):
                result = Planner(geometry, self.config(name)).plan()
                self.assertTrue(result["success"])
                self.assertEqual(tuple(result["path"][0]), geometry.start)
                self.assertEqual(tuple(result["path"][-1]), geometry.goal)
                self.assertTrue(geometry.path_valid(result["path"]))
                costs = [s["cost"] for s in result["solutions"]]
                self.assertTrue(all(b < a for a, b in zip(costs, costs[1:])))
                nodes = {n["id"]: n for n in result["trees"]}
                for n in nodes.values():
                    if n["parent"] is not None:
                        parent = nodes[n["parent"]]
                        self.assertTrue(geometry.segment_valid(parent["point"], n["point"]))
                        self.assertEqual(n["tree"], parent["tree"])
                if name in ("rrt_star", "rrt_sharp"):
                    self.assertAlmostEqual(result["algorithm_cost"], result["path_length"])
                    self.assertGreater(result["rewires"], 1)

    def test_source_rrt_short_step_cost_is_not_silently_changed(self):
        m = empty_map()
        config = PlannerConfig(planner="rrt", search_time=0, max_iterations=1, search_radius=10)
        result = Planner(m, config, PreservedSampler(m, [(1.5, 1, 0)])).plan()
        self.assertAlmostEqual(result["algorithm_cost"], 9.5)  # 2.0 + 7.5
        self.assertAlmostEqual(result["path_length"], 8.0)
        config.cost_mode = "geometric"
        corrected = Planner(m, config, PreservedSampler(m, [(1.5, 1, 0)])).plan()
        self.assertAlmostEqual(corrected["algorithm_cost"], 8.0)

    def test_brrt_reaches_narrow_and_room_with_a_larger_budget(self):
        for name in ("narrow", "room"):
            with self.subTest(map=name):
                m = MapGeometry.load(ROOT / "examples" / name / "map.json")
                result = Planner(m, PlannerConfig(search_time=0, max_iterations=20000,
                    stop_on_first=True)).plan()
                self.assertTrue(result["success"])
                self.assertTrue(m.path_valid(result["path"]))

    def test_source_brrt_connect_keeps_final_bridge_without_target_duplicate(self):
        m = empty_map()
        config = PlannerConfig(search_time=0, max_iterations=1)
        result = Planner(m, config, PreservedSampler(m, [(3, 1, 0)])).plan()
        self.assertEqual(result["path"], [(1., 1., 0.), (3, 1, 0), (5., 1., 0.), (7., 1., 0.), (9., 1., 0.)])
        self.assertEqual(result["nodes_added"], 3)
        self.assertEqual(result["algorithm_cost"], 8)

    def test_bidirectional_tree_identity_survives_swaps(self):
        m = empty_map()
        result = Planner(m, self.config()).plan()
        self.assertTrue(any(e["kind"] == "swap" for e in result["events"]))
        nodes = {n["id"]: n for n in result["trees"]}
        for n in nodes.values():
            root = n
            while root["parent"] is not None:
                root = nodes[root["parent"]]
            self.assertEqual(root["id"], 1 if n["tree"] == 0 else 0)

    def test_source_continues_after_first_solution_by_default(self):
        result = Planner(empty_map(), self.config()).plan()
        self.assertEqual(result["iterations"], 600)
        result = Planner(empty_map(), self.config(stop_on_first=True)).plan()
        self.assertEqual(result["stop_reason"], "first_solution")
        self.assertLess(result["iterations"], 600)

    def test_disconnected_map_returns_failure_with_no_fake_path(self):
        m = MapGeometry(dict(boundary=[[0, 0], [10, 0], [10, 10], [0, 10]],
            obstacles=[[[4, 0], [6, 0], [6, 10], [4, 10]]], start=[1, 1], goal=[9, 9]))
        for name in PLANNERS:
            result = Planner(m, PlannerConfig(planner=name, search_time=0, max_iterations=120)).plan()
            self.assertFalse(result["success"])
            self.assertEqual(result["path"], [])
            self.assertIsNone(result["path_length"])

    def test_informed_and_guild_variants_run(self):
        m = empty_map()
        for name in ("rrt_star", "rrt_sharp", "brrt_star"):
            modes = ("informed", "guild") if name != "brrt_star" else ("informed",)
            for mode in modes:
                with self.subTest(planner=name, sampling=mode):
                    result = Planner(m, PlannerConfig(planner=name, sampling=mode,
                        search_time=0, max_iterations=100)).plan()
                    self.assertTrue(result["success"])
                    self.assertTrue(m.path_valid(result["path"]))

    def test_reparent_propagates_cost_to_descendants(self):
        p = Planner(empty_map(), self.config())
        p.iteration, p.events, p.dropped_events, p.rewires = 1, [], 0, 0
        a = Node(1, (1, 1, 0), 0, cost=0)
        b = Node(2, (3, 1, 0), 0, a, 6, 6)
        c = Node(3, (4, 1, 0), 0, b, 7, 1)
        a.children, b.children = [b], [c]
        p.change_parent(b, a, 2)
        self.assertEqual(b.cost, 2)
        self.assertEqual(c.cost, 3)
        self.assertEqual(a.children, [b])

    def test_same_start_goal_and_node_budget(self):
        m = empty_map(goal=(1, 1))
        result = Planner(m, self.config()).plan()
        self.assertEqual(result["path"], [m.start])
        self.assertEqual(result["path_length"], 0)
        result = Planner(empty_map(), self.config(max_nodes=2)).plan()
        self.assertEqual(result["nodes_added"], 0)
        self.assertEqual(result["stop_reason"], "node_budget")

    def test_seed_reproducibility_with_iteration_budget(self):
        m = empty_map()
        a = Planner(m, self.config()).plan()
        b = Planner(m, self.config()).plan()
        self.assertEqual(a["path"], b["path"])
        self.assertEqual(a["trees"], b["trees"])
        self.assertEqual(a["events"], b["events"])

    def test_nonfinite_rejection_trace_is_valid_json(self):
        m = empty_map()
        result = Planner(m, PlannerConfig(search_time=0, max_iterations=1),
            PreservedSampler(m, [(math.nan, math.nan, math.nan)])).plan()
        json.dumps(result, allow_nan=False)
        self.assertFalse(result["success"])

    def test_trace_limit_is_explicit(self):
        result = Planner(empty_map(), self.config(trace_limit=3)).plan()
        self.assertEqual(len(result["events"]), 3)
        self.assertGreater(result["dropped_events"], 0)

    def test_invalid_configuration(self):
        for kwargs in (dict(steer_length=0), dict(search_time=math.nan),
                       dict(max_nodes=1), dict(planner="rrt", sampling="informed")):
            with self.assertRaises(ValueError):
                PlannerConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
