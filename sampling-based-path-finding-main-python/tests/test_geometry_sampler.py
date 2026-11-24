"""Independent geometry fixtures and sampler-domain/statistical checks."""

import math
import random
import unittest

from path_finder import MapGeometry
from path_finder.geometry import covered_segment, inside, point3
from path_finder.kdtree import KDTree
from path_finder.planners import Node
from path_finder.sampler import BiasSampler, informed_set


def rectangle():
    return MapGeometry(dict(boundary=[[0, 0], [10, 0], [10, 10], [0, 10]],
        obstacles=[], robot_radius=0.1, start=[1, 1], goal=[9, 9]))


def prior():
    return dict(start_region="L", goal_region="R", regions=[
        dict(id="L", score=0.8, polygon=[[0, 0], [5, 0], [5, 10], [0, 10]]),
        dict(id="R", score=0.2, polygon=[[5, 0], [10, 0], [10, 10], [5, 10]])])


class GeometryTests(unittest.TestCase):
    def test_full_segment_check_detects_thin_obstacle(self):
        m = MapGeometry(dict(boundary=[[0, 0], [10, 0], [10, 10], [0, 10]],
            obstacles=[[[4.999, 2], [5.001, 2], [5.001, 8], [4.999, 8]]],
            start=[1, 5], goal=[9, 5]))
        self.assertTrue(m.valid(m.start))
        self.assertTrue(m.valid(m.goal))
        self.assertFalse(m.segment_valid(m.start, m.goal))

    def test_concave_boundary_exit_and_reentry(self):
        poly = [[0, 0], [4, 0], [4, 4], [3, 4], [3, 1], [1, 1], [1, 4], [0, 4]]
        self.assertTrue(inside([0.5, 3], poly))
        self.assertTrue(inside([3.5, 3], poly))
        self.assertFalse(covered_segment([0.5, 3], [3.5, 3], poly))
        with self.assertRaises(ValueError):
            MapGeometry(dict(boundary=poly, start=[0.5, 0.5], goal=[3.5, 0.5],
                obstacles=[[[0.5, 2], [3.5, 2], [3.5, 3], [0.5, 3]]]))

    def test_clearance_checks_both_outer_ring_and_obstacles(self):
        m = rectangle()
        self.assertTrue(m.valid((0.1, 1, 0)))
        self.assertFalse(m.valid((0.09, 1, 0)))
        self.assertFalse(m.valid((0, 1, 0)))
        self.assertFalse(m.valid((math.nan, 1, 0)))
        self.assertTrue(m.segment_valid((0.1, 1, 0), (0.1, 9, 0)))

    def test_bad_polygon_and_negative_clearance_are_rejected(self):
        data = dict(boundary=[[0, 0], [10, 10], [0, 10], [10, 0]], start=[1, 2], goal=[2, 2])
        with self.assertRaises(ValueError):
            MapGeometry(data)
        data["boundary"] = [[0, 0], [10, 0], [10, 10], [0, 10]]
        data["robot_radius"] = -1
        with self.assertRaises(ValueError):
            MapGeometry(data)


class KDTreeTests(unittest.TestCase):
    def test_queries_match_brute_force_with_incremental_insertions(self):
        rng = random.Random(7)
        nodes, kd = [], KDTree()
        for i in range(250):
            n = Node(i, (rng.random(), rng.random(), 0), 0)
            nodes.append(n)
            kd.insert(n)
            if i % 5 == 0:
                p = (rng.random(), rng.random(), 0)
                nearest = min(nodes, key=lambda n: math.dist(n.x, p))
                self.assertIs(kd.nearest(p), nearest)
                self.assertEqual({n.id for n in kd.within(p, 0.2)},
                    {n.id for n in nodes if math.dist(n.x, p) <= 0.2})

    def test_deep_tree_does_not_hit_python_recursion_limit(self):
        kd = KDTree()
        for i in range(1500):
            kd.insert(Node(i, (float(i), 0., 0.), 0))
        self.assertEqual(kd.nearest((1499., 0., 0.)).id, 1499)


class SamplerTests(unittest.TestCase):
    def test_prior_region_frequency_and_uniform_triangle_mean(self):
        s = BiasSampler(rectangle(), guidance="region_prior", data=prior())
        points = [s.sample() for _ in range(5000)]
        self.assertTrue(all(0 <= p[0] <= 10 and 0 <= p[1] <= 10 for p in points))
        fraction = sum(p[0] < 5 for p in points) / len(points)
        self.assertAlmostEqual(fraction, 0.8, delta=0.025)
        self.assertAlmostEqual(sum(p[1] for p in points) / len(points), 5, delta=0.15)

    def test_corridor_is_area_weighted_not_score_weighted(self):
        data = prior()
        data["sequence"] = ["L", "R"]
        s = BiasSampler(rectangle(), guidance="corridor", data=data)
        fraction = sum(s.sample()[0] < 5 for _ in range(4000)) / 4000
        self.assertAlmostEqual(fraction, 0.5, delta=0.03)

    def test_sequence_portal_and_global_branches(self):
        data = prior()
        data.update(sequence=["L", "R"], guided_probability=1,
            portal_probability_within_guidance=1,
            portals=[dict(source="L", target="R", segment=[[5, 1], [5, 9]])])
        s = BiasSampler(rectangle(), guidance="sequence_guided", data=data)
        self.assertTrue(all(s.sample()[0] == 5 for _ in range(100)))
        data["guided_probability"] = 0
        s = BiasSampler(rectangle(), guidance="sequence_guided", data=data)
        s.sample()
        self.assertEqual(s.last_source, "uniform")

    def test_legacy_density_decay_is_opt_in_and_counts_insertions_only(self):
        s = BiasSampler(rectangle(), guidance="region_prior", data=prior(),
            density=dict(enabled=True, update_interval=2))
        s.begin_run(rectangle().goal)
        for _ in range(100):
            s.sample()
        self.assertEqual(s.nodes_added, 0)
        s.record_added_node((5, 2, 0), 1)  # Shared boundary assigned to lexical L.
        s.record_added_node((20, 20, 0), 2)  # Outside node advances epoch only.
        self.assertEqual(s.regions[0]["nodes"], 1)
        self.assertEqual(s.regions[1]["nodes"], 0)
        self.assertLess(s.regions[0]["score"], 0.8)
        self.assertEqual(len(s.density_log), 2)
        s.begin_run(rectangle().goal)
        self.assertEqual(s.regions[0]["score"], 0.8)

    def test_informed_samples_satisfy_focal_distance_bound(self):
        m = rectangle()
        s = BiasSampler(m, mode="informed")
        cost = 15
        s.ellipse = informed_set(cost, m.start, m.goal)
        for _ in range(1000):
            p = s.sample()
            self.assertLessEqual(math.dist(p, m.start) + math.dist(p, m.goal), cost + 1e-9)

    def test_clipped_prior_has_bounded_rejection(self):
        m = rectangle()
        s = BiasSampler(m, mode="informed", guidance="region_prior", data=prior(), clip_to_informed=True)
        s.ellipse = informed_set(0, (20, 20, 0), (20, 20, 0))
        self.assertTrue(all(math.isnan(x) for x in s.sample()))

    def test_bad_guidance_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            BiasSampler(rectangle(), guidance="region_prior")
        for mutate in (lambda d: d["regions"][0].update(score=8),
                       lambda d: d.update(start_region="missing"),
                       lambda d: d["regions"][1].update(id="L")):
            d = prior()
            mutate(d)
            with self.assertRaises(ValueError):
                BiasSampler(rectangle(), guidance="region_prior", data=d)


if __name__ == "__main__":
    unittest.main()
