"""Geometry dimensionality and separately calibrated safe-segment labels."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from build_graph import build_graph
from prepare_scoring_data import load_config, prepare
from score_maps import context_for, explanation_review_flags, score_prepared


ROOT = Path(__file__).resolve().parents[1]


def adjacent_regions():
    graph = build_graph({'regions': [
        {'id': 'A', 'polygon': [[0, 0], [1, 0], [1, 2], [0, 2]]},
        {'id': 'B', 'polygon': [[1, 0], [2, 0], [2, 2], [1, 2]]},
    ]}, required_clearance=0.15)
    graph['planning'] = {'start_regions': ['A'], 'goal_regions': ['B'], 'robot_radius': 0.1}
    return graph


class PortalLabelTests(unittest.TestCase):
    def test_1d_type_is_computed_without_inventing_shortness_or_inradius(self):
        graph = adjacent_regions()
        before = deepcopy(graph)
        result = prepare(graph, load_config())
        edge = result['connections'][0]
        self.assertEqual(edge['structural_features']['intersection_dimension'], 1)
        self.assertAlmostEqual(edge['structural_features']['safe_portal_length'], 1.7)
        self.assertEqual(edge['memberships']['portal_1d']['value'], 1)
        self.assertIn('portal_1d', edge['labels'])
        self.assertIsNone(edge['memberships']['short_safe_portal']['value'])
        self.assertIsNone(edge['memberships']['narrow']['value'])
        self.assertIsNone(edge['structural_features']['intersection_inscribed_ball_radius'])
        self.assertEqual(graph, before)

    def test_explicit_shortness_configuration_uses_actual_safe_segment_length(self):
        graph, config = adjacent_regions(), load_config()
        # Test-only calibration numbers; these are NOT production defaults.
        config['label_rules']['short_safe_portal'].update(threshold=1.7, softness=0.2)
        baseline = prepare(graph, config)
        self.assertAlmostEqual(baseline['connections'][0]['memberships']['short_safe_portal']['value'], 0.5)
        for e in graph['directed_edges']:
            e['safe_portal'] = [[1, 0.9], [1, 1.1]]
            e['safe_portal_width'] = 999  # The feature is derived from coordinates.
        short = prepare(graph, config)
        self.assertGreater(short['connections'][0]['memberships']['short_safe_portal']['value'], 0.99)
        self.assertAlmostEqual(short['connections'][0]['structural_features']['safe_portal_length'], 0.2)
        config['label_rules']['short_safe_portal'].update(threshold=0.1)
        self.assertFalse(prepare(graph, config)['connections'][0]['memberships']['short_safe_portal']['active'])

    def test_overlap_is_2d_even_when_safe_portal_is_a_segment(self):
        graph, config = adjacent_regions(), load_config()
        graph['vertices'][1]['polygon'] = [[0.5, 0], [2, 0], [2, 2], [0.5, 2]]
        config['label_rules']['short_safe_portal'].update(threshold=2, softness=0.1)
        edge = prepare(graph, config)['connections'][0]
        self.assertEqual(edge['structural_features']['intersection_dimension'], 2)
        self.assertEqual(edge['memberships']['portal_1d']['value'], 0)
        self.assertEqual(edge['memberships']['short_safe_portal']['value'], 0)

    def test_point_and_empty_intersections_are_not_inferred_as_1d(self):
        for polygon, dimension in (([[1, 2], [2, 2], [2, 3], [1, 3]], 0),
                                   ([[3, 0], [4, 0], [4, 2], [3, 2]], -1)):
            graph = adjacent_regions()
            graph['vertices'][1]['polygon'] = polygon
            edge = prepare(graph, load_config())['connections'][0]
            self.assertEqual(edge['structural_features']['intersection_dimension'], dimension)
            self.assertFalse(edge['memberships']['portal_1d']['active'])

    def test_missing_or_invalid_polygons_produce_unavailable_memberships(self):
        for polygon in (None, [[0, 0], [1, 1]], [[0, 0], [2, 2], [0, 2], [2, 0]]):
            graph = adjacent_regions()
            graph['vertices'][1]['polygon'] = polygon
            edge = prepare(graph, load_config())['connections'][0]
            for name in ('portal_1d', 'short_safe_portal'):
                self.assertIsNone(edge['memberships'][name]['value'])
                self.assertIn('Cannot determine', edge['memberships'][name]['reason'])

    def test_positive_2d_inradius_is_rejected_for_a_confirmed_1d_intersection(self):
        graph = adjacent_regions()
        for e in graph['directed_edges']:
            e['intersection_inscribed_ball_radius'] = 0.3
        with self.assertRaisesRegex(ValueError, '1D intersection cannot'):
            prepare(graph, load_config())

    def test_point_robot_does_not_block_dimensionality_or_explicit_length_label(self):
        graph, config = adjacent_regions(), load_config()
        graph['planning']['robot_radius'] = 0
        config['label_rules']['short_safe_portal'].update(threshold=1.7, softness=0.2)
        edge = prepare(graph, config)['connections'][0]
        self.assertTrue(edge['memberships']['portal_1d']['active'])
        self.assertAlmostEqual(edge['memberships']['short_safe_portal']['value'], 0.5)
        self.assertIsNone(edge['memberships']['narrow']['value'])

    def test_new_labels_do_not_change_sequences_or_remove_connections(self):
        graph, config = adjacent_regions(), load_config()
        before = prepare(graph, config)
        config['label_rules']['short_safe_portal'].update(threshold=100, softness=0.1)
        after = prepare(graph, config)
        self.assertEqual(before['sequences'], after['sequences'])
        self.assertEqual(len(before['connections']), len(after['connections']))
        self.assertTrue(after['connections'][0]['memberships']['short_safe_portal']['active'])

    def test_all_prepared_maps_use_1d_portals_with_recorded_tolerance_method(self):
        for name in ('flappy', 'hole', 'narrow', 'room'):
            with self.subTest(map=name):
                path = ROOT / 'research/data' / name / 'graph.json'
                original = path.read_bytes()
                result = prepare(json.loads(original), load_config(), name)
                self.assertTrue(result['connections'])
                for e in result['connections']:
                    self.assertTrue(e['memberships']['portal_1d']['active'])
                    self.assertEqual(e['structural_features']['intersection_dimension'], 1)
                    self.assertIsNone(e['memberships']['short_safe_portal']['value'])
                self.assertEqual(original, path.read_bytes())


class PortalConfigurationTests(unittest.TestCase):
    def test_config_rejects_unpaired_or_invalid_calibration(self):
        for threshold, softness in ((1, None), (None, 0.1), (1, 0), (1, -1),
                                     (math.inf, 1), (1, math.nan), (True, 1), ('1', 1)):
            cfg = load_config()
            cfg['label_rules']['short_safe_portal'].update(threshold=threshold, softness=softness)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_old_custom_config_acquires_explicit_null_calibration(self):
        config = load_config()
        for key in ('portal_1d', 'short_safe_portal'):
            del config['label_rules'][key]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            loaded = load_config(path)
        self.assertIsNone(loaded['label_rules']['short_safe_portal']['threshold'])
        self.assertEqual(loaded['label_rules']['portal_1d']['length_tolerance'], 1e-9)

    def test_invalid_numerical_geometry_tolerances_are_rejected(self):
        for value in (0, -1, True, math.nan, math.inf):
            config = load_config()
            config['label_rules']['portal_1d']['length_tolerance'] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)


class PortalScoringTests(unittest.TestCase):
    def test_llm_receives_precomputed_labels_and_cannot_change_them(self):
        data = prepare(adjacent_regions(), load_config())
        frozen, calls = deepcopy(data), []
        def fake(task, payload, **kwargs):
            calls.append((task, payload, kwargs['prompt']))
            return {'status': 'scored', 'score': 0.5, 'reason': 'A retained 1D portal.'}
        with tempfile.TemporaryDirectory() as directory:
            score_prepared(data, output=Path(directory) / 'scores.json', request_fn=fake)
        task = next(item for item in calls if item[0] == 'query_connection_priority')
        self.assertTrue(task[1]['item']['memberships']['portal_1d']['active'])
        self.assertIsNone(task[1]['item']['memberships']['short_safe_portal']['value'])
        self.assertIn('short_safe_portal', task[1]['context']['class1_label_rules'])
        self.assertIn('half a portal length must never replace', task[2])
        self.assertEqual(data, frozen)

    def test_unsupported_short_safe_portal_claim_is_flagged(self):
        edge = prepare(adjacent_regions(), load_config())['connections'][0]
        flags = explanation_review_flags({'reason': 'This is a short_safe_portal.'}, edge, connection=True)
        self.assertTrue(any('short_safe_portal' in flag for flag in flags))
        flags = explanation_review_flags({'reason': 'This is a portal_1d.'}, edge, connection=True)
        self.assertFalse(any('portal_1d' in flag for flag in flags))


if __name__ == '__main__':
    unittest.main()
