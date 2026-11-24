"""Replay state, real Matplotlib controls, exports and CLI replay behavior."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

import demo
from path_finder import MapGeometry, Planner, PlannerConfig
from path_finder.replay import ReplayState
from path_finder.visualization import ReplayFigure, REGION, write_plots, map_title


def fixture():
    m = MapGeometry(dict(boundary=[[0, 0], [10, 0], [10, 10], [0, 10]],
        obstacles=[], robot_radius=0.1, start=[1, 1], goal=[9, 9]))
    result = Planner(m, PlannerConfig(search_time=0, max_iterations=10,
        stop_on_first=True)).plan()
    result["map"] = m.data
    result["regions"] = [dict(id="whole", polygon=m.data["boundary"])]
    return result


class ReplayTests(unittest.TestCase):
    def test_seek_backward_removes_future_nodes_and_solution(self):
        result = fixture()
        state = ReplayState(result)
        state.move_to(len(state.events))
        self.assertTrue(state.path)
        self.assertGreater(len(state.nodes), 2)
        state.move_to(2)
        self.assertEqual(len(state.nodes), 2)
        self.assertEqual(state.region_counts, [[0, 0]])
        self.assertEqual(state.path, [])
        state.move_to(len(state.events))
        self.assertEqual([tuple(p) for p in state.path], [tuple(p) for p in result["path"]])

    def test_rewiring_does_not_double_count_nodes(self):
        state = ReplayState(fixture())
        root = dict(kind="root", node=1, b=[1, 1, 0], tree=0, iteration=0)
        node = dict(kind="add_node", node=2, parent=1, b=[2, 2, 0], tree=0, iteration=1)
        state._apply(root)
        state._apply(node)
        state._apply(dict(node, kind="rewire"))
        self.assertEqual(state.region_counts, [[1, 0]])
        state._apply(dict(node, kind="rewire", tree=1))
        self.assertEqual(state.region_counts, [[0, 1]])

    def test_complete_final_snapshot_works_with_truncated_trace(self):
        result = fixture()
        result["events"] = result["events"][:2]
        result["dropped_events"] = 50
        state = ReplayState(result)
        state.move_to(2)
        self.assertEqual(state.path, [])
        state.final()
        self.assertEqual(len(state.nodes), len(result["trees"]))
        self.assertEqual(state.path, result["path"])
        state.move_to(0)
        self.assertFalse(state.is_final_snapshot)
        self.assertEqual(state.nodes, {})


class MatplotlibTests(unittest.TestCase):
    def setUp(self):
        self.result = fixture()
        self.figure = Figure(figsize=(13, 8.5))
        self.canvas = FigureCanvasAgg(self.figure)
        self.replay = ReplayFigure(self.result, figure=self.figure)

    def tearDown(self):
        self.replay.pause()
        self.figure.clear()

    def click_button(self, name):
        # Real Matplotlib press/release dispatch tests the widget wiring too.
        axis = self.replay.buttons[name].ax
        x, y = axis.transAxes.transform((0.5, 0.5))
        for kind in ("button_press_event", "button_release_event"):
            self.canvas.callbacks.process(kind, MouseEvent(kind, self.canvas, x, y, button=1))

    def test_buttons_timer_and_slider_change_actual_replay_state(self):
        self.click_button("step")
        self.assertEqual(self.replay.state.cursor, 1)
        self.click_button("play")
        self.assertTrue(self.replay.playing)
        self.replay.speed.set_val(1)
        self.replay._tick()
        self.assertEqual(self.replay.state.cursor, 2)
        self.click_button("play")
        self.assertFalse(self.replay.playing)
        self.click_button("final")
        self.assertEqual(self.replay.state.path, self.result["path"])
        self.assertEqual(self.replay.region_patches[0].get_facecolor(), to_rgba(REGION[3]))
        self.replay.seek.set_val(2)
        self.assertEqual(self.replay.state.path, [])
        self.click_button("reset")
        self.assertEqual(self.replay.state.cursor, 0)

    def test_invalid_sample_can_be_displayed_without_exception(self):
        self.replay.state.last = dict(kind="sample", b=[None, None, None], iteration=1)
        self.replay.render()
        self.canvas.draw()
        self.assertEqual(len(self.replay.sample_point.get_offsets()), 0)

    def test_header_shows_only_map_name_for_old_and_new_recordings(self):
        old = copy.deepcopy(self.result)
        old['inputs'] = {'map_file': '/examples/flappy/map.json'}
        self.assertEqual(map_title(old), 'Flappy')
        old['map_name'] = 'Room'
        self.assertEqual(map_title(old), 'Room')
        fig = Figure()
        FigureCanvasAgg(fig)
        ReplayFigure(old, figure=fig)
        header = [text.get_text() for text in fig.texts if text.get_position()[1] >= 0.9]
        self.assertEqual(header, ['Room'])
        fig.clear()

    def test_export_creates_matplotlib_png_svg_and_keeps_input_unchanged(self):
        before = copy.deepcopy(self.result)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "brrt"
            write_plots(self.result, base)
            self.assertTrue(base.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
            self.assertIn("<svg", base.with_suffix(".svg").read_text())
            self.assertFalse(base.with_suffix(".html").exists())
        self.assertEqual(before, self.result)

    def test_zero_event_trace_still_shows_final_result(self):
        result = copy.deepcopy(self.result)
        result["events"] = []
        fig = Figure()
        FigureCanvasAgg(fig)
        replay = ReplayFigure(result, figure=fig)
        self.assertFalse(replay.seek.active)
        replay.toggle()
        self.assertFalse(replay.playing)
        replay.final()
        self.assertEqual(replay.state.path, result["path"])
        fig.clear()

    def test_saved_json_replay_does_not_call_planner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recorded.json"
            path.write_text(json.dumps(self.result))
            args = demo.parser().parse_args(["--replay", str(path), "--no-show",
                "--output", str(Path(folder) / "plots")])
            with patch("demo.Planner", side_effect=AssertionError("Planner must not run")):
                self.assertEqual(demo.run(args), 0)
            self.assertTrue((Path(folder) / "plots/brrt.png").exists())


if __name__ == "__main__":
    unittest.main()
