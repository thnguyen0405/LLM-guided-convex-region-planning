"""Native Matplotlib event replay and PNG/SVG export of the same figure."""

from pathlib import Path
import textwrap

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider

from .replay import ReplayState, finite_xy

BG, PANEL, TEXT, MUTED = "#080808", "#101010", "#eeeeee", "#aaaaaa"
TREE = ("#2699e6", "#e28c35")
REGION = ("#1b1b1b", "#143047", "#422c16", "#34284b")


def map_title(result):
    """Read new recordings and recover map names from older replay metadata."""
    explicit = result.get("map_name") or result.get("map", {}).get("name")
    if explicit:
        return str(explicit)
    source = result.get("inputs", {}).get("map_file")
    if source:
        path = Path(source)
        return (path.parent.name if path.stem == "map" else path.stem).replace("_", " ").title()
    return "Map"


def region_center(polygon):
    """Area centroid of an ordered convex region, independent of winding."""
    origin = np.asarray(polygon[0], dtype=float)
    relative = np.asarray(polygon, dtype=float) - origin
    following = np.roll(relative, -1, axis=0)
    weights = relative[:, 0] * following[:, 1] - following[:, 0] * relative[:, 1]
    area2 = weights.sum()
    if abs(area2) <= 1e-12:
        return np.asarray(polygon, dtype=float).mean(axis=0)
    return origin + ((relative + following) * weights[:, None]).sum(axis=0) / (3 * area2)


class ReplayFigure:
    """Widget/timer references live as long as the native figure window."""

    def __init__(self, result, figure=None, controls=True, events_per_frame=40):
        self.result, self.state = result, ReplayState(result)
        self.playing, self.controls, self.timer = False, controls, None
        self.events_per_frame = events_per_frame
        if figure is None:
            import matplotlib.pyplot as plt
            figure = plt.figure(figsize=(13, 8.5))
        self.figure = figure
        figure.set_facecolor(BG)
        figure._planner_replay = self
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title(map_title(result))
        config = result["config"]
        figure.text(0.055, 0.955, map_title(result), color=TEXT, size=19, weight="bold")
        self.ax = figure.add_axes([0.055, 0.23 if controls else 0.10, 0.64, 0.64 if controls else 0.77])
        self.ax.set_facecolor(PANEL)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.tick_params(colors=MUTED, labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444444")
        self.ax.set_xlabel("Map x", color=MUTED)
        self.ax.set_ylabel("Map y", color=MUTED)
        m = result["map"]
        self.ax.add_patch(Polygon(m["boundary"], facecolor=REGION[0], edgecolor="#757575", lw=1.4, zorder=0))
        self.region_patches = []
        self.region_labels = []
        for r in self.state.regions:
            patch = Polygon(r["polygon"], facecolor=REGION[0], edgecolor="#5c5c5c", lw=0.65, zorder=1)
            self.ax.add_patch(patch)
            self.region_patches.append(patch)
            x, y = region_center(r["polygon"])
            label = self.ax.text(x, y, r["id"], ha="center", va="center", fontsize=7.5,
                                 color=TEXT, weight="normal", zorder=10, clip_on=True,
                                 path_effects=[patheffects.withStroke(linewidth=2.2, foreground=BG)])
            self.region_labels.append(label)
        for polygon in m.get("obstacles", []):
            self.ax.add_patch(Polygon(polygon, facecolor="#050505", edgecolor="#5a5a5a", lw=0.8, zorder=2))
        self.tree_lines, self.tree_points = [], []
        for color in TREE:
            lines = LineCollection([], colors=color, linewidths=0.8, alpha=0.7, zorder=3)
            self.ax.add_collection(lines)
            self.tree_lines.append(lines)
            self.tree_points.append(self.ax.scatter([], [], s=6, color=color, zorder=4))
        self.path_line, = self.ax.plot([], [], color="#35d19c", lw=2.8, zorder=5)
        self.active_line, = self.ax.plot([], [], color="#c79b22", lw=1.8, zorder=6)
        self.sample_point = self.ax.scatter([], [], s=45, color="#e6b52b", edgecolor="#7d5a00", zorder=7)
        for key, label, color in (("start", "S", TREE[0]), ("goal", "G", TREE[1])):
            p = m[key]
            self.ax.scatter([p[0]], [p[1]], s=64, facecolor=color, edgecolor=BG, lw=1.5, zorder=8)
            self.ax.annotate(label, p[:2], xytext=(8, 5), textcoords="offset points", color=TEXT,
                             size=11, weight="bold", zorder=11,
                             path_effects=[patheffects.withStroke(linewidth=2.2, foreground=BG)])
        xs, ys = zip(*m["boundary"])
        margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.04
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        figure.text(0.745, 0.855, "RUN SUMMARY", color=TEXT, size=11, weight="bold")
        keys = (("Seed", config["seed"]), ("Iterations", result["iterations"]),
                ("Nodes added", result["nodes_added"]), ("Rewires", result["rewires"]),
                ("First solution (s)", result["time_to_first_solution"]),
                ("Planning time (s)", result["planning_time"]),
                ("Path length", result["path_length"]), ("Algorithm cost", result["algorithm_cost"]),
                ("Stop reason", result["stop_reason"]))
        for i, (label, value) in enumerate(keys):
            y = 0.815 - i * 0.033
            figure.text(0.745, y, label, color=MUTED, size=10)
            if isinstance(value, float):
                value = f"{value:.3f}"
            figure.text(0.97, y, "-" if value is None else str(value), ha="right", color=TEXT, size=10)
        handles = [Line2D([], [], color=color, marker="o", ls="none", markersize=7, label=label)
                   for color, label in ((TREE[0], "Start tree"), (TREE[1], "Goal tree"),
                   (REGION[3], "Both trees in region"), ("#35d19c", "Best path so far"),
                   ("#e6b52b", "Current random sample"))]
        figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.735, 0.495),
                      facecolor=BG, edgecolor=BG, labelcolor=TEXT, fontsize=10)
        figure.text(0.745, 0.29, "CURRENT EVENT", color=TEXT, size=11, weight="bold")
        self.event_text = figure.text(0.745, 0.26, "Ready", va="top", color=MUTED, size=10, linespacing=1.5)
        if controls:
            self._make_controls()
        figure.canvas.mpl_connect("close_event", self._close)
        if controls:
            self.timer = figure.canvas.new_timer(interval=40)
            self.timer.add_callback(self._tick)
        self.render()

    def _make_controls(self):
        self.buttons = {}
        actions = (("play", "Play", self.toggle), ("step", "Next event", self.step),
                   ("reset", "Reset", self.reset), ("final", "Final result", self.final))
        for i, (key, label, callback) in enumerate(actions):
            axis = self.figure.add_axes([0.055 + i * 0.117, 0.135, 0.108, 0.04])
            button = Button(axis, label, color="#202020", hovercolor="#383838")
            for spine in axis.spines.values():
                spine.set_edgecolor("#555555")
            button.label.set_color(TEXT)
            button.on_clicked(callback)
            self.buttons[key] = button
        speed_axis = self.figure.add_axes([0.64, 0.14, 0.29, 0.025], facecolor=PANEL)
        self.speed = Slider(speed_axis, "Events/frame", 1, 200, valinit=self.events_per_frame,
                            valstep=1, valfmt="%d", color="#2ab79c", track_color="#333333")
        seek_axis = self.figure.add_axes([0.1, 0.075, 0.83, 0.025], facecolor=PANEL)
        self.seek = Slider(seek_axis, "Event", 0, max(1, len(self.state.events)),
                           valinit=0, valstep=1, valfmt="%d", color="#2ab79c", track_color="#333333")
        for slider in (self.speed, self.seek):
            slider.label.set_color(MUTED)
            slider.valtext.set_color(TEXT)
            slider.vline.set_color("#777777")
        self.seek.on_changed(self._seek)
        self.seek.set_active(bool(self.state.events))

    def pause(self):
        self.playing = False
        if self.timer:
            self.timer.stop()
        if self.controls:
            self.buttons["play"].label.set_text("Play")

    def toggle(self, _event=None):
        if self.playing:
            self.pause()
        elif self.state.events:
            if self.state.cursor >= len(self.state.events):
                self.state.move_to(0)
            self.playing = True
            self.buttons["play"].label.set_text("Pause")
            self.timer.start()
        self.render()

    def _tick(self):
        if not self.playing:
            return
        self.state.move_to(self.state.cursor + int(self.speed.val))
        if self.state.cursor >= len(self.state.events):
            self.pause()
        self.render()

    def step(self, _event=None):
        self.pause()
        self.state.move_to(self.state.cursor + 1)
        self.render()

    def reset(self, _event=None):
        self.pause()
        self.state.move_to(0)
        self.render()

    def final(self, _event=None):
        self.pause()
        self.state.final()
        self.render()

    def _seek(self, value):
        self.pause()
        self.state.move_to(value)
        self.render()

    def _close(self, _event=None):
        self.pause()

    def render(self):
        state = self.state
        for patch, (a, b) in zip(self.region_patches, state.region_counts):
            patch.set_facecolor(REGION[(1 if a else 0) + (2 if b else 0)])
        for tree in (0, 1):
            nodes = [n for n in state.nodes.values() if n["tree"] == tree]
            segments = [[state.nodes[n["parent"]]["point"][:2], n["point"][:2]]
                        for n in nodes if n["parent"] in state.nodes]
            self.tree_lines[tree].set_segments(segments)
            self.tree_points[tree].set_offsets(np.array([n["point"][:2] for n in nodes]).reshape(-1, 2))
        self.path_line.set_data([p[0] for p in state.path], [p[1] for p in state.path])
        self.active_line.set_data([], [])
        self.sample_point.set_offsets(np.empty((0, 2)))
        last = state.last
        if last:
            a, b, kind = last.get("a"), last.get("b"), last["kind"]
            if kind == "sample" and finite_xy(b):
                self.sample_point.set_offsets([b[:2]])
            if kind in ("collision_reject", "steer", "nearest") and finite_xy(a) and finite_xy(b):
                self.active_line.set_data([a[0], b[0]], [a[1], b[1]])
                self.active_line.set_color("#df4761" if kind == "collision_reject" else "#b4891e")
            text = f"Iteration {last['iteration']} | {kind}"
            if "tree" in last:
                text += " | " + ("start tree" if last["tree"] == 0 else "goal tree")
            if "node" in last:
                text += f" | node {last['node']}"
            if "source" in last:
                text += f" | {last['source']}"
        else:
            text = "Ready. Press Play or Next event."
        if self.result.get("dropped_events", 0):
            text += f" | Trace capped: {self.result['dropped_events']:,} events omitted"
        self.event_text.set_text(textwrap.fill(text, width=32))
        if self.controls:
            self.seek.eventson = False
            self.seek.set_val(state.cursor)
            self.seek.eventson = True
        self.figure.canvas.draw_idle()


def write_plots(result, base):
    """Export final snapshots through Agg without opening a window."""
    base = Path(base)
    fig = Figure(figsize=(13, 8.5))
    FigureCanvasAgg(fig)
    replay = ReplayFigure(result, figure=fig, controls=False)
    replay.final()
    for suffix in (".png", ".svg"):
        fig.savefig(base.with_suffix(suffix), dpi=150, facecolor=BG)
    fig.clear()


def show_replays(results):
    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"} or any(
            name in backend for name in ("inline", "webagg", "nbagg", "ipympl")):
        raise RuntimeError("A native Matplotlib GUI backend is required. Use MacOSX, QtAgg or TkAgg; "
                           "use --no-show for PNG/SVG export without a window.")
    windows = [ReplayFigure(result) for result in results]
    try:
        plt.show(block=True)
    finally:
        for window in windows:
            window.pause()
