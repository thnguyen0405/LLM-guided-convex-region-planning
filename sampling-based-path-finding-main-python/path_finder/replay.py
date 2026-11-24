"""Replay actual planner events without changing or rerunning the planner."""

import math
from .geometry import inside


class ReplayState:
    def __init__(self, result):
        self.result = result
        self.events = result.get("events", [])
        self.regions = result.get("regions", [])
        self.membership = {}
        self.reset()

    def reset(self):
        self.cursor = 0
        self.nodes = {}
        self.path = []
        self.last = None
        self.is_final_snapshot = False
        self.region_counts = [[0, 0] for _ in self.regions]

    def _set_node(self, node_id, point, tree, parent):
        old = self.nodes.get(node_id)
        if old and old["parent"] is not None:
            for i in self.membership[node_id]:
                self.region_counts[i][old["tree"]] -= 1
        if node_id not in self.membership:
            self.membership[node_id] = [i for i, r in enumerate(self.regions) if inside(point, r["polygon"])]
        self.nodes[node_id] = dict(point=point, tree=tree, parent=parent)
        if parent is not None:
            for i in self.membership[node_id]:
                self.region_counts[i][tree] += 1

    def _apply(self, event):
        self.last = event
        if event["kind"] == "root":
            self._set_node(event["node"], event["b"], event["tree"], None)
        elif event["kind"] in ("add_node", "rewire"):
            self._set_node(event["node"], event["b"], event["tree"], event["parent"])
        elif event["kind"] == "solution":
            self.path = event["path"]

    def move_to(self, cursor):
        cursor = max(0, min(len(self.events), int(cursor)))
        if cursor < self.cursor or self.is_final_snapshot:
            self.reset()
        while self.cursor < cursor:
            self._apply(self.events[self.cursor])
            self.cursor += 1

    def final(self):
        """Use the complete final tree even if the recorded trace was capped."""
        self.reset()
        for n in self.result.get("trees", []):
            self._set_node(n["id"], n["point"], n["tree"], n["parent"])
        self.path = self.result.get("path", [])
        self.cursor = len(self.events)
        self.last = dict(kind="final_result", iteration=self.result.get("iterations", 0))
        self.is_final_snapshot = True


def finite_xy(point):
    return (point is not None and len(point) >= 2
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in point[:2]))
