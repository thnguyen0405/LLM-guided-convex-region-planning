"""Ports of rrt.h, rrt_star.h, rrt_sharp.h, brrt.h and brrt_star.h.

Copyright (C) 2022 Hongkai Ye; BRRT also credits Longji Yin.
See THIRD_PARTY_NOTICES.md for the full source terms and PORTING_NOTES.md for
intentional differences, source quirks, and the scope of equivalence checks.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
import math
import time

from .geometry import distance, point3, steer
from .kdtree import KDTree
from .sampler import BiasSampler


PLANNERS = ("rrt", "rrt_star", "rrt_sharp", "brrt", "brrt_star")


@dataclass
class PlannerConfig:
    planner: str = "brrt"
    steer_length: float = 2.0
    search_radius: float = 6.0
    search_time: float = 1.0
    max_nodes: int = 5000
    max_iterations: int = 20000
    seed: int = 42
    sampling: str = "uniform"
    stop_on_first: bool = False
    cost_mode: str = "source"
    trace_limit: int = 100000

    def __post_init__(self):
        if self.planner not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}")
        for key in ("steer_length", "search_radius"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.search_time) or self.search_time < 0:
            raise ValueError("search_time must be finite and nonnegative; 0 disables wall-clock limit")
        for key, minimum in (("max_nodes", 2), ("max_iterations", 1), ("trace_limit", 0)):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.cost_mode not in ("source", "geometric"):
            raise ValueError("cost_mode must be source or geometric")
        if self.sampling not in ("uniform", "informed", "guild"):
            raise ValueError("Unknown sampling mode")
        if self.planner in ("rrt", "brrt") and self.sampling != "uniform":
            raise ValueError("The source RRT and BRRT do not enable informed/GUILD updates")
        if self.planner == "brrt_star" and self.sampling == "guild":
            raise ValueError("The source BRRT* does not implement GUILD")


@dataclass(eq=False)
class Node:
    id: int
    x: tuple
    tree: int
    parent: object = None
    cost: float = math.inf
    edge_cost: float = 0.0
    heuristic: float = 0.0
    children: list = field(default_factory=list)


class Planner:
    def __init__(self, geometry, config=None, sampler=None, event_callback=None):
        self.map = geometry
        self.config = config or PlannerConfig()
        self.sampler = sampler or BiasSampler(geometry, self.config.seed, self.config.sampling)
        if isinstance(self.sampler, BiasSampler) and self.sampler.mode != self.config.sampling:
            raise ValueError("Sampler mode must match PlannerConfig.sampling")
        self.event_callback = event_callback

    def emit(self, kind, **fields):
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (list, tuple)):
                return [finite(v) for v in value]
            return value
        fields = {k: finite(v) for k, v in fields.items()}
        event = dict(iteration=self.iteration, kind=kind, **fields)
        if len(self.events) < self.config.trace_limit:
            self.events.append(event)
        else:
            self.dropped_events += 1
        if self.event_callback:
            self.event_callback(dict(event))

    def check(self, a, b):
        ok = self.map.segment_valid(a, b)
        self.collision_checks += 1
        self.emit("collision_clear" if ok else "collision_reject", a=a, b=b)
        return ok

    def add_node(self, parent, x, cost, edge_cost, heuristic=0):
        if len(self.nodes) >= self.config.max_nodes:
            raise RuntimeError("Node budget exhausted")
        node = Node(len(self.nodes), x, parent.tree, parent, cost, edge_cost, heuristic)
        parent.children.append(node)
        self.nodes.append(node)
        self.sampler.record_added_node(x, self.iteration)
        self.emit("add_node", node=node.id, parent=parent.id, tree=node.tree, a=parent.x, b=x)
        return node

    def change_parent(self, node, parent, edge_cost):
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent, node.edge_cost, node.tree = parent, edge_cost, parent.tree
        node.cost = parent.cost + edge_cost
        parent.children.append(node)
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for child in current.children:
                child.cost = current.cost + child.edge_cost
                queue.append(child)
        self.rewires += 1
        self.emit("rewire", node=node.id, parent=parent.id, tree=node.tree, a=parent.x, b=node.x)

    @staticmethod
    def branch(node):
        path, seen = [], set()
        while node is not None:
            if node.id in seen:
                raise RuntimeError("Cycle in planner parent links")
            seen.add(node.id)
            path.append(node.x)
            node = node.parent
        return path[::-1]

    def joined_path(self, a, b):
        if a.tree == 1:
            a, b = b, a
        return self.branch(a) + self.branch(b)[::-1]

    def record_solution(self, cost, path, update_sampler=True):
        if self.solutions and cost >= self.solutions[-1]["cost"]:
            return
        self.solutions.append(dict(cost=cost, time=time.perf_counter() - self.started,
                                   iteration=self.iteration, path=list(path)))
        self.best_path = list(path)
        self.emit("solution", path=list(path), cost=cost)
        if update_sampler:
            self.sampler.update_solution(cost, self.start, self.goal, self.nodes)

    def fixed_edge_cost(self, a, b):
        # Preserve the source's full-steer-length accounting even on a short step.
        return self.config.steer_length if self.config.cost_mode == "source" else distance(a, b)

    def sample(self):
        x = point3(self.sampler.sample()) if not isinstance(self.sampler, BiasSampler) else self.sampler.sample()
        self.emit("sample", b=x, source=getattr(self.sampler, "last_source", "preserved"))
        if not self.map.valid(x):
            self.emit("sample_reject", b=x)
            return None
        return x

    def iterations(self):
        for i in range(1, self.config.max_iterations + 1):
            if len(self.nodes) >= self.config.max_nodes:
                self.stop_reason = "node_budget"
                return
            if self.config.search_time and time.perf_counter() - self.started >= self.config.search_time:
                self.stop_reason = "time_budget"
                return
            if self.config.stop_on_first and self.solutions:
                self.stop_reason = "first_solution"
                return
            self.iteration = i
            yield i
        self.stop_reason = "iteration_budget"

    def plan(self, start=None, goal=None):
        s = point3(start if start is not None else self.map.start)
        g = point3(goal if goal is not None else self.map.goal)
        if s[2] != g[2] or s[2] != self.map.start[2]:
            raise ValueError("Planar start/goal z must match the map planning plane")
        if not self.map.valid(s) or not self.map.valid(g):
            raise ValueError("Invalid start or goal")
        self.sampler.validate_endpoints(s, g)
        self.sampler.begin_run(g)
        self.iteration, self.rewires, self.collision_checks, self.dropped_events = 0, 0, 0, 0
        self.events, self.solutions, self.best_path = [], [], []
        bidirectional = self.config.planner in ("brrt", "brrt_star")
        # Keep source pool IDs: goal=0, start=1. Tree identities never swap.
        self.goal = Node(0, g, 1, cost=0.0 if bidirectional else math.inf)
        self.start = Node(1, s, 0, cost=0.0, heuristic=distance(s, g))
        self.nodes = [self.goal, self.start]
        self.started, self.stop_reason = time.perf_counter(), "iteration_budget"
        self.emit("root", node=1, tree=0, b=s)
        self.emit("root", node=0, tree=1, b=g)
        if distance(s, g) <= 1e-12:
            self.record_solution(0.0, [s], update_sampler=False)
            self.stop_reason = "same_start_goal"
        elif self.config.planner == "brrt":
            self._brrt()
        elif self.config.planner == "brrt_star":
            self._brrt_star()
        else:
            self._single_tree()
        elapsed = time.perf_counter() - self.started
        if self.solutions and not bidirectional and distance(s, g) > 1e-12:
            self.best_path = self.branch(self.goal)
        valid = self.map.path_valid(self.best_path) if self.best_path else False
        if self.solutions and not valid:
            raise RuntimeError("Planner returned a path that fails independent final collision validation")
        path_length = sum(distance(a, b) for a, b in zip(self.best_path, self.best_path[1:])) if valid else None
        return dict(schema="python-sampling-planner/v1", config=asdict(self.config),
            success=bool(self.solutions), stop_reason=self.stop_reason, iterations=self.iteration,
            nodes_added=len(self.nodes) - 2, collision_checks=self.collision_checks, rewires=self.rewires,
            planning_time=elapsed, time_to_first_solution=self.solutions[0]["time"] if self.solutions else None,
            algorithm_cost=(self.solutions[-1]["cost"] if bidirectional else self.goal.cost) if valid and len(self.best_path) > 1 else (0.0 if valid else None),
            path_length=path_length, path_valid=valid, path=self.best_path,
            solutions=self.solutions, events=self.events, dropped_events=self.dropped_events,
            trees=[dict(id=n.id, point=n.x, tree=n.tree, parent=n.parent.id if n.parent else None,
                        cost=n.cost if math.isfinite(n.cost) else None) for n in self.nodes],
            guidance=getattr(self.sampler, "guidance", "none"),
            sampler_settings=dict(clip_guidance_to_informed=getattr(self.sampler, "clip", False),
                                  density_config=getattr(self.sampler, "density", {})),
            density_log=getattr(self.sampler, "density_log", []))

    def _single_tree(self):
        kd = KDTree()
        kd.insert(self.start)
        c = self.config
        star, sharp = c.planner != "rrt", c.planner == "rrt_sharp"
        for _ in self.iterations():
            x_rand = self.sample()
            if x_rand is None:
                continue
            nearest = kd.nearest(x_rand)
            x_new = steer(nearest.x, x_rand, c.steer_length)
            self.emit("nearest", a=nearest.x, b=x_rand, tree=0)
            self.emit("steer", a=nearest.x, b=x_new, tree=0)
            if not self.check(nearest.x, x_new):
                continue
            neighbours = kd.within(x_new, c.search_radius) if star else []
            checked = {nearest.id: True}
            edge_cost = distance(nearest.x, x_new) if star else self.fixed_edge_cost(nearest.x, x_new)
            parent, cost = nearest, nearest.cost + edge_cost
            for n in neighbours:
                d = distance(n.x, x_new)
                if n is not nearest and n.cost + d < cost:
                    checked[n.id] = self.check(n.x, x_new)
                    if checked[n.id]:
                        parent, cost, edge_cost = n, n.cost + d, d
            heuristic = distance(x_new, self.goal.x)
            if sharp and cost + heuristic >= self.goal.cost:
                self.emit("cost_reject", b=x_new)
                continue
            new = self.add_node(parent, x_new, cost, edge_cost, heuristic)
            kd.insert(new)
            if heuristic <= c.search_radius and self.check(x_new, self.goal.x):
                if cost + heuristic < self.goal.cost:
                    self.change_parent(self.goal, new, heuristic)
                    self.record_solution(self.goal.cost, self.branch(self.goal))
            queue = []
            for n in neighbours:
                d = distance(new.x, n.x)
                if (new.cost + d < n.cost
                        and new.cost + d + distance(n.x, self.goal.x) < self.goal.cost):
                    connected = checked[n.id] if n.id in checked else self.check(new.x, n.x)
                    if connected:
                        old_cost = self.goal.cost
                        self.change_parent(n, new, d)
                        if sharp:
                            queue.append(n)
                        if self.goal.cost < old_cost:
                            self.record_solution(self.goal.cost, self.branch(self.goal))
            # RRT# adds best-first, cascading neighbour relaxation to RRT*.
            while queue:
                # Re-sort mutable g+h keys before selection; no stale heap keys.
                current = min(queue, key=lambda n: (n.cost + n.heuristic, n.id))
                queue.remove(current)
                for n in kd.within(current.x, c.search_radius):
                    d = distance(current.x, n.x)
                    if (current.cost + d < n.cost
                            and current.cost + d + n.heuristic < self.goal.cost
                            and self.check(current.x, n.x)):
                        old_cost = self.goal.cost
                        self.change_parent(n, current, d)
                        queue.append(n)
                        if self.goal.cost < old_cost:
                            # Source's cascading phase records but does not update its sampler.
                            self.record_solution(self.goal.cost, self.branch(self.goal), update_sampler=False)

    def _greedy_steer(self, a, target):
        connects, current = [], a
        # Source does NOT insert target into the opposite tree: the final
        # segment bridges its last intermediate node to the active tree node.
        while distance(current, target) > self.config.steer_length:
            new = steer(current, target, self.config.steer_length)
            if not self.check(current, new):
                return False, connects
            connects.append(new)
            current = new
        if not self.check(current, target):
            return False, connects
        return True, connects

    def _brrt(self):
        trees = [KDTree(), KDTree()]
        trees[0].insert(self.start)
        trees[1].insert(self.goal)
        active, other = 0, 1
        best_cost = math.inf
        c = self.config
        for _ in self.iterations():
            x_rand = self.sample()
            if x_rand is None:
                continue
            nearest = trees[active].nearest(x_rand)
            x_new = steer(nearest.x, x_rand, c.steer_length)
            self.emit("nearest", a=nearest.x, b=x_rand, tree=active)
            self.emit("steer", a=nearest.x, b=x_new, tree=active)
            if not self.map.valid(x_new) or not self.check(nearest.x, x_new):
                active, other = other, active
                self.emit("swap", active_tree=active)
                continue
            edge = self.fixed_edge_cost(nearest.x, x_new)
            new_a = self.add_node(nearest, x_new, nearest.cost + edge, edge)
            trees[active].insert(new_a)
            near_b = trees[other].nearest(x_new)
            connected, connects = self._greedy_steer(near_b.x, x_new)
            # Match source's conservative >= check, including discarded partial chains.
            if connects and len(self.nodes) + len(connects) >= c.max_nodes:
                self.stop_reason = "node_budget"
                break
            new_b = near_b
            for p in connects:
                edge = self.fixed_edge_cost(new_b.x, p)
                new_b = self.add_node(new_b, p, new_b.cost + edge, edge)
                trees[other].insert(new_b)
            if connected:
                cost = new_a.cost + new_b.cost + distance(new_a.x, new_b.x)
                if cost < best_cost:
                    best_cost = cost
                    self.record_solution(cost, self.joined_path(new_a, new_b), update_sampler=False)
            active, other = other, active
            self.emit("swap", active_tree=active)

    def _brrt_star(self):
        # Source IB-RRT*: insert x_rand directly, NOT the BRRT Extend/Connect loop.
        trees = [KDTree(), KDTree()]
        trees[0].insert(self.start)
        trees[1].insert(self.goal)
        best_cost = math.inf
        c = self.config
        for _ in self.iterations():
            x = self.sample()
            if x is None:
                continue
            neighbours = [tree.within(x, c.search_radius) for tree in trees]
            parents, costs, edge_costs, checks = [], [], [], []
            for group in neighbours:
                group.sort(key=lambda n: n.cost + distance(n.x, x))
                parent, cost, edge, checked = None, math.inf, math.inf, {}
                for n in group:
                    checked[n.id] = self.check(n.x, x)
                    if checked[n.id]:
                        parent, edge = n, distance(n.x, x)
                        cost = n.cost + edge
                        break
                parents.append(parent)
                costs.append(cost)
                edge_costs.append(edge)
                checks.append(checked)
            if parents[0] is None and parents[1] is None:
                continue
            selected = 0 if costs[0] < costs[1] else 1
            target = self.goal.x if selected == 0 else self.start.x
            if costs[selected] + distance(x, target) >= best_cost:
                self.emit("cost_reject", b=x)
                continue
            new = self.add_node(parents[selected], x, costs[selected], edge_costs[selected])
            trees[selected].insert(new)
            for n in neighbours[selected]:
                d = distance(new.x, n.x)
                if new.cost + d < n.cost and new.cost + d + distance(n.x, target) < best_cost:
                    connected = checks[selected].get(n.id)
                    if connected is None:
                        connected = self.check(new.x, n.x)
                    if connected:
                        self.change_parent(n, new, d)
            if all(p is not None for p in parents):
                cost = costs[0] + costs[1]
                if cost < best_cost:
                    best_cost = cost
                    path = self.joined_path(new, parents[1]) if selected == 0 else self.joined_path(parents[0], new)
                    self.record_solution(cost, path)
