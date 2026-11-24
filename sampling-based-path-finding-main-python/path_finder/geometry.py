"""Planar JSON geometry ported from occ_grid/map_geometry.h.

All polygon edges are checked analytically; collision checks are not based on
point samples along a segment. Artificial decomposition boundaries are NOT
obstacles. The robot is a disk with radius + safety_margin clearance.
"""

import json
import math
from pathlib import Path


def point3(p):
    if len(p) not in (2, 3) or not all(math.isfinite(float(x)) for x in p):
        raise ValueError("A point needs two or three finite coordinates")
    return (float(p[0]), float(p[1]), float(p[2]) if len(p) == 3 else 0.0)


def distance(a, b):
    return math.dist(a, b)


def lerp(a, b, t):
    return tuple(x + t * (y - x) for x, y in zip(a, b))


def steer(a, b, length):
    d = distance(a, b)
    return b if d <= length else lerp(a, b, length / d)


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def edges(poly):
    return zip(poly, poly[1:] + poly[:1])


def point_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) if length else 0.0
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def segment_distance(a, b, c, d):
    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return 0.0
    return min(point_segment(a, c, d), point_segment(b, c, d),
               point_segment(c, a, b), point_segment(d, a, b))


def inside(p, poly):
    result = False
    for a, b in edges(poly):
        if point_segment(p, a, b) < 1e-10:
            return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
                result = not result
    return result


def polygon_area(poly):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges(poly))) / 2


def read_polygon(raw):
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("Polygon needs at least three [x,y] vertices")
    if any(len(p) != 2 for p in raw):
        raise ValueError("Polygon coordinates must be [x,y]")
    poly = [point3(p)[:2] for p in raw]
    for i, (a, b) in enumerate(edges(poly)):
        if math.dist(a, b) < 1e-10:
            raise ValueError("Duplicate consecutive polygon vertices")
        for j in range(i + 1, len(poly)):
            if j == i + 1 or (i == 0 and j == len(poly) - 1):
                continue
            if segment_distance(a, b, poly[j], poly[(j + 1) % len(poly)]) < 1e-10:
                raise ValueError("Self-intersecting polygon")
    if polygon_area(poly) < 1e-10:
        raise ValueError("Zero-area polygon")
    return poly


def covered_segment(a, b, poly):
    """Check the WHOLE segment, including crossings of a concave outer ring."""
    if not inside(a, poly) or not inside(b, poly):
        return False
    cuts = [0.0, 1.0]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    for c, d in edges(poly):
        ex, ey = d[0] - c[0], d[1] - c[1]
        det = dx * ey - dy * ex
        if abs(det) > 1e-12:
            t = ((c[0] - a[0]) * ey - (c[1] - a[1]) * ex) / det
            u = ((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / det
            if -1e-10 <= t <= 1 + 1e-10 and -1e-10 <= u <= 1 + 1e-10:
                cuts.append(max(0, min(1, t)))
        elif abs(cross(a, b, c)) < 1e-10 and length:
            for p in (c, d):
                cuts.append(max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)))
    cuts.sort()
    return all(inside(lerp(a, b, (x + y) / 2), poly) for x, y in zip(cuts, cuts[1:]))


class MapGeometry:
    def __init__(self, data):
        self.data = data.get("map", data)
        m = self.data
        self.boundary = read_polygon(m["boundary"])
        self.obstacles = [read_polygon(p) for p in m.get("obstacles", [])]
        self.robot_radius = float(m.get("robot_radius", 0))
        self.safety_margin = float(m.get("safety_margin", 0))
        if any(not math.isfinite(v) or v < 0 for v in (self.robot_radius, self.safety_margin)):
            raise ValueError("Invalid robot radius or safety margin")
        self.clearance = self.robot_radius + self.safety_margin
        for i, p in enumerate(self.obstacles):
            if not all(covered_segment(a, b, self.boundary) for a, b in edges(p)):
                raise ValueError("Obstacle extends outside boundary")
            for q in self.obstacles[:i]:
                if (any(segment_distance(a, b, c, d) < 1e-10 for a, b in edges(p) for c, d in edges(q))
                        or inside(p[0], q) or inside(q[0], p)):
                    raise ValueError("Obstacles must not overlap or touch")
        self.minimum = tuple(min(p[k] for p in self.boundary) for k in (0, 1))
        self.maximum = tuple(max(p[k] for p in self.boundary) for k in (0, 1))
        self.start, self.goal = point3(m["start"]), point3(m["goal"])
        if self.start[2] != self.goal[2]:
            raise ValueError("This planar port requires start and goal at the same z")
        if not self.valid(self.start) or not self.valid(self.goal):
            raise ValueError("Start/goal violates free space or robot clearance")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def point_clearance(self, p):
        return min(point_segment(p, a, b) for poly in [self.boundary, *self.obstacles] for a, b in edges(poly))

    def valid(self, p):
        if not all(math.isfinite(v) for v in p) or not inside(p, self.boundary):
            return False
        if any(inside(p, poly) for poly in self.obstacles):
            return False
        c = self.point_clearance(p)
        return c > 1e-10 and c + 1e-9 >= self.clearance

    def segment_valid(self, a, b):
        if not self.valid(a) or not self.valid(b):
            return False
        for poly in [self.boundary, *self.obstacles]:
            for c, d in edges(poly):
                gap = segment_distance(a, b, c, d)
                if gap < 1e-10 or gap + 1e-9 < self.clearance:
                    return False
        return True

    def path_valid(self, path):
        return bool(path) and all(self.valid(p) for p in path) and all(self.segment_valid(a, b) for a, b in zip(path, path[1:]))
