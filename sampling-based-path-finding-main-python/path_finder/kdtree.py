"""Incremental KD tree: same 3-coordinate queries as the C source.

Adapted from John Tsiombikas' kdtree; nearest search credited to Tamas Nepusz.
Full copyright and redistribution terms are in THIRD_PARTY_NOTICES.md.
"""


def squared(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


class KDTree:
    def __init__(self):
        self.root = None

    def insert(self, node):
        entry = [node, 0, None, None]
        if self.root is None:
            self.root = entry
            return
        cursor = self.root
        while True:
            side = 2 if node.x[cursor[1]] < cursor[0].x[cursor[1]] else 3
            if cursor[side] is None:
                entry[1] = (cursor[1] + 1) % 3
                cursor[side] = entry
                return
            cursor = cursor[side]

    def nearest(self, p):
        if self.root is None:
            raise ValueError("Empty KD tree")
        best, best_d = self.root[0], squared(self.root[0].x, p)
        # Explicit traversal avoids Python recursion depth limits on thin maps.
        stack = [(self.root, False, 0)]
        while stack:
            entry, visited, split = stack.pop()
            if entry is None:
                continue
            node, axis, left, right = entry
            dx = p[axis] - node.x[axis]
            near, far = (left, right) if dx <= 0 else (right, left)
            if not visited:
                stack.append((entry, True, dx))
                stack.append((near, False, 0))
            else:
                d = squared(node.x, p)
                if d < best_d:
                    best, best_d = node, d
                if split * split < best_d:
                    stack.append((far, False, 0))
        return best

    def within(self, p, radius):
        result, stack = [], [self.root]
        while stack:
            entry = stack.pop()
            if entry is None:
                continue
            node, axis, left, right = entry
            if squared(node.x, p) <= radius * radius:
                result.append(node)
            dx = p[axis] - node.x[axis]
            near, far = (left, right) if dx <= 0 else (right, left)
            if abs(dx) < radius:
                stack.append(far)
            stack.append(near)
        # C rlist_insert prepends unordered range results.
        return result[::-1]
