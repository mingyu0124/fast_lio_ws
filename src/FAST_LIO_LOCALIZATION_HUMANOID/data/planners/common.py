from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
Polygon2D = List[Point2D]


@dataclass
class PlannerRequest:
    planner_name: str
    start: Point2D
    goal: Point2D
    obstacle_polygons: List[Polygon2D]
    planning_bounds: Tuple[float, float, float, float]
    static_nodes: Optional[List[Point2D]] = None
    static_adj: Optional[List[List[Tuple[int, float]]]] = None


@dataclass
class PlannerResult:
    path_world: Optional[List[Point2D]]
    tree_edges: Optional[List[Tuple[Point2D, Point2D]]] = None


def segment_intersect(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D) -> bool:
    def orient(a: Point2D, b: Point2D, c: Point2D) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point2D, b: Point2D, c: Point2D) -> bool:
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-9 and on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-9 and on_segment(p1, p2, q2):
        return True
    if abs(o3) < 1e-9 and on_segment(q1, q2, p1):
        return True
    if abs(o4) < 1e-9 and on_segment(q1, q2, p2):
        return True
    return False


def point_in_polygon(pt: Point2D, poly: Sequence[Sequence[float]]) -> bool:
    x, y = pt
    n = len(poly)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-9:
            if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return True
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1 + 1e-15)
            x_int = x1 + t * (x2 - x1)
            if x_int > x:
                inside = not inside
    return inside


def is_segment_visible(p: Point2D, q: Point2D, polys: List[Polygon2D]) -> bool:
    if p == q:
        return False
    mid = ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)
    for poly in polys:
        if point_in_polygon(mid, poly):
            return False
    for poly in polys:
        m = len(poly)
        for k in range(m):
            a, b = poly[k], poly[(k + 1) % m]
            if segment_intersect(p, q, a, b):
                if p == a or p == b or q == a or q == b:
                    continue
                return False
    return True


def densify_path(path_world: List[Point2D], interval_m: float) -> List[Point2D]:
    if not path_world or interval_m <= 0.0:
        return path_world
    if len(path_world) == 1:
        return list(path_world)
    result: List[Point2D] = [path_world[0]]
    for i in range(len(path_world) - 1):
        ax, ay = path_world[i]
        bx, by = path_world[i + 1]
        d = math.hypot(bx - ax, by - ay)
        if d < 1e-9:
            continue
        n = max(1, int(math.ceil(d / interval_m)))
        for j in range(1, n):
            t = j / n
            result.append((ax + t * (bx - ax), ay + t * (by - ay)))
        result.append((bx, by))
    return result
