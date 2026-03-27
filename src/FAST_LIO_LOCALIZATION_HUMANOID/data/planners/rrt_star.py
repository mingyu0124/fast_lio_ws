from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .common import PlannerRequest, PlannerResult, Point2D, is_segment_visible, point_in_polygon

RRT_MAX_ITER = 1500
RRT_STEP_SIZE_M = 0.35
RRT_GOAL_SAMPLE_RATE = 0.20
RRT_REWIRE_RADIUS_M = 1.2
RRT_GOAL_TOLERANCE_M = 0.35
RRT_RANDOM_SEED = 42


def get_rrt_star_config() -> dict:
    return {
        "max_iter": int(RRT_MAX_ITER),
        "step_size_m": float(RRT_STEP_SIZE_M),
        "goal_sample_rate": float(RRT_GOAL_SAMPLE_RATE),
        "rewire_radius_m": float(RRT_REWIRE_RADIUS_M),
        "goal_tolerance_m": float(RRT_GOAL_TOLERANCE_M),
        "random_seed": int(RRT_RANDOM_SEED),
    }


def plan_rrt_star(req: PlannerRequest) -> PlannerResult:
    rng = random.Random(RRT_RANDOM_SEED)
    start, goal = req.start, req.goal
    min_x, min_y, max_x, max_y = req.planning_bounds
    if not (min_x < max_x and min_y < max_y):
        return PlannerResult(path_world=None, tree_edges=None)

    for poly in req.obstacle_polygons:
        if point_in_polygon(start, poly) or point_in_polygon(goal, poly):
            return PlannerResult(path_world=None, tree_edges=None)

    @dataclass
    class Node:
        p: Point2D
        parent: int
        cost: float

    nodes: List[Node] = [Node(start, -1, 0.0)]
    best_goal_idx: Optional[int] = None
    best_goal_cost = float("inf")

    def collides_point(p: Point2D) -> bool:
        for poly in req.obstacle_polygons:
            if point_in_polygon(p, poly):
                return True
        return False

    def collides_segment(a: Point2D, b: Point2D) -> bool:
        return not is_segment_visible(a, b, req.obstacle_polygons)

    def sample_free() -> Point2D:
        if rng.random() < RRT_GOAL_SAMPLE_RATE:
            return goal
        for _ in range(100):
            p = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if not collides_point(p):
                return p
        return goal

    def nearest_index(p: Point2D) -> int:
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(nodes):
            d = (n.p[0] - p[0]) ** 2 + (n.p[1] - p[1]) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def steer(a: Point2D, b: Point2D, step: float) -> Point2D:
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        if d <= step:
            return b
        s = step / (d + 1e-12)
        return (a[0] + dx * s, a[1] + dy * s)

    def near_indices(p: Point2D, radius: float) -> List[int]:
        r2 = radius * radius
        out: List[int] = []
        for i, n in enumerate(nodes):
            if (n.p[0] - p[0]) ** 2 + (n.p[1] - p[1]) ** 2 <= r2:
                out.append(i)
        return out

    for _ in range(RRT_MAX_ITER):
        x_rand = sample_free()
        i_near = nearest_index(x_rand)
        x_new = steer(nodes[i_near].p, x_rand, RRT_STEP_SIZE_M)
        if collides_point(x_new) or collides_segment(nodes[i_near].p, x_new):
            continue

        neighbor_ids = near_indices(x_new, RRT_REWIRE_RADIUS_M)
        parent = i_near
        best_cost = nodes[i_near].cost + math.hypot(x_new[0] - nodes[i_near].p[0], x_new[1] - nodes[i_near].p[1])
        for nid in neighbor_ids:
            cand_cost = nodes[nid].cost + math.hypot(x_new[0] - nodes[nid].p[0], x_new[1] - nodes[nid].p[1])
            if cand_cost < best_cost and not collides_segment(nodes[nid].p, x_new):
                parent = nid
                best_cost = cand_cost

        new_idx = len(nodes)
        nodes.append(Node(x_new, parent, best_cost))

        for nid in neighbor_ids:
            new_cost = best_cost + math.hypot(nodes[nid].p[0] - x_new[0], nodes[nid].p[1] - x_new[1])
            if new_cost + 1e-9 < nodes[nid].cost and not collides_segment(x_new, nodes[nid].p):
                nodes[nid].parent = new_idx
                nodes[nid].cost = new_cost

        d_goal = math.hypot(x_new[0] - goal[0], x_new[1] - goal[1])
        if d_goal <= RRT_GOAL_TOLERANCE_M and not collides_segment(x_new, goal):
            goal_cost = best_cost + d_goal
            if goal_cost < best_goal_cost:
                best_goal_cost = goal_cost
                best_goal_idx = new_idx

    tree_edges: List[tuple[Point2D, Point2D]] = []
    for i in range(1, len(nodes)):
        parent = nodes[i].parent
        if parent >= 0:
            tree_edges.append((nodes[parent].p, nodes[i].p))

    if best_goal_idx is None:
        return PlannerResult(path_world=None, tree_edges=tree_edges)

    path: List[Point2D] = [goal]
    cur = best_goal_idx
    while cur >= 0:
        path.append(nodes[cur].p)
        cur = nodes[cur].parent
    path.reverse()
    return PlannerResult(path_world=path, tree_edges=tree_edges)
