from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .common import PlannerRequest, PlannerResult, Point2D, is_segment_visible


def _build_runtime_graph_from_static(
    static_nodes: List[Point2D],
    static_adj: List[List[Tuple[int, float]]],
    inflated_polys: List[List[Tuple[float, float]]],
    start_w: Point2D,
    goal_w: Point2D,
) -> Tuple[List[Point2D], List[List[Tuple[int, float]]], int, int]:
    nodes = list(static_nodes)
    adj = [[(int(j), float(c)) for (j, c) in nbrs] for nbrs in static_adj]
    start_idx = len(nodes)
    goal_idx = len(nodes) + 1
    nodes.append(tuple(start_w))
    nodes.append(tuple(goal_w))
    adj.append([])
    adj.append([])

    def add_undirected(i: int, j: int) -> None:
        cost = float(math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]))
        adj[i].append((j, cost))
        adj[j].append((i, cost))

    for i in range(start_idx):
        if is_segment_visible(nodes[start_idx], nodes[i], inflated_polys):
            add_undirected(start_idx, i)

    for i in range(goal_idx):
        if i == goal_idx:
            continue
        if is_segment_visible(nodes[goal_idx], nodes[i], inflated_polys):
            add_undirected(goal_idx, i)

    return nodes, adj, start_idx, goal_idx


def _shortest_path_visibility_graph(
    nodes: List[Point2D],
    adj: List[List[Tuple[int, float]]],
    start_idx: int,
    goal_idx: int,
) -> Optional[List[Point2D]]:
    import heapq

    n = len(nodes)
    if not (0 <= start_idx < n and 0 <= goal_idx < n):
        return None
    g_score = [float("inf")] * n
    g_score[start_idx] = 0.0
    parent: Dict[int, int] = {}
    open_heap: List[Tuple[float, int]] = []
    heapq.heappush(open_heap, (0.0, start_idx))
    closed = [False] * n

    def h(i: int) -> float:
        return math.hypot(nodes[i][0] - nodes[goal_idx][0], nodes[i][1] - nodes[goal_idx][1])

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == goal_idx:
            path_idx = [cur]
            while cur != start_idx:
                cur = parent[cur]
                path_idx.append(cur)
            path_idx.reverse()
            return [nodes[i] for i in path_idx]
        for nbr, cost in adj[cur]:
            if closed[nbr]:
                continue
            cand = g_score[cur] + cost
            if cand < g_score[nbr]:
                g_score[nbr] = cand
                parent[nbr] = cur
                heapq.heappush(open_heap, (cand + h(nbr), nbr))
    return None


def plan_visibility_from_static(req: PlannerRequest) -> PlannerResult:
    static_nodes = req.static_nodes or []
    static_adj = req.static_adj or [[] for _ in range(len(static_nodes))]
    nodes, adj, s_idx, g_idx = _build_runtime_graph_from_static(
        static_nodes=static_nodes,
        static_adj=static_adj,
        inflated_polys=req.obstacle_polygons,
        start_w=req.start,
        goal_w=req.goal,
    )
    return PlannerResult(
        path_world=_shortest_path_visibility_graph(nodes, adj, start_idx=s_idx, goal_idx=g_idx),
        tree_edges=None,
    )
