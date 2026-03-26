#!/usr/bin/env python3
"""
런타임 global path 전용: 사전에 preprocess_pcd_map.py 로 만든
global_path_obstacle_map(JSON)만 읽고, 가시 그래프로 경로만 생성합니다.

localization용 point cloud 맵(PCD)은 여기서 사용하지 않습니다.
(맵 빌드·장애물 추출은 전부 오프라인 전처리에서 끝난 상태를 가정)

사용 예:
  python3 pcd_grid_path_planner.py test_global_obstacles.json --start 0.0 0.0 --goal 0.0 -5.0

출력:
  - 경로: out_path.json (world 좌표)
  - 장애물 요약: out_obstacles.json (옵션)
  - 시각화: out_debug.html (plotly, 옵션)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    resolution: float
    origin_x: float  # world x at grid(0,0) lower-left corner
    origin_y: float  # world y at grid(0,0) lower-left corner
    width: int  # cols (x index)
    height: int  # rows (y index)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((x - self.origin_x) / self.resolution))
        gy = int(math.floor((y - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world_center(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height


@dataclass
class ObstacleSummary:
    id: int
    center: Tuple[float, float]
    size: Tuple[float, float]
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    cell_count: int


# ----- 다각형 기반 가시 그래프 경로 생성용 기하 유틸 -----


# 역할: 두 2D 선분이 교차(끝점 포함)하는지 판별한다.
def _segment_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
) -> bool:
    """두 선분 p1-p2, q1-q2가 교차하는지 (끝점에서 만나는 것도 교차로 간주)."""
    def orient(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

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


# 역할: 점이 다각형 내부 또는 경계에 있는지 검사한다.
def _point_in_polygon(pt: Tuple[float, float], poly: Sequence[Sequence[float]]) -> bool:
    """점이 다각형 내부(또는 경계)에 있는지 ray-casting으로 검사."""
    x, y = pt
    n = len(poly)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # 경계 위인지
        if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-9:
            if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return True
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1 + 1e-15)
            x_int = x1 + t * (x2 - x1)
            if x_int > x:
                inside = not inside
    return inside


# 역할: 선분 p-q가 장애물 다각형들과 충돌 없이 가시한지 검사한다.
def _is_segment_visible(
    p: Tuple[float, float],
    q: Tuple[float, float],
    polys: List[List[Tuple[float, float]]],
) -> bool:
    if p == q:
        return False
    mid = ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)
    for poly in polys:
        if _point_in_polygon(mid, poly):
            return False
    for poly in polys:
        m = len(poly)
        for k in range(m):
            a, b = poly[k], poly[(k + 1) % m]
            if _segment_intersect(p, q, a, b):
                if p == a or p == b or q == a or q == b:
                    continue
                return False
    return True


# 역할: 정적 그래프에 시작/목표 노드를 임시 연결해 런타임 그래프를 만든다.
def build_runtime_graph_from_static(
    static_nodes: List[Tuple[float, float]],
    static_adj: List[List[Tuple[int, float]]],
    inflated_polys: List[List[Tuple[float, float]]],
    start_w: Tuple[float, float],
    goal_w: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], List[List[Tuple[int, float]]], int, int]:
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
        if _is_segment_visible(nodes[start_idx], nodes[i], inflated_polys):
            add_undirected(start_idx, i)

    for i in range(goal_idx):
        if i == goal_idx:
            continue
        if _is_segment_visible(nodes[goal_idx], nodes[i], inflated_polys):
            add_undirected(goal_idx, i)

    return nodes, adj, start_idx, goal_idx


# 역할: 가시 그래프에서 A*로 start_idx부터 goal_idx까지 최단 경로를 찾는다.
def shortest_path_visibility_graph(
    nodes: List[Tuple[float, float]],
    adj: List[List[Tuple[int, float]]],
    start_idx: int,
    goal_idx: int,
) -> Optional[List[Tuple[float, float]]]:
    """가시 그래프 위에서 A*로 최단 경로 (world 좌표 리스트)."""
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


# 역할: 경로 구간을 interval 간격으로 선형 보간해 점을 촘촘히 만든다.
def densify_path(
    path_world: List[Tuple[float, float]],
    interval_m: float,
) -> List[Tuple[float, float]]:
    """
    경로의 각 직선 구간을 따라 지정한 거리(interval_m)마다 점을 보간하여 반환.
    interval_m <= 0 이면 원본 경로를 그대로 반환.
    """
    if not path_world or interval_m <= 0.0:
        return path_world
    if len(path_world) == 1:
        return list(path_world)
    result: List[Tuple[float, float]] = [path_world[0]]
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


# 역할: 객체를 UTF-8 JSON 파일로 저장한다.
def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 역할: 전처리 산출물(map_obstacles_v1) JSON을 로드하고 핵심 필드를 추출한다.
def load_obstacle_map_json(
    path: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Optional[GridSpec], Optional[Dict[str, object]]]:
    """
    preprocess가 생성한 map_obstacles_v1 JSON 로드.

    returns:
      (obstacles 리스트, obstacle_hulls, grid용 GridSpec 또는 None, visibility_graph)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("map_obstacles_v1 형식(JSON object)이 아닙니다.")

    obstacle_hulls = list(data.get("obstacle_hulls") or [])
    obstacles = list(data.get("obstacles") or [])
    grid = data.get("grid")
    spec: Optional[GridSpec] = None
    if isinstance(grid, dict):
        origin = grid.get("origin") or [0.0, 0.0]
        spec = GridSpec(
            resolution=float(grid["resolution"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            width=int(grid["width"]),
            height=int(grid["height"]),
        )
    vis_graph = data.get("visibility_graph")
    if not isinstance(vis_graph, dict):
        raise ValueError("visibility_graph가 없습니다. preprocess_pcd_map.py를 다시 실행해 주세요.")
    return obstacles, obstacle_hulls, spec, vis_graph


# 역할: 장애물/시작/목표/경로를 plotly HTML로 시각화 저장한다.
def _try_write_debug_html(
    out_path: str,
    start_w: Tuple[float, float],
    goal_w: Tuple[float, float],
    path_world: Optional[List[Tuple[float, float]]],
    obstacle_hulls: Optional[List[Dict[str, object]]] = None,
    map_xy: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    spec: Optional[GridSpec] = None,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    fig = go.Figure()

    if map_xy is not None:
        map_xy = np.asarray(map_xy)
        if map_xy.ndim == 2 and map_xy.shape[1] == 2 and map_xy.shape[0] > 0:
            n_map = int(map_xy.shape[0])
            if n_map > 80000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_map, size=80000, replace=False)
                map_xy_show = map_xy[idx]
            else:
                map_xy_show = map_xy
            fig.add_trace(
                go.Scattergl(
                    x=map_xy_show[:, 0],
                    y=map_xy_show[:, 1],
                    mode="markers",
                    marker=dict(size=3, color="dimgray", opacity=0.75),
                    name="map (xy)",
                )
            )

    # 장애물별 그리드 셀을 색깔로 표시 (labels: 0=free, 1..K=장애물 id)
    grid_colors = [
        "crimson", "darkorange", "gold", "limegreen", "dodgerblue",
        "mediumpurple", "cyan", "hotpink", "peru", "teal",
    ]
    if labels is not None and labels.size > 0 and spec is not None:
        max_id = int(np.max(labels))
        for cid in range(1, max_id + 1):
            ys, xs = np.nonzero(labels == cid)
            if xs.size == 0:
                continue
            wx = spec.origin_x + (xs.astype(np.float64) + 0.5) * spec.resolution
            wy = spec.origin_y + (ys.astype(np.float64) + 0.5) * spec.resolution
            color = grid_colors[(cid - 1) % len(grid_colors)]
            fig.add_trace(
                go.Scattergl(
                    x=wx,
                    y=wy,
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.85, line=dict(width=0)),
                    name=f"obstacle {cid}",
                )
            )

    # 장애물 외곽선(볼록 껍질)을 선으로 표시
    if obstacle_hulls:
        for item in obstacle_hulls:
            cid = int(item.get("id", -1))
            hull = np.asarray(item.get("hull", []), dtype=np.float64)
            if hull.ndim != 2 or hull.shape[0] < 2:
                continue
            # 폐곡선이 되도록 시작점을 한 번 더 추가
            xs = hull[:, 0].tolist() + [hull[0, 0]]
            ys = hull[:, 1].tolist() + [hull[0, 1]]
            color = grid_colors[(cid - 1) % len(grid_colors)] if cid > 0 else "black"
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=3, color=color),
                    name=f"obstacle {cid} hull",
                )
            )

    fig.add_trace(
        go.Scattergl(
            x=[start_w[0]],
            y=[start_w[1]],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="start",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[goal_w[0]],
            y=[goal_w[1]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="goal",
        )
    )
    if path_world:
        fig.add_trace(
            go.Scattergl(
                x=[p[0] for p in path_world],
                y=[p[1] for p in path_world],
                mode="lines+markers",
                line=dict(width=3, color="royalblue"),
                marker=dict(size=5, color="royalblue"),
                name="path",
            )
        )

    fig.update_layout(
        title="장애물 다각형 + 경로 (world XY)",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    fig.write_html(out_path)
    return True


# 역할: CLI 인자를 파싱하고 정적 그래프 기반 경로 생성을 실행한다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="전처리 장애물 JSON 기반 가시 그래프 경로 생성기 (그리드 추출은 preprocess_pcd_map.py 전용)"
    )
    parser.add_argument(
        "obstacles_json",
        help="preprocess_pcd_map.py가 생성한 map_obstacles_v1 JSON (*_obstacles.json)",
    )
    parser.add_argument("--path-point-interval", type=float, default=0.1, help="경로를 저장/시각화할 때 이 거리[m]마다 점을 보간 (0이면 꺾이는 지점만). 기본 0.1")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="시작점 world 좌표 [m]")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"), help="목표점 world 좌표 [m]")
    parser.add_argument("--out", type=str, default="out_path.json", help="경로 출력 JSON (기본 out_path.json)")
    parser.add_argument("--out-obstacles", type=str, default="out_obstacles.json", help="장애물 요약 JSON 출력 (옵션)")
    parser.add_argument("--out-debug-html", type=str, default="out_debug.html", help="디버그 HTML 출력 (plotly 필요)")
    args = parser.parse_args()

    oj = args.obstacles_json
    if not os.path.isabs(oj):
        oj = os.path.abspath(oj)
    if not os.path.isfile(oj):
        raise SystemExit(f"장애물 JSON을 찾을 수 없습니다: {oj}")

    try:
        obstacles_loaded, obstacle_hulls, spec_from_json, visibility_graph = load_obstacle_map_json(oj)
    except Exception as e:
        raise SystemExit(f"장애물 JSON 로드 실패: {e}") from e
    if not obstacle_hulls:
        raise SystemExit("obstacles JSON에 유효한 obstacle_hulls가 없습니다.")
    hull_dict = {int(item["id"]): item.get("hull", []) for item in obstacle_hulls if "id" in item}
    summaries: List[ObstacleSummary] = []
    seen_ids: Set[int] = set()
    for o in obstacles_loaded:
        if not isinstance(o, dict) or "id" not in o:
            continue
        cid = int(o["id"])
        seen_ids.add(cid)
        c = o.get("center", [0.0, 0.0])
        sz = o.get("size", [0.0, 0.0])
        bmin = o.get("bbox_min", [0.0, 0.0])
        bmax = o.get("bbox_max", [0.0, 0.0])
        summaries.append(
            ObstacleSummary(
                id=cid,
                center=(float(c[0]), float(c[1])),
                size=(float(sz[0]), float(sz[1])),
                bbox_min=(float(bmin[0]), float(bmin[1])),
                bbox_max=(float(bmax[0]), float(bmax[1])),
                cell_count=int(o.get("cell_count", 0)),
            )
        )
    for cid in sorted(hull_dict.keys()):
        if cid not in seen_ids:
            summaries.append(
                ObstacleSummary(
                    id=cid,
                    center=(0.0, 0.0),
                    size=(0.0, 0.0),
                    bbox_min=(0.0, 0.0),
                    bbox_max=(0.0, 0.0),
                    cell_count=0,
                )
            )
    if spec_from_json is None:
        spec = GridSpec(resolution=1.0, origin_x=0.0, origin_y=0.0, width=1, height=1)
    else:
        spec = spec_from_json

    start_w = (float(args.start[0]), float(args.start[1]))
    goal_w = (float(args.goal[0]), float(args.goal[1]))

    nodes_raw = visibility_graph.get("nodes") or []
    adj_raw = visibility_graph.get("adj") or []
    inflated_hulls_raw = visibility_graph.get("inflated_obstacle_hulls") or []
    static_nodes: List[Tuple[float, float]] = []
    for p in nodes_raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            static_nodes.append((float(p[0]), float(p[1])))
    static_adj: List[List[Tuple[int, float]]] = []
    for nbrs in adj_raw:
        row: List[Tuple[int, float]] = []
        if isinstance(nbrs, list):
            for item in nbrs:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    row.append((int(item[0]), float(item[1])))
        static_adj.append(row)
    if len(static_adj) != len(static_nodes):
        static_adj = [[] for _ in range(len(static_nodes))]

    inflated_polys: List[List[Tuple[float, float]]] = []
    for item in inflated_hulls_raw:
        hull = np.asarray(item.get("hull", []), dtype=np.float64) if isinstance(item, dict) else np.empty((0, 2))
        if hull.ndim != 2 or hull.shape[0] < 3:
            continue
        inflated_polys.append([(float(hull[k, 0]), float(hull[k, 1])) for k in range(hull.shape[0])])

    planner_name = "visibility_static_plus_sg"
    planner_robot_radius = float(visibility_graph.get("robot_radius", 0.0))
    nodes, adj, s_idx, g_idx = build_runtime_graph_from_static(
        static_nodes=static_nodes,
        static_adj=static_adj,
        inflated_polys=inflated_polys,
        start_w=start_w,
        goal_w=goal_w,
    )
    path_world = shortest_path_visibility_graph(nodes, adj, start_idx=s_idx, goal_idx=g_idx)
    if path_world and args.path_point_interval > 0:
        path_world = densify_path(path_world, interval_m=float(args.path_point_interval))

    out_obj: Dict[str, object] = {
        "obstacles_json": oj,
        "grid": {
            "resolution": spec.resolution,
            "origin": [spec.origin_x, spec.origin_y],
            "width": spec.width,
            "height": spec.height,
        },
        "planner": {
            "name": planner_name,
            "robot_radius": planner_robot_radius,
            "obstacles_from_preprocess_json": True,
            "used_precomputed_static_graph": True,
        },
        "path_point_interval": float(args.path_point_interval),
        "start": {"world": [start_w[0], start_w[1]]},
        "goal": {"world": [goal_w[0], goal_w[1]]},
        "path": {
            "found": path_world is not None,
            "world": [[x, y] for (x, y) in path_world] if path_world else None,
        },
        "obstacles": [
            {
                "id": s.id,
                "center": [s.center[0], s.center[1]],
                "size": [s.size[0], s.size[1]],
                "bbox_min": [s.bbox_min[0], s.bbox_min[1]],
                "bbox_max": [s.bbox_max[0], s.bbox_max[1]],
                "cell_count": s.cell_count,
                "hull": hull_dict.get(s.id, None),
            }
            for s in summaries
        ],
        "obstacle_hulls": obstacle_hulls,
    }

    _write_json(args.out, out_obj)

    if args.out_obstacles:
        _write_json(
            args.out_obstacles,
            [
                {
                    "id": s.id,
                    "center": s.center,
                    "size": s.size,
                    "bbox_min": s.bbox_min,
                    "bbox_max": s.bbox_max,
                    "cell_count": s.cell_count,
                    "hull": hull_dict.get(s.id, None),
                }
                for s in summaries
            ],
        )

    if args.out_debug_html:
        ok = _try_write_debug_html(
            args.out_debug_html,
            start_w=start_w,
            goal_w=goal_w,
            path_world=path_world,
            obstacle_hulls=obstacle_hulls,
        )
        if ok:
            try:
                import webbrowser

                webbrowser.open("file://" + os.path.abspath(args.out_debug_html))
            except Exception:
                pass


if __name__ == "__main__":
    main()

