#!/usr/bin/env python3
"""
2D PCD로부터 다각형 장애물을 추출하고, 가시 그래프(visibility graph)로 경로를 생성합니다.

흐름:
- 3D PCD → 2D 투영 (Z 무시)
- 2D Occupancy Grid → 장애물 군집화(8-neighborhood) → 볼록 껍질(hull) 다각형
- (옵션) 큰 장애물을 --max-obstacle-cells 기준으로 세분화
- 장애물 다각형을 --robot-radius 만큼 팽창 후 가시 그래프 구축
- 시작/목표 사이 최단 경로 (world 좌표) 계산

사용 예 (한 줄로 실행, 백슬래시 줄나눔 없이):
  python3 pcd_grid_path_planner.py test_global.pcd --start 0.0 0.0 --goal 0.0 -5.0 --max-obstacle-cells 100 --out-debug-html out_debug.html



출력:
  - 경로: out_path.json (world 좌표)
  - 장애물 요약: out_obstacles.json (옵션)
  - 시각화: out_debug.html (plotly 필요)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


def read_pcd_xyz(path: str) -> np.ndarray:
    """PCD 파일에서 x,y,z 포인트만 읽기 (ASCII/binary 모두 시도)."""
    with open(path, "rb") as f:
        header: List[str] = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line.startswith("DATA"):
                break

        fields_line = [l for l in header if l.startswith("FIELDS ")][0]
        fields = fields_line.split()[1:]
        try:
            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
        except ValueError:
            x_idx, y_idx, z_idx = 0, 1, 2
        num_fields = len(fields)

        points_line = [l for l in header if l.startswith("POINTS ")][0]
        n_points = int(points_line.split()[1])

        if "DATA ascii" in " ".join(header):
            data: List[List[float]] = []
            for _ in range(n_points):
                line = f.readline().decode("ascii")
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            return np.array(data, dtype=np.float32) if data else np.zeros((0, 3), dtype=np.float32)

        size_line = [l for l in header if l.startswith("SIZE ")][0]
        sizes = list(map(int, size_line.split()[1:]))
        type_line = [l for l in header if l.startswith("TYPE ")][0]
        types = type_line.split()[1:]
        dtype_map = {"F": np.float32, "I": np.int32, "U": np.uint8}
        row_size = sum(sizes)
        buf = f.read(n_points * row_size)

        offsets = [sum(sizes[:k]) for k in range(num_fields)]
        xs, ys, zs = [], [], []
        for i in range(n_points):
            base = i * row_size
            x = np.frombuffer(
                buf[base + offsets[x_idx] : base + offsets[x_idx] + sizes[x_idx]],
                dtype=dtype_map.get(types[x_idx], np.float32),
            )[0]
            y = np.frombuffer(
                buf[base + offsets[y_idx] : base + offsets[y_idx] + sizes[y_idx]],
                dtype=dtype_map.get(types[y_idx], np.float32),
            )[0]
            z = np.frombuffer(
                buf[base + offsets[z_idx] : base + offsets[z_idx] + sizes[z_idx]],
                dtype=dtype_map.get(types[z_idx], np.float32),
            )[0]
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
        return np.column_stack([xs, ys, zs]).astype(np.float32, copy=False)


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


def build_occupancy_grid(
    xy: np.ndarray,
    resolution: float,
    padding: float = 0.5,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, GridSpec]:
    """
    xy: (N,2) world points
    returns:
      occ: (H,W) bool (True=occupied)
      spec: grid spec
    """
    if xy.size == 0:
        raise ValueError("입력 포인트가 비어 있습니다.")
    if resolution <= 0:
        raise ValueError("resolution은 0보다 커야 합니다.")

    if bounds is None:
        min_x = float(np.min(xy[:, 0])) - padding
        max_x = float(np.max(xy[:, 0])) + padding
        min_y = float(np.min(xy[:, 1])) - padding
        max_y = float(np.max(xy[:, 1])) + padding
    else:
        min_x, max_x, min_y, max_y = bounds

    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    width = max(width, 1)
    height = max(height, 1)

    spec = GridSpec(
        resolution=resolution,
        origin_x=min_x,
        origin_y=min_y,
        width=width,
        height=height,
    )

    gx = np.floor((xy[:, 0] - spec.origin_x) / resolution).astype(np.int32)
    gy = np.floor((xy[:, 1] - spec.origin_y) / resolution).astype(np.int32)
    valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    gx = gx[valid]
    gy = gy[valid]
    occ = np.zeros((height, width), dtype=bool)
    if gx.size:
        lin = gy.astype(np.int64) * width + gx.astype(np.int64)
        lin = np.unique(lin)
        occ.flat[lin] = True
    return occ, spec


@dataclass
class ObstacleSummary:
    id: int
    center: Tuple[float, float]
    size: Tuple[float, float]
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    cell_count: int


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    2D 점 집합의 볼록 껍질을 Monotone chain 알고리즘으로 계산.
    입력: (N,2) np.ndarray
    출력: (M,2) np.ndarray (시계/반시계 정렬, 시작/끝 점 동일하게 닫지는 않음)
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points는 (N,2) 형태여야 합니다.")
    # 중복 제거
    if len(pts) == 0:
        return pts
    pts = np.unique(pts, axis=0)
    if len(pts) <= 1:
        return pts

    # x, 그 다음 y 기준 정렬
    pts_sorted = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 마지막 점은 중복이므로 제거
    hull = np.concatenate([lower[:-1], upper[:-1]], axis=0)
    return hull


def compute_obstacle_hulls(labels: np.ndarray, spec: GridSpec) -> List[Dict[str, object]]:
    """
    각 장애물(cluster id별) 그리드 셀의 중심 점들을 이용해서
    볼록 껍질(외곽 선)을 world 좌표계에서 계산.

    returns:
      [{"id": cid, "hull": [[x0,y0], [x1,y1], ...]}, ...]
    """
    h, w = labels.shape
    if h != spec.height or w != spec.width:
        raise ValueError("labels와 GridSpec 크기가 일치하지 않습니다.")

    max_id = int(labels.max())
    if max_id <= 0:
        return []

    hulls: List[Dict[str, object]] = []
    for cid in range(1, max_id + 1):
        ys, xs = np.nonzero(labels == cid)
        if xs.size == 0:
            continue
        # 각 occupied 셀의 중심을 world 좌표로 변환
        wx = spec.origin_x + (xs.astype(np.float64) + 0.5) * spec.resolution
        wy = spec.origin_y + (ys.astype(np.float64) + 0.5) * spec.resolution
        pts = np.column_stack([wx, wy])
        if pts.shape[0] < 3:
            # 점이 2개 이하라면 그대로 사용
            hull_pts = pts
        else:
            hull_pts = _convex_hull_2d(pts)
        hulls.append({"id": cid, "hull": hull_pts.tolist()})
    return hulls


def cluster_obstacles_region_growing(occ: np.ndarray) -> np.ndarray:
    """
    이어지는 occupied 셀을 하나의 장애물로, 떨어진 덩어리는 서로 다른 장애물로 구분.
    8방향(상하좌우+대각선) region-growing으로 클러스터링.
    returns labels: (H,W) int32, 0=free, 1..K=장애물 id (연결된 덩어리마다 1씩 증가)
    """
    h, w = occ.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cid = 0

    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if not occ[y, x] or labels[y, x] != 0:
                continue
            cid += 1
            q: Deque[Tuple[int, int]] = deque()
            q.append((y, x))
            labels[y, x] = cid
            while q:
                cy, cx = q.popleft()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and occ[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = cid
                        q.append((ny, nx))
    return labels


def subdivide_obstacles_by_size(
    labels: np.ndarray,
    max_cells_per_subcluster: int,
) -> np.ndarray:
    """
    큰 장애물 클러스터를 여러 개의 작은 연결 성분으로 다시 쪼갠다.

    - 입력 labels: (H,W) int32, 0=free, 1..K=클러스터 id
    - max_cells_per_subcluster: 한 서브 클러스터가 가질 수 있는 최대 셀 수

    아이디어:
      한 클러스터(id)를 대상으로, 아직 배정되지 않은 셀에서 BFS를 시작하되
      방문 셀 수가 max_cells_per_subcluster를 넘으면 큐에 남은 셀을
      다음 서브 클러스터로 넘기면서 같은 방식으로 반복한다.
      이렇게 하면 긴 벽 모양 장애물이 둘레를 따라 여러 조각으로 잘려 나간다.
    """
    if max_cells_per_subcluster <= 0:
        return labels

    h, w = labels.shape
    new_labels = np.zeros_like(labels, dtype=np.int32)
    next_id = 0

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    max_orig_id = int(labels.max())
    for orig_id in range(1, max_orig_id + 1):
        # 해당 원본 클러스터에 속한 셀들 중, 아직 새 id가 없는 것들만 대상으로 반복
        while True:
            ys, xs = np.nonzero((labels == orig_id) & (new_labels == 0))
            if xs.size == 0:
                break

            # 아직 배정되지 않은 셀 하나에서 BFS 시작
            sy, sx = int(ys[0]), int(xs[0])
            next_id += 1
            q: Deque[Tuple[int, int]] = deque()
            q.append((sy, sx))
            new_labels[sy, sx] = next_id
            count = 1

            while q:
                cy, cx = q.popleft()
                if count >= max_cells_per_subcluster:
                    # 현재 서브 클러스터의 크기를 제한하기 위해,
                    # 큐에 남아 있던 셀들은 다음 서브 클러스터에서 사용
                    q.clear()
                    break
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if labels[ny, nx] != orig_id:
                        continue
                    if new_labels[ny, nx] != 0:
                        continue
                    new_labels[ny, nx] = next_id
                    count += 1
                    q.append((ny, nx))

    return new_labels


def summarize_obstacles(labels: np.ndarray, spec: GridSpec) -> List[ObstacleSummary]:
    h, w = labels.shape
    if h != spec.height or w != spec.width:
        raise ValueError("labels와 GridSpec 크기가 일치하지 않습니다.")

    max_id = int(labels.max())
    if max_id <= 0:
        return []

    summaries: List[ObstacleSummary] = []
    for cid in range(1, max_id + 1):
        ys, xs = np.nonzero(labels == cid)
        if xs.size == 0:
            continue
        min_gx = int(xs.min())
        max_gx = int(xs.max())
        min_gy = int(ys.min())
        max_gy = int(ys.max())

        # world bbox (cell boundary 기준)
        min_x = spec.origin_x + min_gx * spec.resolution
        max_x = spec.origin_x + (max_gx + 1) * spec.resolution
        min_y = spec.origin_y + min_gy * spec.resolution
        max_y = spec.origin_y + (max_gy + 1) * spec.resolution
        center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        size = (max_x - min_x, max_y - min_y)

        summaries.append(
            ObstacleSummary(
                id=cid,
                center=center,
                size=size,
                bbox_min=(min_x, min_y),
                bbox_max=(max_x, max_y),
                cell_count=int(xs.size),
            )
        )
    return summaries


# ----- 다각형 기반 가시 그래프 경로 생성용 기하 유틸 -----


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


def _point_segment_distance(pt: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """점과 선분 a-b 사이 최소 거리."""
    px, py = pt
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-15:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


def _segment_segment_distance(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
) -> float:
    """두 선분 사이 최소 거리. 교차 시 0."""
    if _segment_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        _point_segment_distance(p1, q1, q2),
        _point_segment_distance(p2, q1, q2),
        _point_segment_distance(q1, p1, p2),
        _point_segment_distance(q2, p1, p2),
    )


def _inflate_convex_polygon(
    poly: Sequence[Sequence[float]],
    r: float,
) -> List[Tuple[float, float]]:
    """
    볼록 다각형을 바깥으로 r만큼 팽창한 꼭짓점 리스트 반환.
    CCW 또는 CW 모두: 각 edge를 바깥 방향으로 r 밀어서 생기는 새 꼭짓점을 구함.
    """
    if r <= 0.0 or len(poly) < 3:
        return [(float(p[0]), float(p[1])) for p in poly]
    n = len(poly)
    out: List[Tuple[float, float]] = []
    for i in range(n):
        a = (float(poly[(i - 1) % n][0]), float(poly[(i - 1) % n][1]))
        b = (float(poly[i][0]), float(poly[i][1]))
        c = (float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1]))
        e1 = (b[0] - a[0], b[1] - a[1])
        e2 = (c[0] - b[0], c[1] - b[1])
        L1 = math.hypot(e1[0], e1[1])
        L2 = math.hypot(e2[0], e2[1])
        if L1 < 1e-12 or L2 < 1e-12:
            out.append(b)
            continue
        # 바깥 법선: 다각형이 CCW면 왼쪽이 내부 -> 법선은 (e.y, -e.x) 방향이 바깥
        n1 = (e1[1] / L1, -e1[0] / L1)
        n2 = (e2[1] / L2, -e2[0] / L2)
        # 시계 방향이면 반대로
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if cross < 0:
            n1 = (-n1[0], -n1[1])
            n2 = (-n2[0], -n2[1])
        p1 = (a[0] + r * n1[0], a[1] + r * n1[1])
        p2 = (b[0] + r * n1[0], b[1] + r * n1[1])
        q1 = (b[0] + r * n2[0], b[1] + r * n2[1])
        q2 = (c[0] + r * n2[0], c[1] + r * n2[1])
        # 직선 p1-p2 와 q1-q2 의 교점
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = q2[0] - q1[0], q2[1] - q1[1]
        det = dx1 * dy2 - dy1 * dx2
        if abs(det) < 1e-12:
            out.append((b[0] + r * n1[0], b[1] + r * n1[1]))
            continue
        t = ((q1[0] - p1[0]) * dy2 - (q1[1] - p1[1]) * dx2) / det
        px = p1[0] + t * dx1
        py = p1[1] + t * dy1
        out.append((px, py))
    return out


def build_visibility_graph_with_radius(
    obstacle_hulls: List[Dict[str, object]],
    start_w: Tuple[float, float],
    goal_w: Tuple[float, float],
    robot_radius: float = 0.0,
) -> Tuple[List[Tuple[float, float]], List[List[Tuple[int, float]]]]:
    """
    다각형 장애물 기준 가시 그래프 구축.

    원리:
      - 노드 = 시작점, 목표점, (팽창된) 장애물 꼭짓점들.
      - 두 노드가 '서로 보인다' = 그 사이 직선이 어떤 장애물 내부도 통과하지 않고
        어떤 장애물 변과도 교차하지 않음 (끝점에서 맞닿는 것은 허용).
      - robot_radius > 0 이면: 먼저 각 장애물 다각형을 바깥으로 robot_radius 만큼
        팽창(inflate)한 뒤, 이 팽창된 다각형을 장애물로 해서 위 조건으로 가시 그래프를
        만든다. 이렇게 하면 경로는 원래 장애물에서 최소 robot_radius 만큼 떨어지게 됨.

    returns: (nodes, adj)  nodes[0]=start, nodes[1]=goal, 이후 (팽창된) hull 꼭짓점. adj[i] = [(j, cost), ...]
    """
    # robot_radius > 0 이면 장애물을 팽창한 다각형을 사용 (그래프 노드/가시성 모두 팽창된 것 기준)
    use_inflated = robot_radius > 0.0
    polys: List[List[Tuple[float, float]]] = []
    for item in obstacle_hulls:
        hull = np.asarray(item.get("hull", []), dtype=np.float64)
        if hull.ndim != 2 or hull.shape[0] < 3:
            continue
        poly_list = [(float(hull[k, 0]), float(hull[k, 1])) for k in range(hull.shape[0])]
        if use_inflated:
            poly_list = _inflate_convex_polygon(poly_list, robot_radius)
        polys.append(poly_list)

    nodes: List[Tuple[float, float]] = [tuple(start_w), tuple(goal_w)]
    for poly in polys:
        for pt in poly:
            nodes.append(pt)

    n_nodes = len(nodes)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]

    def visible(i: int, j: int) -> bool:
        p, q = nodes[i], nodes[j]
        if p == q:
            return False
        # 세그먼트 중간점이 어떤 (팽창된) 다각형 내부에 있으면 비가시
        mid = ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)
        for poly in polys:
            if _point_in_polygon(mid, poly):
                return False
        # 세그먼트가 어떤 다각형의 변과 교차하면 비가시 (끝점이 그 변의 끝이면 허용)
        for poly in polys:
            m = len(poly)
            for k in range(m):
                a, b = poly[k], poly[(k + 1) % m]
                if _segment_intersect(p, q, a, b):
                    if (p == a or p == b or q == a or q == b):
                        continue
                    return False
        return True

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not visible(i, j):
                continue
            cost = float(math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]))
            adj[i].append((j, cost))
            adj[j].append((i, cost))
    return nodes, adj


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


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _try_write_debug_html(
    out_path: str,
    map_xy: np.ndarray,
    occ: np.ndarray,
    spec: GridSpec,
    start_w: Tuple[float, float],
    goal_w: Tuple[float, float],
    path_world: Optional[List[Tuple[float, float]]],
    labels: Optional[np.ndarray] = None,
    obstacle_hulls: Optional[List[Dict[str, object]]] = None,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    # map points (original XY)
    map_xy = np.asarray(map_xy)
    if map_xy.ndim != 2 or map_xy.shape[1] != 2:
        return False
    n_map = int(map_xy.shape[0])
    if n_map > 80000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_map, size=80000, replace=False)
        map_xy_show = map_xy[idx]
    else:
        map_xy_show = map_xy

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=map_xy_show[:, 0],
            y=map_xy_show[:, 1],
            mode="markers",
            marker=dict(size=3, color="dimgray", opacity=0.75),
            name="map (PCD xy)",
        )
    )

    # 장애물별 그리드 셀을 색깔로 표시 (labels: 0=free, 1..K=장애물 id)
    grid_colors = [
        "crimson", "darkorange", "gold", "limegreen", "dodgerblue",
        "mediumpurple", "cyan", "hotpink", "peru", "teal",
    ]
    if labels is not None and labels.size > 0:
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
        title="PCD + 장애물 다각형 + 경로 (world XY)",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    fig.write_html(out_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="PCD 기반 2D 다각형 가시 그래프 경로 생성기")
    parser.add_argument("pcd", help="입력 PCD 경로")
    parser.add_argument("--resolution", type=float, default=0.1, help="장애물 추출용 그리드 해상도 [m] (기본 0.05)")
    parser.add_argument("--padding", type=float, default=0.5, help="포인트 bbox 외곽 패딩 [m] (기본 0.5)")
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"), help="그리드 bounds 고정")
    parser.add_argument("--robot-radius", type=float, default=0.10, help="다각형 팽창 반경 [m] (장애물에서 이만큼 떨어진 경로, 기본 0.15)")
    parser.add_argument("--path-point-interval", type=float, default=0.1, help="경로를 저장/시각화할 때 이 거리[m]마다 점을 보간 (0이면 꺾이는 지점만). 기본 0.1")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="시작점 world 좌표 [m]")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"), help="목표점 world 좌표 [m]")
    parser.add_argument("--out", type=str, default="out_path.json", help="경로 출력 JSON (기본 out_path.json)")
    parser.add_argument("--out-obstacles", type=str, default="out_obstacles.json", help="장애물 요약 JSON 출력 (옵션)")
    parser.add_argument("--out-debug-html", type=str, default="out_debug.html", help="디버그 HTML 출력 (plotly 필요)")
    parser.add_argument(
        "--max-obstacle-cells",
        type=int,
        default=0,
        help="장애물 하나가 가질 수 있는 최대 셀 수. "
        "0 이하면 분할하지 않음. 값이 작을수록 하나의 큰 장애물이 여러 조각으로 잘게 나뉨.",
    )
    args = parser.parse_args()

    pcd_path = args.pcd
    if not os.path.isabs(pcd_path):
        pcd_path = os.path.abspath(pcd_path)
    if not os.path.isfile(pcd_path):
        raise SystemExit(f"파일을 찾을 수 없습니다: {pcd_path}")

    xyz = read_pcd_xyz(pcd_path)
    if xyz.size == 0:
        raise SystemExit("PCD 포인트가 비어 있습니다.")

    xy = xyz[:, :2].astype(np.float32, copy=False)  # Step2: Z 무시 (투영)
    occ, spec = build_occupancy_grid(
        xy,
        resolution=args.resolution,
        padding=args.padding,
        bounds=tuple(args.bounds) if args.bounds else None,
    )

    # 장애물 군집화: robot_radius 없이 원본 그리드만으로 이어진 셀 = 하나의 장애물
    labels = cluster_obstacles_region_growing(occ)
    # 옵션에 따라 큰 장애물을 여러 개의 작은 서브 클러스터로 분할
    if int(args.max_obstacle_cells) > 0:
        labels = subdivide_obstacles_by_size(labels, max_cells_per_subcluster=int(args.max_obstacle_cells))
    summaries = summarize_obstacles(labels, spec)
    obstacle_hulls = compute_obstacle_hulls(labels, spec)
    hull_dict = {int(item["id"]): item.get("hull", []) for item in obstacle_hulls}

    start_w = (float(args.start[0]), float(args.start[1]))
    goal_w = (float(args.goal[0]), float(args.goal[1]))

    # 다각형 가시 그래프로 경로 생성 (robot_radius = 다각형 팽창 반경)
    nodes, adj = build_visibility_graph_with_radius(
        obstacle_hulls,
        start_w=start_w,
        goal_w=goal_w,
        robot_radius=max(0.0, float(args.robot_radius)),
    )
    path_world = shortest_path_visibility_graph(nodes, adj, start_idx=0, goal_idx=1)
    if path_world and args.path_point_interval > 0:
        path_world = densify_path(path_world, interval_m=float(args.path_point_interval))

    out_obj: Dict[str, object] = {
        "pcd": pcd_path,
        "grid": {
            "resolution": spec.resolution,
            "origin": [spec.origin_x, spec.origin_y],
            "width": spec.width,
            "height": spec.height,
        },
        "planner": {"name": "visibility", "robot_radius": float(args.robot_radius)},
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
            map_xy=xy,
            occ=occ,
            spec=spec,
            start_w=start_w,
            goal_w=goal_w,
            path_world=path_world,
            labels=labels,
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

