#!/usr/bin/env python3
"""
PCD 맵 전처리: localization용 global map 생성 (Open3D 미사용)
- 아웃라이어 제거 (Statistical, 1회)
- (기본) 그리드 해상도·패딩·분할 옵션으로 점유맵 → 군집화 → 볼록 껍질 다각형 → JSON 저장
- 결과를 HTML로 미리보기 (plotly)

사용법:
  python3 preprocess_pcd_map.py [입력.pcd] [출력.pcd]
  python3 preprocess_pcd_map.py [입력.pcd]              # 출력: 입력_global.pcd
  python3 preprocess_pcd_map.py                          # 기본: ./test.pcd -> ./test_global.pcd

  옵션: --no-html, --no-obstacles (장애물 JSON 생략)
  장애물 JSON 기본: <출력_stem>_obstacles.json (map_obstacles_v1, pcd_grid_path_planner 인자로 전달)
  출력 PCD는 binary(x,y,z float32)만 사용. 생성된 파일은 open3d_loc의 path_map(launch)에 지정.
"""
import sys
import os
import json
import math
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "test.pcd")
DEFAULT_PREVIEW_HTML = "global_map_preview.html"


@dataclass(frozen=True)
class _GridSpec:
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int


@dataclass
class _ObstacleSummary:
    id: int
    center: Tuple[float, float]
    size: Tuple[float, float]
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    cell_count: int


# 역할: 두 2D 선분이 교차(끝점 포함)하는지 판별한다.
def _segment_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
) -> bool:
    def orient(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
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


# 역할: 점이 다각형 내부 또는 경계에 있는지 검사한다.
def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    x, y = pt
    n = len(poly)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-9:
            if (
                min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
                and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
            ):
                return True
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1 + 1e-15)
            x_int = x1 + t * (x2 - x1)
            if x_int > x:
                inside = not inside
    return inside


# 역할: 볼록 다각형을 로봇 반경만큼 바깥으로 팽창한다.
def _inflate_convex_polygon(
    poly: List[Tuple[float, float]],
    r: float,
) -> List[Tuple[float, float]]:
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
        l1 = math.hypot(e1[0], e1[1])
        l2 = math.hypot(e2[0], e2[1])
        if l1 < 1e-12 or l2 < 1e-12:
            out.append(b)
            continue
        n1 = (e1[1] / l1, -e1[0] / l1)
        n2 = (e2[1] / l2, -e2[0] / l2)
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if cross < 0:
            n1 = (-n1[0], -n1[1])
            n2 = (-n2[0], -n2[1])
        p1 = (a[0] + r * n1[0], a[1] + r * n1[1])
        p2 = (b[0] + r * n1[0], b[1] + r * n1[1])
        q1 = (b[0] + r * n2[0], b[1] + r * n2[1])
        q2 = (c[0] + r * n2[0], c[1] + r * n2[1])
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = q2[0] - q1[0], q2[1] - q1[1]
        det = dx1 * dy2 - dy1 * dx2
        if abs(det) < 1e-12:
            out.append((b[0] + r * n1[0], b[1] + r * n1[1]))
            continue
        t = ((q1[0] - p1[0]) * dy2 - (q1[1] - p1[1]) * dx2) / det
        out.append((p1[0] + t * dx1, p1[1] + t * dy1))
    return out


# 역할: 장애물 hull로 정적 가시성 그래프(nodes/adj)를 오프라인 생성한다.
def _build_static_visibility_graph(
    obstacle_hulls: List[Dict[str, object]],
    robot_radius: float,
) -> Dict[str, object]:
    polys_with_id: List[Tuple[int, List[Tuple[float, float]]]] = []
    for item in obstacle_hulls:
        cid = int(item.get("id", 0))
        hull = np.asarray(item.get("hull", []), dtype=np.float64)
        if hull.ndim != 2 or hull.shape[0] < 3:
            continue
        poly = [(float(hull[k, 0]), float(hull[k, 1])) for k in range(hull.shape[0])]
        if robot_radius > 0.0:
            poly = _inflate_convex_polygon(poly, robot_radius)
        polys_with_id.append((cid, poly))

    nodes: List[Tuple[float, float]] = []
    for _, poly in polys_with_id:
        nodes.extend(poly)

    n_nodes = len(nodes)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]

    def visible(i: int, j: int) -> bool:
        p = nodes[i]
        q = nodes[j]
        if p == q:
            return False
        mid = ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)
        for _, poly in polys_with_id:
            if _point_in_polygon(mid, poly):
                return False
        for _, poly in polys_with_id:
            m = len(poly)
            for k in range(m):
                a, b = poly[k], poly[(k + 1) % m]
                if _segment_intersect(p, q, a, b):
                    if p == a or p == b or q == a or q == b:
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

    inflated_hulls = [{"id": cid, "hull": [[p[0], p[1]] for p in poly]} for cid, poly in polys_with_id]
    return {
        "version": "visibility_static_v1",
        "robot_radius": float(robot_radius),
        "nodes": [[p[0], p[1]] for p in nodes],
        "adj": [[[int(j), float(c)] for (j, c) in nbrs] for nbrs in adj],
        "inflated_obstacle_hulls": inflated_hulls,
    }


# 역할: resolution에 맞게 x,y 점을 격자점으로 생성
def _build_occupancy_grid(
    xy: np.ndarray,
    resolution: float,
    padding: float = 0.5,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, _GridSpec]:
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

    spec = _GridSpec(
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


# 역할: 장애물의 바깥점들을 이용해 볼록 다각형으로 계산
def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points는 (N,2) 형태여야 합니다.")
    if len(pts) == 0:
        return pts
    pts = np.unique(pts, axis=0)
    if len(pts) <= 1:
        return pts

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

    hull = np.concatenate([lower[:-1], upper[:-1]], axis=0)
    return hull


# 역할: 라벨링된 장애물 셀을 obstacle hull(id+hull) 목록으로 변환한다.
def _compute_obstacle_hulls(labels: np.ndarray, spec: _GridSpec) -> List[Dict[str, object]]:
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
        wx = spec.origin_x + (xs.astype(np.float64) + 0.5) * spec.resolution
        wy = spec.origin_y + (ys.astype(np.float64) + 0.5) * spec.resolution
        pts = np.column_stack([wx, wy])
        if pts.shape[0] < 3:
            hull_pts = pts
        else:
            hull_pts = _convex_hull_2d(pts)
        hulls.append({"id": cid, "hull": hull_pts.tolist()})
    return hulls


# 역할: 점유 격자에서 8-이웃 region growing으로 장애물 군집 라벨을 만든다.
def _cluster_obstacles_region_growing(occ: np.ndarray) -> np.ndarray:
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


# 역할: 너무 큰 장애물 라벨을 셀 수 상한 기준으로 여러 서브클러스터로 분할한다.
def _subdivide_obstacles_by_size(
    labels: np.ndarray,
    max_cells_per_subcluster: int,
) -> np.ndarray:
    if max_cells_per_subcluster <= 0:
        return labels

    h, w = labels.shape
    new_labels = np.zeros_like(labels, dtype=np.int32)
    next_id = 0

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    max_orig_id = int(labels.max())
    for orig_id in range(1, max_orig_id + 1):
        while True:
            ys, xs = np.nonzero((labels == orig_id) & (new_labels == 0))
            if xs.size == 0:
                break

            sy, sx = int(ys[0]), int(xs[0])
            next_id += 1
            q: Deque[Tuple[int, int]] = deque()
            q.append((sy, sx))
            new_labels[sy, sx] = next_id
            count = 1

            while q:
                cy, cx = q.popleft()
                if count >= max_cells_per_subcluster:
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


# 역할: 각 장애물의 중심/크기/bbox/셀 수 요약 정보를 계산한다.
def _summarize_obstacles(labels: np.ndarray, spec: _GridSpec) -> List[_ObstacleSummary]:
    h, w = labels.shape
    if h != spec.height or w != spec.width:
        raise ValueError("labels와 GridSpec 크기가 일치하지 않습니다.")

    max_id = int(labels.max())
    if max_id <= 0:
        return []

    summaries: List[_ObstacleSummary] = []
    for cid in range(1, max_id + 1):
        ys, xs = np.nonzero(labels == cid)
        if xs.size == 0:
            continue
        min_gx = int(xs.min())
        max_gx = int(xs.max())
        min_gy = int(ys.min())
        max_gy = int(ys.max())

        min_x = spec.origin_x + min_gx * spec.resolution
        max_x = spec.origin_x + (max_gx + 1) * spec.resolution
        min_y = spec.origin_y + min_gy * spec.resolution
        max_y = spec.origin_y + (max_gy + 1) * spec.resolution
        center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        size = (max_x - min_x, max_y - min_y)

        summaries.append(
            _ObstacleSummary(
                id=cid,
                center=center,
                size=size,
                bbox_min=(min_x, min_y),
                bbox_max=(max_x, max_y),
                cell_count=int(xs.size),
            )
        )
    return summaries


# 역할: XY 점군에서 장애물 추출 전체 파이프라인을 수행한다.
def extract_obstacle_data_from_xy(
    xy: np.ndarray,
    resolution: float,
    padding: float = 0.5,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_obstacle_cells: int = 0,
) -> Tuple[List[_ObstacleSummary], List[Dict[str, object]], np.ndarray, np.ndarray, _GridSpec]:
    occ, spec = _build_occupancy_grid(
        xy,
        resolution=resolution,
        padding=padding,
        bounds=bounds,
    )
    labels = _cluster_obstacles_region_growing(occ)
    if max_obstacle_cells > 0:
        labels = _subdivide_obstacles_by_size(
            labels, max_cells_per_subcluster=int(max_obstacle_cells)
        )
    summaries = _summarize_obstacles(labels, spec)
    obstacle_hulls = _compute_obstacle_hulls(labels, spec)
    return summaries, obstacle_hulls, occ, labels, spec


# 역할: 장애물/그리드/추출 설정/정적 그래프를 포함한 JSON 문서를 구성한다.
def build_obstacle_map_document(
    source_pcd: str,
    map_pcd: str,
    summaries: List[_ObstacleSummary],
    obstacle_hulls: List[Dict[str, object]],
    spec: _GridSpec,
    extraction: Dict[str, object],
    visibility_graph: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    hull_dict = {int(item["id"]): item.get("hull", []) for item in obstacle_hulls}
    obstacles_out: List[Dict[str, object]] = []
    for s in summaries:
        obstacles_out.append(
            {
                "id": s.id,
                "center": [s.center[0], s.center[1]],
                "size": [s.size[0], s.size[1]],
                "bbox_min": [s.bbox_min[0], s.bbox_min[1]],
                "bbox_max": [s.bbox_max[0], s.bbox_max[1]],
                "cell_count": s.cell_count,
                "hull": hull_dict.get(s.id),
            }
        )
    doc = {
        "format": "map_obstacles_v1",
        "source_pcd": source_pcd,
        "map_pcd": map_pcd,
        "grid": {
            "resolution": spec.resolution,
            "origin": [spec.origin_x, spec.origin_y],
            "width": spec.width,
            "height": spec.height,
        },
        "extraction": extraction,
        "obstacles": obstacles_out,
        "obstacle_hulls": obstacle_hulls,
    }
    if visibility_graph is not None:
        doc["visibility_graph"] = visibility_graph
    return doc


# 역할: map_obstacles_v1 문서를 JSON 파일로 저장한다.
def _write_map_obstacles_json(path: str, doc: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


# 축 범위: None = 자동, [최소, 최대] = 고정 (예: [-10, 10])
AXIS_X_RANGE = [-10.0, 10.0]
AXIS_Y_RANGE = [-10.0, 10.0]
AXIS_Z_RANGE = [-10.0, 0.5]


# 역할: PCD 파일에서 x,y,z를 numpy 배열로 로드한다.
def _read_pcd_numpy(path):
    """PCD에서 x,y,z만 읽기."""
    with open(path, "rb") as f:
        header = []
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
        except (ValueError, IndexError):
            x_idx, y_idx, z_idx = 0, 1, 2
        num_fields = len(fields)
        points_line = [l for l in header if l.startswith("POINTS ")][0]
        n_points = int(points_line.split()[1])
        if "DATA ascii" in " ".join(header):
            data = []
            for _ in range(n_points):
                line = f.readline().decode("ascii")
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            return np.array(data) if data else None
        size_line = [l for l in header if l.startswith("SIZE ")][0]
        sizes = list(map(int, size_line.split()[1:]))
        type_line = [l for l in header if l.startswith("TYPE ")][0]
        types = type_line.split()[1:]
        dtype_map = {"F": np.float32, "I": np.int32, "U": np.uint8}
        row_size = sum(sizes)
        buf = f.read(n_points * row_size)
        offsets = [sum(sizes[:k]) for k in range(num_fields)]
        xs = [np.frombuffer(buf[i * row_size + offsets[x_idx]:i * row_size + offsets[x_idx] + sizes[x_idx]], dtype=dtype_map.get(types[x_idx], np.float32))[0] for i in range(n_points)]
        ys = [np.frombuffer(buf[i * row_size + offsets[y_idx]:i * row_size + offsets[y_idx] + sizes[y_idx]], dtype=dtype_map.get(types[y_idx], np.float32))[0] for i in range(n_points)]
        zs = [np.frombuffer(buf[i * row_size + offsets[z_idx]:i * row_size + offsets[z_idx] + sizes[z_idx]], dtype=dtype_map.get(types[z_idx], np.float32))[0] for i in range(n_points)]
        return np.column_stack([xs, ys, zs])
    return None


# 역할: 통계 기반(outlier) 필터를 numpy만으로 계산한다.
def _outlier_remove_statistical_pure_numpy(xyz, k=20, std_ratio=2.0, batch_size=2000):
    """거리 기반 Statistical Outlier Removal (numpy만 사용).
    각 점에서 k개 최근접 이웃까지 평균 거리를 구하고,
    median + std_ratio*std 보다 큰 점을 아웃라이어로 제거.
    """
    n = len(xyz)
    k_use = min(k, n - 1)
    if k_use < 1:
        return xyz
    mean_d = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (end-start, n) 거리
        D_batch = np.sqrt(((xyz[start:end, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
        for i in range(end - start):
            d = D_batch[i].copy()
            d[start + i] = np.inf
            part = np.partition(d, k_use)[: k_use + 1]
            part.sort()
            mean_d[start + i] = part[1 : k_use + 1].mean()
    thresh = np.median(mean_d) + std_ratio * (np.std(mean_d) + 1e-9)
    return xyz[mean_d < thresh]


# 역할: numpy (N,3) 포인트를 binary PCD 형식으로 저장한다.
def _write_pcd_binary(xyz, path):
    """numpy (N,3)를 binary PCD로 저장 (x,y,z float32)."""
    n = len(xyz)
    xyz32 = np.ascontiguousarray(xyz, dtype=np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz32.tobytes())


# 역할: 전처리된 맵을 plotly 3D HTML로 시각화 저장한다.
def _write_preview_html(xyz, html_path, title="Global map (3D)"):
    """plotly로 3D scatter HTML 저장 (view_pcd.py와 동일 방식)."""
    n = len(xyz)
    if n > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=50000, replace=False)
        xyz_show = xyz[idx]
    else:
        xyz_show = xyz
    try:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(xyz_show, columns=["x", "y", "z"])
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="z", title=title)
        fig.update_traces(marker=dict(size=1.0, opacity=0.7))
        scene = {}
        if AXIS_X_RANGE is not None:
            scene["xaxis"] = dict(range=AXIS_X_RANGE)
        if AXIS_Y_RANGE is not None:
            scene["yaxis"] = dict(range=AXIS_Y_RANGE)
        if AXIS_Z_RANGE is not None:
            scene["zaxis"] = dict(range=AXIS_Z_RANGE)
        if scene:
            fig.update_layout(scene=scene)
        fig.write_html(html_path)
        return True
    except ImportError:
        return False
    except Exception:
        return False


# 역할: PCD 전처리(로딩/아웃라이어 제거/저장)를 실행한다.
def run_preprocess(input_path, output_path, nb_neighbors, std_ratio):
    """numpy만 사용한 전처리: outlier 제거(1회) → binary PCD 저장."""
    xyz = _read_pcd_numpy(input_path)
    if xyz is None or len(xyz) == 0:
        return False, "포인트를 읽을 수 없습니다."
    n_before = len(xyz)
    xyz = _outlier_remove_statistical_pure_numpy(xyz, k=nb_neighbors, std_ratio=std_ratio)
    n_after = len(xyz)
    _write_pcd_binary(xyz, output_path)
    return True, (n_before, n_after, xyz)


# 역할: CLI 인자를 파싱하고 전처리/장애물 JSON/미리보기를 오케스트레이션한다.
def main():
    parser = argparse.ArgumentParser(description="PCD 맵 전처리: outlier 제거 (1회, 3D 그대로 / Open3D 미사용)")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="입력 PCD 경로")
    parser.add_argument("output", nargs="?", default=None, help="출력 PCD 경로 (미지정 시 입력_global.pcd)")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="아웃라이어 이웃 수 (기본 20)")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="아웃라이어 표준편차 배수 (기본 2.0)")
    parser.add_argument("--no-html", action="store_true", help="HTML 미리보기 생성 생략")
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="다각형 장애물 JSON 생성 생략 (기본: 출력과 같은 stem에 _obstacles.json)",
    )
    parser.add_argument(
        "--out-obstacles",
        type=str,
        default=None,
        help="장애물 JSON 경로 (미지정 시 <출력_stem>_obstacles.json)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="장애물 추출 그리드 해상도 [m] (기본 0.05)",
    )
    parser.add_argument("--padding", type=float, default=0.5, help="점군 bbox 패딩 [m] (기본 0.5)")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help="그리드 bounds 고정 (미지정 시 점군+padding)",
    )
    parser.add_argument(
        "--max-obstacle-cells",
        type=int,
        default=300,
        help="큰 장애물 분할 시 셀 상한 (0이면 분할 안 함)",
    )
    parser.add_argument(
        "--planner-robot-radius",
        type=float,
        default=0.10,
        help="전처리 시 정적 가시 그래프에 적용할 로봇 반경 [m] (기본 0.10)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        print(f"파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_global" + ext
    else:
        output_path = args.output

    ok, result = run_preprocess(
        input_path,
        output_path,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )
    if not ok:
        print(result)
        sys.exit(1)
    n_before, n_after, xyz = result
    print(f"완료: {output_path}")
    print(f"  원본: {n_before} → outlier 제거: {n_after}")

    if not args.no_obstacles:
        if args.out_obstacles:
            obstacles_path = args.out_obstacles
            if not os.path.isabs(obstacles_path):
                obstacles_path = os.path.abspath(obstacles_path)
        else:
            base_out, _ = os.path.splitext(output_path)
            obstacles_path = base_out + "_obstacles.json"
        xy = xyz[:, :2].astype(np.float32, copy=False)
        bounds_t = tuple(args.bounds) if args.bounds is not None else None
        summaries, obstacle_hulls, _occ, _labels, spec = extract_obstacle_data_from_xy(
            xy,
            resolution=float(args.resolution),
            padding=float(args.padding),
            bounds=bounds_t,
            max_obstacle_cells=int(args.max_obstacle_cells),
        )
        doc = build_obstacle_map_document(
            source_pcd=input_path,
            map_pcd=output_path,
            summaries=summaries,
            obstacle_hulls=obstacle_hulls,
            spec=spec,
            extraction={
                "resolution": float(args.resolution),
                "padding": float(args.padding),
                "bounds": list(args.bounds) if args.bounds is not None else None,
                "max_obstacle_cells": int(args.max_obstacle_cells),
            },
            visibility_graph=_build_static_visibility_graph(
                obstacle_hulls=obstacle_hulls,
                robot_radius=max(0.0, float(args.planner_robot_radius)),
            ),
        )
        _write_map_obstacles_json(obstacles_path, doc)
        print(f"장애물 JSON: {obstacles_path} (obstacles={len(obstacle_hulls)})")

    if not args.no_html:
        html_path = os.path.join(SCRIPT_DIR, DEFAULT_PREVIEW_HTML)
        if _write_preview_html(xyz, html_path, title=os.path.basename(output_path) + " (3D)"):
            print(f"미리보기: {html_path}")
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))
        else:
            print("HTML 미리보기 실패 (plotly/pandas 필요: pip install plotly pandas)")


if __name__ == "__main__":
    main()
