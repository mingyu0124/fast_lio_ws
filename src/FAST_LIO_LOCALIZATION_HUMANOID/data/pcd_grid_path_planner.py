#!/usr/bin/env python3
"""
2D PCD(혹은 3D PCD에서 Z 무시)로부터
- 3D->2D 투영(=Z 무시)
- 2D Occupancy Grid 생성
- 장애물 군집화(8-neighborhood region-growing)
- 장애물 요약(바운딩 박스/중심/크기)
- 시작/목표를 받아 Theta* 경로 생성

사용 예:
  python3 pcd_grid_path_planner.py \
    test_global.pcd \
    --start -0.0 -0.0 \
    --goal 2.0 -4.5

출력:
  - 장애물 요약: out_obstacles.json (옵션)
  - 경로: out_path.json (grid/world 좌표 포함)
  - 시각화: out_debug.html (plotly 설치 시)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

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


def dilate_occupancy(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    """로봇 반경 등을 반영하기 위한 간단한 팽창(Manhattan/원형 근사)."""
    if radius_cells <= 0:
        return occ
    h, w = occ.shape
    out = occ.copy()
    ys, xs = np.nonzero(occ)
    if len(xs) == 0:
        return out

    rr = radius_cells
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = max(0, y - rr)
        y1 = min(h - 1, y + rr)
        x0 = max(0, x - rr)
        x1 = min(w - 1, x + rr)
        for yy in range(y0, y1 + 1):
            dy = yy - y
            max_dx = int(math.floor(math.sqrt(max(0, rr * rr - dy * dy))))
            xx0 = max(x0, x - max_dx)
            xx1 = min(x1, x + max_dx)
            out[yy, xx0 : xx1 + 1] = True
    return out


@dataclass
class ObstacleSummary:
    id: int
    center: Tuple[float, float]
    size: Tuple[float, float]
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    cell_count: int


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


def theta_star_grid(
    occ: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diag: bool = True,
    neighbor_radius_cells: int = 1,
) -> Optional[List[Tuple[int, int]]]:
    """
    Theta*: Any-angle grid path planning.
    탐색 중 neighbor를 current가 아니라 current의 parent로 직접 연결(LOS 가능 시)하여
    지그재그를 줄이고 직선에 가까운 경로를 유도합니다.
    """
    h, w = occ.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if occ[sy, sx] or occ[gy, gx]:
        return None

    if neighbor_radius_cells < 1:
        neighbor_radius_cells = 1

    def make_neighbor_offsets(radius: int, diag: bool) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if not diag and (dx != 0 and dy != 0):
                    continue
                offsets.append((dx, dy))
        return offsets

    nbrs = make_neighbor_offsets(neighbor_radius_cells, allow_diag)

    def bresenham_cells(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx_ = 1 if x0 < x1 else -1
        sy_ = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx_
            if e2 <= dx:
                err += dx
                y += sy_

    def segment_is_free(x0: int, y0: int, x1: int, y1: int) -> bool:
        first = True
        for x, y in bresenham_cells(x0, y0, x1, y1):
            if first:
                first = False
                continue
            if not (0 <= x < w and 0 <= y < h):
                return False
            if occ[y, x]:
                return False
        return True

    def heuristic(ax: int, ay: int) -> float:
        dx = abs(ax - gx)
        dy = abs(ay - gy)
        return math.hypot(dx, dy)

    import heapq

    open_heap: List[Tuple[float, int, int]] = []
    heapq.heappush(open_heap, (heuristic(sx, sy), sx, sy))

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {(sx, sy): (sx, sy)}
    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[sy, sx] = 0.0
    closed = np.zeros((h, w), dtype=bool)

    while open_heap:
        _, cx, cy = heapq.heappop(open_heap)
        if closed[cy, cx]:
            continue
        if (cx, cy) == (gx, gy):
            path = [(cx, cy)]
            while (cx, cy) != (sx, sy):
                cx, cy = parent[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path
        closed[cy, cx] = True

        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if closed[ny, nx] or occ[ny, nx]:
                continue
            if not segment_is_free(cx, cy, nx, ny):
                continue

            px, py = parent.get((cx, cy), (cx, cy))
            if (px, py) != (cx, cy) and segment_is_free(px, py, nx, ny):
                cand_parent = (px, py)
                cand_g = float(g_score[py, px] + math.hypot(nx - px, ny - py))
            else:
                cand_parent = (cx, cy)
                cand_g = float(g_score[cy, cx] + math.hypot(dx, dy))

            if cand_g < float(g_score[ny, nx]):
                g_score[ny, nx] = cand_g
                parent[(nx, ny)] = cand_parent
                f = cand_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f, nx, ny))
    return None


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
        title="Map (PCD xy) + obstacles (grid) + path (world XY)",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    fig.write_html(out_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="PCD 기반 2D 그리드 경로 생성기")
    parser.add_argument("pcd", help="입력 PCD 경로")
    parser.add_argument("--resolution", type=float, default=0.05, help="그리드 해상도 [m] (기본 0.05)")
    parser.add_argument("--padding", type=float, default=0.5, help="포인트 bbox 외곽 패딩 [m] (기본 0.5)")
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"), help="그리드 bounds 고정")
    parser.add_argument("--robot-radius", type=float, default=0.25, help="장애물 팽창 반경 [m] (기본 0.25)")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="시작점 world 좌표 [m]")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"), help="목표점 world 좌표 [m]")
    parser.add_argument("--no-diag", action="store_true", help="Theta*에서 대각 이동 금지")
    parser.add_argument(
        "--neighbor-radius-cells",
        type=int,
        default=1,
        help="Theta* 이웃 탐색 반경(셀 단위). 1=8방향, 2=24방향 근사 (기본 1)",
    )
    parser.add_argument("--out", type=str, default="out_path.json", help="경로 출력 JSON (기본 out_path.json)")
    parser.add_argument("--out-obstacles", type=str, default="out_obstacles.json", help="장애물 요약 JSON 출력 (옵션)")
    parser.add_argument("--out-debug-html", type=str, default="out_debug.html", help="디버그 HTML 출력 (plotly 필요)")
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
    occ, spec = build_occupancy_grid(xy, resolution=args.resolution, padding=args.padding, bounds=tuple(args.bounds) if args.bounds else None)

    # 장애물 군집화: robot_radius 없이 원본 그리드만으로 이어진 셀 = 하나의 장애물
    labels = cluster_obstacles_region_growing(occ)
    summaries = summarize_obstacles(labels, spec)

    # 경로 계획용만 robot_radius 반영 (팽창)
    if args.robot_radius > 0:
        radius_cells = int(math.ceil(args.robot_radius / spec.resolution))
        occ = dilate_occupancy(occ, radius_cells=radius_cells)

    start_w = (float(args.start[0]), float(args.start[1]))
    goal_w = (float(args.goal[0]), float(args.goal[1]))
    start_g = spec.world_to_grid(*start_w)
    goal_g = spec.world_to_grid(*goal_w)

    path_g = theta_star_grid(
        occ,
        start=start_g,
        goal=goal_g,
        allow_diag=not args.no_diag,
        neighbor_radius_cells=int(args.neighbor_radius_cells),
    )
    if path_g is None:
        path_world: Optional[List[Tuple[float, float]]] = None
    else:
        path_world = [spec.grid_to_world_center(gx, gy) for gx, gy in path_g]

    out_obj: Dict[str, object] = {
        "pcd": pcd_path,
        "grid": {
            "resolution": spec.resolution,
            "origin": [spec.origin_x, spec.origin_y],
            "width": spec.width,
            "height": spec.height,
        },
        "planner": {
            "name": "theta",
            "allow_diag": not args.no_diag,
            "neighbor_radius_cells": int(args.neighbor_radius_cells),
        },
        "start": {"world": [start_w[0], start_w[1]], "grid": [start_g[0], start_g[1]]},
        "goal": {"world": [goal_w[0], goal_w[1]], "grid": [goal_g[0], goal_g[1]]},
        "path": {
            "found": path_world is not None,
            "grid": [[gx, gy] for (gx, gy) in path_g] if path_g else None,
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
            }
            for s in summaries
        ],
    }

    _write_json(args.out, out_obj)

    if args.out_obstacles:
        _write_json(
            args.out_obstacles,
            [{"id": s.id, "center": s.center, "size": s.size, "bbox_min": s.bbox_min, "bbox_max": s.bbox_max, "cell_count": s.cell_count} for s in summaries],
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
        )
        if ok:
            try:
                import webbrowser

                webbrowser.open("file://" + os.path.abspath(args.out_debug_html))
            except Exception:
                pass


if __name__ == "__main__":
    main()

