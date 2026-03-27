from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .common import Point2D, Polygon2D, densify_path, is_segment_visible

BEZIER_ITERATIONS = 3
BEZIER_RESAMPLE_INTERVAL_M = 0.05


def get_bezier_smoothing_config() -> Dict[str, float]:
    return {
        "iterations": int(BEZIER_ITERATIONS),
        "resample_interval_m": float(BEZIER_RESAMPLE_INTERVAL_M),
    }


def _chaikin_smoothing(path_world: List[Point2D], iterations: int) -> List[Point2D]:
    if len(path_world) < 3 or iterations <= 0:
        return list(path_world)
    pts = [tuple(p) for p in path_world]
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts: List[Point2D] = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            q1 = (0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1])
            q2 = (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])
            new_pts.append(q1)
            new_pts.append(q2)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def _chaikin_selective_smoothing(path_world: List[Point2D], polys: List[Polygon2D]) -> Tuple[List[Point2D], int]:
    """충돌 없는 코너만 선택적으로 Chaikin 보정."""
    if len(path_world) < 3:
        return list(path_world), 0

    pts = [tuple(p) for p in path_world]
    out: List[Point2D] = [pts[0]]
    smoothed_corners = 0

    for i in range(1, len(pts) - 1):
        a = pts[i - 1]
        b = pts[i]
        c = pts[i + 1]

        q_in = (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])
        q_out = (0.75 * b[0] + 0.25 * c[0], 0.75 * b[1] + 0.25 * c[1])

        can_smooth = (
            is_segment_visible(out[-1], q_in, polys)
            and is_segment_visible(q_in, q_out, polys)
        )
        if can_smooth:
            out.append(q_in)
            out.append(q_out)
            smoothed_corners += 1
        else:
            # 해당 코너는 보정하지 않고 원래 점 유지
            if is_segment_visible(out[-1], b, polys):
                out.append(b)
            else:
                return list(path_world), 0

    if is_segment_visible(out[-1], pts[-1], polys):
        out.append(pts[-1])
    else:
        return list(path_world), 0
    return out, smoothed_corners


def _path_is_collision_free(path_world: List[Point2D], polys: List[Polygon2D]) -> bool:
    if len(path_world) < 2:
        return True
    for i in range(len(path_world) - 1):
        if not is_segment_visible(path_world[i], path_world[i + 1], polys):
            return False
    return True


def apply_bezier_smoothing(
    path_world: Optional[List[Point2D]],
    obstacle_polygons: List[Polygon2D],
) -> Tuple[Optional[List[Point2D]], bool, str]:
    if not path_world:
        return path_world, False, "empty_path"
    if len(path_world) < 3:
        return path_world, False, "too_few_points"

    # 반복을 누적 적용하되, 각 반복마다 코너 단위로 가능한 부분만 보정한다.
    best_safe = list(path_world)
    applied_iter = 0
    total_smoothed_corners = 0
    current = list(path_world)
    max_iter = max(1, int(BEZIER_ITERATIONS))

    for it in range(1, max_iter + 1):
        current, smoothed_corners = _chaikin_selective_smoothing(current, obstacle_polygons)
        cand = current
        if BEZIER_RESAMPLE_INTERVAL_M > 0.0:
            cand = densify_path(cand, interval_m=float(BEZIER_RESAMPLE_INTERVAL_M))
        if not _path_is_collision_free(cand, obstacle_polygons):
            if applied_iter > 0 and total_smoothed_corners > 0:
                return best_safe, True, f"partial_until_iter_{applied_iter}"
            return path_world, False, "collision_at_first_iter"
        if smoothed_corners == 0:
            break
        best_safe = cand
        applied_iter = it
        total_smoothed_corners += smoothed_corners

    if total_smoothed_corners == 0:
        return path_world, False, "no_smoothable_corner"
    return best_safe, True, f"applied_iter_{applied_iter}_corners_{total_smoothed_corners}"
