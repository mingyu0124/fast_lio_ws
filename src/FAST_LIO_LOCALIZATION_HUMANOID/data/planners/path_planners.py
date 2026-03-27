from __future__ import annotations

from .common import PlannerRequest, PlannerResult
from .rrt_star import plan_rrt_star
from .visibility import plan_visibility_from_static


def plan_path(req: PlannerRequest) -> PlannerResult:
    if req.planner_name == "visibility":
        return plan_visibility_from_static(req)
    if req.planner_name == "rrt_star":
        return plan_rrt_star(req)
    raise ValueError(f"지원하지 않는 planner_name: {req.planner_name}")
