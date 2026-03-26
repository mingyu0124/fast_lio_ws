#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node

from open3d_loc.srv import PlanPath


def _load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@dataclass
class _CachedMap:
    obstacle_hulls: List[dict]


class GridPathPlannerService(Node):
    def __init__(self) -> None:
        super().__init__("grid_path_planner_service")

        self.declare_parameter(
            "planner_py_path",
            "/home/robotics/fast_lio_ws/src/FAST_LIO_LOCALIZATION_HUMANOID/data/pcd_grid_path_planner.py",
        )
        self.declare_parameter(
            "obstacles_json_path",
            "",
        )
        self.declare_parameter("publish_topic", "/global_plan")
        self.declare_parameter("frame_id", "map")

        planner_py_path = str(self.get_parameter("planner_py_path").value)
        if not os.path.isabs(planner_py_path):
            planner_py_path = os.path.abspath(planner_py_path)
        if not os.path.isfile(planner_py_path):
            raise RuntimeError(f"planner_py_path not found: {planner_py_path}")

        self._planner = _load_module_from_path("pcd_grid_path_planner", planner_py_path)

        self._cached_map: Optional[_CachedMap] = None

        publish_topic = str(self.get_parameter("publish_topic").value)
        self._pub_path = self.create_publisher(Path, publish_topic, 10)

        self._srv = self.create_service(PlanPath, "plan_path", self._handle_plan_path)

        self.get_logger().info(f"Loaded planner from: {planner_py_path}")
        self._build_cache_or_throw()

    def _build_cache_or_throw(self) -> None:
        obstacles_json_path = str(self.get_parameter("obstacles_json_path").value).strip()
        if not obstacles_json_path:
            raise RuntimeError(
                "obstacles_json_path must be set (preprocess_pcd_map.py로 생성한 *_obstacles.json)"
            )
        if not os.path.isabs(obstacles_json_path):
            obstacles_json_path = os.path.abspath(obstacles_json_path)
        if not os.path.isfile(obstacles_json_path):
            raise RuntimeError(f"obstacles_json_path not found: {obstacles_json_path}")

        t0 = time.time()
        _obstacles, obstacle_hulls, _spec = self._planner.load_obstacle_map_json(obstacles_json_path)
        if not obstacle_hulls:
            raise RuntimeError("obstacles_json has no obstacle_hulls")
        self._cached_map = _CachedMap(obstacle_hulls=obstacle_hulls)
        dt = time.time() - t0
        self.get_logger().info(
            f"Cached obstacles from JSON: {len(obstacle_hulls)} ({obstacles_json_path}, {dt:.2f}s)"
        )

    def _pose_to_xy(self, pose: PoseStamped) -> Tuple[float, float]:
        return float(pose.pose.position.x), float(pose.pose.position.y)

    def _handle_plan_path(self, req: PlanPath.Request, resp: PlanPath.Response) -> PlanPath.Response:
        if self._cached_map is None:
            resp.success = False
            resp.message = "map cache not ready"
            resp.path = Path()
            return resp

        frame_id = str(self.get_parameter("frame_id").value)

        start_xy = self._pose_to_xy(req.start)
        goal_xy = self._pose_to_xy(req.goal)
        robot_radius = float(req.robot_radius) if req.robot_radius > 0.0 else 0.0
        interval = float(req.path_point_interval) if req.path_point_interval > 0.0 else 0.0

        try:
            nodes, adj = self._planner.build_visibility_graph_with_radius(
                self._cached_map.obstacle_hulls,
                start_w=start_xy,
                goal_w=goal_xy,
                robot_radius=robot_radius,
            )
            path_world = self._planner.shortest_path_visibility_graph(nodes, adj, start_idx=0, goal_idx=1)
            if path_world and interval > 0.0:
                path_world = self._planner.densify_path(path_world, interval_m=interval)
        except Exception as e:
            resp.success = False
            resp.message = f"planning failed: {e}"
            resp.path = Path()
            return resp

        if not path_world:
            resp.success = False
            resp.message = "no path found"
            resp.path = Path()
            return resp

        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = []
        for x, y in path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self._pub_path.publish(msg)

        resp.success = True
        resp.message = f"ok (n={len(msg.poses)})"
        resp.path = msg
        return resp


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = GridPathPlannerService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

