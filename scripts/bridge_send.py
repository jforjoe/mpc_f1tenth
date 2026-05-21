#!/usr/bin/env python3
"""
Robot-side WebSocket bridge for kinematic_mpc.

Runs on the F1tenth compute as a WebSocket SERVER.
  - Subscribes to robot ROS2 topics and broadcasts them as JSON to connected clients.
  - Receives JSON messages from clients (e.g., /initialpose from RViz) and
    republishes them to local ROS2 topics.

Usage:
    python3 bridge_send.py [--host 0.0.0.0] [--port 9090]

Environment variables:
    BRIDGE_HOST   Bind address  (default: 0.0.0.0)
    BRIDGE_PORT   Bind port     (default: 9090)
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Imu
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int32

import asyncio
import websockets
import json
import time
import argparse
import os
from collections import deque
from typing import Set

# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("BRIDGE_PORT", "9090"))

# Maximum forward rate (Hz) per outbound topic — 0 = no limit
RATE_LIMITS: dict = {
    "/odom":                20.0,
    "/odometry/filtered":   20.0,
    "/scan":                10.0,   # LaserScan is large; cap at 10 Hz
    "/sensors/imu":         20.0,
    "/sensors/imu/raw":     10.0,
    "/tf":                   0.0,   # no limit — dropped frames break RViz pose display
    "/tf_static":            0.0,   # always forward static transforms
    "/mpc/raceline":         2.0,   # static path; very low refresh needed
    "/mpc/horizon":         20.0,
    "/mpc/ref_horizon":     20.0,
    "/mpc/lap":              0.0,   # event-driven; always forward
    "/drive":               10.0,
    "/map":                  0.0,   # latched; forward every time it arrives
}

# ─── QoS profiles ──────────────────────────────────────────────────────────────

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
QOS_SENSOR = qos_profile_sensor_data
QOS_TF_STATIC = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# map_server publishes /map as TRANSIENT_LOCAL (latched) — must match to receive it
QOS_MAP = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# ─── Serializers (ROS → JSON) ──────────────────────────────────────────────────

def _stamp(s) -> dict:
    return {"sec": s.sec, "nanosec": s.nanosec}


def ser_odom(msg: Odometry, topic: str) -> dict:
    return {
        "topic": topic,
        "frame_id": msg.header.frame_id,
        "stamp": _stamp(msg.header.stamp),
        "child_frame_id": msg.child_frame_id,
        "pose": {
            "position":    {"x": msg.pose.pose.position.x,
                            "y": msg.pose.pose.position.y,
                            "z": msg.pose.pose.position.z},
            "orientation": {"x": msg.pose.pose.orientation.x,
                            "y": msg.pose.pose.orientation.y,
                            "z": msg.pose.pose.orientation.z,
                            "w": msg.pose.pose.orientation.w},
            "covariance":  list(msg.pose.covariance),
        },
        "twist": {
            "linear":      {"x": msg.twist.twist.linear.x,
                            "y": msg.twist.twist.linear.y,
                            "z": msg.twist.twist.linear.z},
            "angular":     {"x": msg.twist.twist.angular.x,
                            "y": msg.twist.twist.angular.y,
                            "z": msg.twist.twist.angular.z},
            "covariance":  list(msg.twist.covariance),
        },
    }


def ser_scan(msg: LaserScan) -> dict:
    return {
        "topic":           "/scan",
        "frame_id":        msg.header.frame_id,
        "stamp":           _stamp(msg.header.stamp),
        "angle_min":       msg.angle_min,
        "angle_max":       msg.angle_max,
        "angle_increment": msg.angle_increment,
        "time_increment":  msg.time_increment,
        "scan_time":       msg.scan_time,
        "range_min":       msg.range_min,
        "range_max":       msg.range_max,
        "ranges":          list(msg.ranges),
        "intensities":     list(msg.intensities),
    }


def ser_imu(msg: Imu, topic: str) -> dict:
    return {
        "topic":    topic,
        "frame_id": msg.header.frame_id,
        "stamp":    _stamp(msg.header.stamp),
        "orientation": {
            "x": msg.orientation.x, "y": msg.orientation.y,
            "z": msg.orientation.z, "w": msg.orientation.w,
            "covariance": list(msg.orientation_covariance),
        },
        "angular_velocity": {
            "x": msg.angular_velocity.x, "y": msg.angular_velocity.y,
            "z": msg.angular_velocity.z,
            "covariance": list(msg.angular_velocity_covariance),
        },
        "linear_acceleration": {
            "x": msg.linear_acceleration.x, "y": msg.linear_acceleration.y,
            "z": msg.linear_acceleration.z,
            "covariance": list(msg.linear_acceleration_covariance),
        },
    }


def ser_tf(msg: TFMessage, topic: str) -> dict:
    transforms = []
    for t in msg.transforms:
        transforms.append({
            "frame_id":       t.header.frame_id,
            "child_frame_id": t.child_frame_id,
            "stamp":          _stamp(t.header.stamp),
            "translation":    {"x": t.transform.translation.x,
                               "y": t.transform.translation.y,
                               "z": t.transform.translation.z},
            "rotation":       {"x": t.transform.rotation.x,
                               "y": t.transform.rotation.y,
                               "z": t.transform.rotation.z,
                               "w": t.transform.rotation.w},
        })
    return {"topic": topic, "transforms": transforms}


def ser_marker(msg: Marker, topic: str) -> dict:
    return {
        "topic":    topic,
        "frame_id": msg.header.frame_id,
        "stamp":    _stamp(msg.header.stamp),
        "ns":       msg.ns,
        "id":       msg.id,
        "type":     msg.type,
        "action":   msg.action,
        "scale_x":  msg.scale.x,
        "scale_y":  msg.scale.y,
        "scale_z":  msg.scale.z,
        "color":    {"r": msg.color.r, "g": msg.color.g,
                     "b": msg.color.b, "a": msg.color.a},
        "points":   [[p.x, p.y, p.z] for p in msg.points],
    }


def ser_drive(msg: AckermannDriveStamped) -> dict:
    return {
        "topic":                    "/drive",
        "frame_id":                 msg.header.frame_id,
        "stamp":                    _stamp(msg.header.stamp),
        "steering_angle":           msg.drive.steering_angle,
        "steering_angle_velocity":  msg.drive.steering_angle_velocity,
        "speed":                    msg.drive.speed,
        "acceleration":             msg.drive.acceleration,
        "jerk":                     msg.drive.jerk,
    }


def ser_lap(msg: Int32) -> dict:
    return {"topic": "/mpc/lap", "lap": msg.data}


def ser_map(msg: OccupancyGrid) -> dict:
    return {
        "topic":      "/map",
        "frame_id":   msg.header.frame_id,
        "stamp":      _stamp(msg.header.stamp),
        "resolution": msg.info.resolution,
        "width":      msg.info.width,
        "height":     msg.info.height,
        "origin": {
            "position":    {"x": msg.info.origin.position.x,
                            "y": msg.info.origin.position.y,
                            "z": msg.info.origin.position.z},
            "orientation": {"x": msg.info.origin.orientation.x,
                            "y": msg.info.origin.orientation.y,
                            "z": msg.info.origin.orientation.z,
                            "w": msg.info.origin.orientation.w},
        },
        "data": list(msg.data),   # flat int8 array: -1=unknown, 0=free, 100=occupied
    }


# ─── Deserializers (JSON → ROS) ────────────────────────────────────────────────

def deser_initialpose(data: dict) -> PoseWithCovarianceStamped:
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = data.get("frame_id", "map")
    s = data.get("stamp", {})
    msg.header.stamp.sec     = s.get("sec", 0)
    msg.header.stamp.nanosec = s.get("nanosec", 0)
    pose = data["pose"]
    p = pose["position"]
    msg.pose.pose.position.x = p["x"]
    msg.pose.pose.position.y = p["y"]
    msg.pose.pose.position.z = p["z"]
    o = pose["orientation"]
    msg.pose.pose.orientation.x = o["x"]
    msg.pose.pose.orientation.y = o["y"]
    msg.pose.pose.orientation.z = o["z"]
    msg.pose.pose.orientation.w = o["w"]
    msg.pose.covariance = pose.get("covariance", [0.0] * 36)
    return msg


def deser_goal_pose(data: dict) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = data.get("frame_id", "map")
    s = data.get("stamp", {})
    msg.header.stamp.sec     = s.get("sec", 0)
    msg.header.stamp.nanosec = s.get("nanosec", 0)
    pose = data["pose"]
    p = pose["position"]
    msg.pose.position.x = p["x"]
    msg.pose.position.y = p["y"]
    msg.pose.position.z = p["z"]
    o = pose["orientation"]
    msg.pose.orientation.x = o["x"]
    msg.pose.orientation.y = o["y"]
    msg.pose.orientation.z = o["z"]
    msg.pose.orientation.w = o["w"]
    return msg


# ─── ROS2 Node ─────────────────────────────────────────────────────────────────

class RobotBridge(Node):
    """
    Subscribes to local robot topics, serialises them into JSON, and
    queues them for WebSocket broadcast. Also handles inbound JSON from
    the laptop and republishes as local ROS2 topics.
    """

    def __init__(self):
        super().__init__("robot_ws_bridge")

        # Ring buffer of outgoing JSON strings (robot → laptop)
        self._outgoing: deque = deque(maxlen=500)
        self._last_sent: dict = {}
        # Last payload for latched topics — replayed to every new client on connect
        self._latch_cache: dict = {}

        # ── Publishers (laptop → robot) ──────────────────────────────────────
        self.pub_initialpose = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)
        self.pub_goal_pose = self.create_publisher(
            PoseStamped, "/goal_pose", 10)

        # ── Subscriptions (robot → laptop via WebSocket) ─────────────────────
        # self.create_subscription(
        #     Odometry, "/odom",
        #     lambda m: self._enqueue(ser_odom(m, "/odom"), "/odom"),
        #     QOS_RELIABLE)

        # self.create_subscription(
        #     Odometry, "/odometry/filtered",
        #     lambda m: self._enqueue(ser_odom(m, "/odometry/filtered"), "/odometry/filtered"),
        #     QOS_RELIABLE)

        self.create_subscription(
            LaserScan, "/scan",
            lambda m: self._enqueue(ser_scan(m), "/scan"),
            QOS_SENSOR)

        # self.create_subscription(
        #     Imu, "/sensors/imu",
        #     lambda m: self._enqueue(ser_imu(m, "/sensors/imu"), "/sensors/imu"),
        #     QOS_SENSOR)

        # self.create_subscription(
        #     Imu, "/sensors/imu/raw",
        #     lambda m: self._enqueue(ser_imu(m, "/sensors/imu/raw"), "/sensors/imu/raw"),
        #     QOS_SENSOR)

        self.create_subscription(
            TFMessage, "/tf",
            lambda m: self._enqueue(ser_tf(m, "/tf"), "/tf"),
            QOS_RELIABLE)

        self.create_subscription(
            TFMessage, "/tf_static",
            lambda m: self._enqueue(ser_tf(m, "/tf_static"), "/tf_static"),
            QOS_TF_STATIC)

        self.create_subscription(
            Marker, "/mpc/raceline",
            lambda m: self._enqueue(ser_marker(m, "/mpc/raceline"), "/mpc/raceline"),
            10)

        self.create_subscription(
            Marker, "/mpc/horizon",
            lambda m: self._enqueue(ser_marker(m, "/mpc/horizon"), "/mpc/horizon"),
            10)

        self.create_subscription(
            Marker, "/mpc/ref_horizon",
            lambda m: self._enqueue(ser_marker(m, "/mpc/ref_horizon"), "/mpc/ref_horizon"),
            10)

        self.create_subscription(
            Int32, "/mpc/lap",
            lambda m: self._enqueue(ser_lap(m), "/mpc/lap"),
            10)

        self.create_subscription(
            AckermannDriveStamped, "/drive",
            lambda m: self._enqueue(ser_drive(m), "/drive"),
            QOS_RELIABLE)

        self.create_subscription(
            OccupancyGrid, "/map",
            lambda m: self._enqueue(ser_map(m), "/map"),
            QOS_MAP)

        self.get_logger().info("RobotBridge ready")

    # ── Rate limiter ─────────────────────────────────────────────────────────

    def _should_send(self, topic: str) -> bool:
        rate = RATE_LIMITS.get(topic, 0.0)
        if rate <= 0.0:
            return True
        now = time.monotonic()
        if now - self._last_sent.get(topic, 0.0) >= 1.0 / rate:
            self._last_sent[topic] = now
            return True
        return False

    def _enqueue(self, payload: dict, topic: str):
        if self._should_send(topic):
            serialized = json.dumps(payload)
            if topic in ("/tf_static", "/map"):
                self._latch_cache[topic] = serialized
            self._outgoing.append(serialized)

    # ── Inbound handler (laptop → robot) ─────────────────────────────────────

    def handle_client_message(self, raw: str):
        try:
            data = json.loads(raw)
            topic = data.get("topic")
            if topic == "/initialpose":
                self.pub_initialpose.publish(deser_initialpose(data))
                self.get_logger().info("[RX] /initialpose published")
            elif topic == "/goal_pose":
                self.pub_goal_pose.publish(deser_goal_pose(data))
                self.get_logger().info("[RX] /goal_pose published")
            else:
                self.get_logger().warn(f"[RX] Unhandled inbound topic: {topic!r}")
        except Exception as e:
            self.get_logger().error(f"[RX] Parse error: {e}  raw={raw[:120]!r}")


# ─── WebSocket server ──────────────────────────────────────────────────────────

async def run_server(node: RobotBridge, executor: SingleThreadedExecutor,
                     host: str, port: int):

    clients: Set = set()

    # ── Per-client handler ────────────────────────────────────────────────────
    async def client_handler(websocket):
        clients.add(websocket)
        addr = websocket.remote_address
        node.get_logger().info(
            f"[WS] Client connected: {addr}  (total={len(clients)})")

        # Immediately replay /tf_static and /map so RViz has them even if
        # they were published before this client connected or after a reconnect.
        for topic, payload in node._latch_cache.items():
            try:
                await websocket.send(payload)
                node.get_logger().info(f"[WS] Replayed latched {topic!r} → {addr}")
            except Exception as e:
                node.get_logger().warn(f"[WS] Replay failed for {topic!r}: {e}")

        try:
            async for message in websocket:
                node.handle_client_message(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            clients.discard(websocket)
            node.get_logger().info(
                f"[WS] Client disconnected: {addr}  (total={len(clients)})")

    # ── ROS2 spin (interleaved with asyncio) ──────────────────────────────────
    async def ros_spin_loop():
        while rclpy.ok():
            executor.spin_once(timeout_sec=0)
            await asyncio.sleep(0.001)  # ~1 kHz poll keeps ROS responsive

    # ── Drain outgoing queue and broadcast to all clients ─────────────────────
    async def broadcast_loop():
        while True:
            while node._outgoing:
                data = node._outgoing.popleft()
                if clients:
                    results = await asyncio.gather(
                        *[c.send(data) for c in list(clients)],
                        return_exceptions=True,
                    )
                    for r in results:
                        if isinstance(r, Exception):
                            node.get_logger().debug(f"[WS] Broadcast error: {r}")
            await asyncio.sleep(0.001)

    # ── Start server ──────────────────────────────────────────────────────────
    async with websockets.serve(
        client_handler, host, port,
        ping_interval=20,
        ping_timeout=30,
        max_size=2 ** 22,   # 4 MB — accommodates full LaserScan payloads
    ):
        node.get_logger().info(f"[WS] Server listening on ws://{host}:{port}")
        await asyncio.gather(ros_spin_loop(), broadcast_loop())


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot-side WebSocket bridge (server)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port (default: 9090)")
    args = parser.parse_args()

    rclpy.init()
    node = RobotBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        asyncio.run(run_server(node, executor, args.host, args.port))
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down robot bridge...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
