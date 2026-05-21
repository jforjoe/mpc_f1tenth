#!/usr/bin/env python3
"""
Laptop-side WebSocket bridge for kinematic_mpc.

Runs on the remote laptop as a WebSocket CLIENT.
  - Connects to the F1tenth compute's WebSocket server and republishes all
    robot data (odom, scan, IMU, TF, MPC visualizations) as local ROS2 topics.
  - Subscribes to laptop-originated topics (/initialpose, /goal_pose from RViz)
    and forwards them over the WebSocket back to the robot.

Usage:
    python3 bridge_recv.py [--robot-ip 192.168.1.100] [--port 9090]

Environment variables:
    ROBOT_IP      F1tenth compute IP  (default: 192.168.123.160)
    BRIDGE_PORT   WebSocket port      (default: 9090)
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
from geometry_msgs.msg import (
    TransformStamped, PoseWithCovarianceStamped, PoseStamped, Point,
)
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from std_msgs.msg import Int32

import asyncio
import websockets
import json
import argparse
import os
from collections import deque

# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.123.160")
DEFAULT_PORT     = int(os.environ.get("BRIDGE_PORT", "9090"))
RECONNECT_BASE   = 2.0    # initial retry delay (seconds)
RECONNECT_MAX    = 30.0   # maximum retry delay (seconds)

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
# Must be TRANSIENT_LOCAL so RViz receives the map even if it subscribes after publish
QOS_MAP = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# ─── Serializers (local ROS → JSON for sending to robot) ──────────────────────

def _stamp_from_ros(stamp) -> dict:
    return {"sec": stamp.sec, "nanosec": stamp.nanosec}


def ser_initialpose(msg: PoseWithCovarianceStamped) -> dict:
    return {
        "topic":    "/initialpose",
        "frame_id": msg.header.frame_id,
        "stamp":    _stamp_from_ros(msg.header.stamp),
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
    }


def ser_goal_pose(msg: PoseStamped) -> dict:
    return {
        "topic":    "/goal_pose",
        "frame_id": msg.header.frame_id,
        "stamp":    _stamp_from_ros(msg.header.stamp),
        "pose": {
            "position":    {"x": msg.pose.position.x,
                            "y": msg.pose.position.y,
                            "z": msg.pose.position.z},
            "orientation": {"x": msg.pose.orientation.x,
                            "y": msg.pose.orientation.y,
                            "z": msg.pose.orientation.z,
                            "w": msg.pose.orientation.w},
        },
    }


# ─── ROS2 Node ─────────────────────────────────────────────────────────────────

class LaptopBridge(Node):
    """
    Receives JSON from the robot WebSocket server and republishes as ROS2
    topics for RViz. Also listens for RViz events (/initialpose, /goal_pose)
    and queues them for transmission back to the robot.
    """

    def __init__(self, robot_ip: str, port: int):
        super().__init__("laptop_ws_bridge")
        self.uri = f"ws://{robot_ip}:{port}"

        # Ring buffer of outgoing JSON strings (laptop → robot)
        self._outgoing: deque = deque(maxlen=50)

        # ── Publishers (robot data → local ROS2 for RViz) ────────────────────
        self.pub_odom            = self.create_publisher(Odometry, "/odom", QOS_RELIABLE)
        self.pub_odom_filtered   = self.create_publisher(Odometry, "/odometry/filtered", QOS_RELIABLE)
        self.pub_scan            = self.create_publisher(LaserScan, "/scan", QOS_RELIABLE)
        self.pub_imu             = self.create_publisher(Imu, "/sensors/imu", QOS_SENSOR)
        self.pub_imu_raw         = self.create_publisher(Imu, "/sensors/imu/raw", QOS_SENSOR)
        self.pub_mpc_raceline    = self.create_publisher(Marker, "/mpc/raceline", 5)
        self.pub_mpc_horizon     = self.create_publisher(Marker, "/mpc/horizon", 5)
        self.pub_mpc_ref_horizon = self.create_publisher(Marker, "/mpc/ref_horizon", 5)
        self.pub_mpc_lap         = self.create_publisher(Int32, "/mpc/lap", 5)
        self.pub_drive           = self.create_publisher(AckermannDriveStamped, "/drive", QOS_RELIABLE)
        self.pub_map             = self.create_publisher(OccupancyGrid, "/map", QOS_MAP)

        # TF broadcasters
        self.tf_broadcaster        = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ── Subscriptions (local RViz events → robot via WebSocket) ──────────
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose",
            lambda msg: self._outgoing.append(json.dumps(ser_initialpose(msg))),
            QOS_RELIABLE)

        self.create_subscription(
            PoseStamped, "/goal_pose",
            lambda msg: self._outgoing.append(json.dumps(ser_goal_pose(msg))),
            QOS_RELIABLE)

        # ── Incoming message dispatch table ──────────────────────────────────
        # Maps topic string → handler method
        self._dispatch = {
            "/odom":              self._pub_odom,
            "/odometry/filtered": self._pub_odom_filtered,
            "/scan":              self._pub_scan,
            "/sensors/imu":       lambda d: self._pub_imu(d, self.pub_imu),
            "/sensors/imu/raw":   lambda d: self._pub_imu(d, self.pub_imu_raw),
            "/tf":                lambda d: self._pub_tf(d, static=False),
            "/tf_static":         lambda d: self._pub_tf(d, static=True),
            "/mpc/raceline":      lambda d: self._pub_marker(d, self.pub_mpc_raceline),
            "/mpc/horizon":       lambda d: self._pub_marker(d, self.pub_mpc_horizon),
            "/mpc/ref_horizon":   lambda d: self._pub_marker(d, self.pub_mpc_ref_horizon),
            "/mpc/lap":           self._pub_lap,
            "/drive":             self._pub_drive,
            "/map":               self._pub_map,
        }

        self.get_logger().info(f"LaptopBridge ready → {self.uri}")

    # ── Top-level message router ──────────────────────────────────────────────

    def handle_message(self, raw: str):
        try:
            data  = json.loads(raw)
            topic = data.get("topic")
            fn    = self._dispatch.get(topic)
            if fn:
                fn(data)
            else:
                self.get_logger().warn(f"[RX] Unhandled topic: {topic!r}")
        except Exception as e:
            self.get_logger().error(f"[RX] Error: {e}  raw={raw[:120]!r}")

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _fill_header(self, msg, data: dict):
        msg.header.frame_id      = data.get("frame_id", "")
        s = data.get("stamp", {})
        msg.header.stamp.sec     = s.get("sec", 0)
        msg.header.stamp.nanosec = s.get("nanosec", 0)

    def _build_odom(self, data: dict) -> Odometry:
        msg = Odometry()
        self._fill_header(msg, data)
        msg.child_frame_id = data.get("child_frame_id", "base_link")

        p = data["pose"]["position"]
        msg.pose.pose.position.x = p["x"]
        msg.pose.pose.position.y = p["y"]
        msg.pose.pose.position.z = p["z"]

        o = data["pose"]["orientation"]
        msg.pose.pose.orientation.x = o["x"]
        msg.pose.pose.orientation.y = o["y"]
        msg.pose.pose.orientation.z = o["z"]
        msg.pose.pose.orientation.w = o["w"]
        msg.pose.covariance = data["pose"].get("covariance", [0.0] * 36)

        l = data["twist"]["linear"]
        msg.twist.twist.linear.x = l["x"]
        msg.twist.twist.linear.y = l["y"]
        msg.twist.twist.linear.z = l["z"]

        a = data["twist"]["angular"]
        msg.twist.twist.angular.x = a["x"]
        msg.twist.twist.angular.y = a["y"]
        msg.twist.twist.angular.z = a["z"]
        msg.twist.covariance = data["twist"].get("covariance", [0.0] * 36)
        return msg

    # ── Publish methods ───────────────────────────────────────────────────────

    def _pub_odom(self, data: dict):
        self.pub_odom.publish(self._build_odom(data))

    def _pub_odom_filtered(self, data: dict):
        self.pub_odom_filtered.publish(self._build_odom(data))

    def _pub_scan(self, data: dict):
        msg = LaserScan()
        self._fill_header(msg, data)
        msg.angle_min       = data["angle_min"]
        msg.angle_max       = data["angle_max"]
        msg.angle_increment = data["angle_increment"]
        msg.time_increment  = data["time_increment"]
        msg.scan_time       = data["scan_time"]
        msg.range_min       = data["range_min"]
        msg.range_max       = data["range_max"]
        msg.ranges          = data["ranges"]
        msg.intensities     = data.get("intensities", [])
        self.pub_scan.publish(msg)

    def _pub_imu(self, data: dict, publisher):
        msg = Imu()
        self._fill_header(msg, data)

        ori = data["orientation"]
        msg.orientation.x = ori["x"]
        msg.orientation.y = ori["y"]
        msg.orientation.z = ori["z"]
        msg.orientation.w = ori["w"]
        msg.orientation_covariance = ori.get("covariance", [0.0] * 9)

        av = data["angular_velocity"]
        msg.angular_velocity.x = av["x"]
        msg.angular_velocity.y = av["y"]
        msg.angular_velocity.z = av["z"]
        msg.angular_velocity_covariance = av.get("covariance", [0.0] * 9)

        la = data["linear_acceleration"]
        msg.linear_acceleration.x = la["x"]
        msg.linear_acceleration.y = la["y"]
        msg.linear_acceleration.z = la["z"]
        msg.linear_acceleration_covariance = la.get("covariance", [0.0] * 9)

        publisher.publish(msg)

    def _pub_tf(self, data: dict, static: bool = False):
        transforms = []
        for t in data.get("transforms", []):
            tf = TransformStamped()
            tf.header.frame_id       = t["frame_id"]
            tf.child_frame_id        = t["child_frame_id"]
            tf.header.stamp.sec      = t["stamp"]["sec"]
            tf.header.stamp.nanosec  = t["stamp"]["nanosec"]
            tf.transform.translation.x = t["translation"]["x"]
            tf.transform.translation.y = t["translation"]["y"]
            tf.transform.translation.z = t["translation"]["z"]
            tf.transform.rotation.x  = t["rotation"]["x"]
            tf.transform.rotation.y  = t["rotation"]["y"]
            tf.transform.rotation.z  = t["rotation"]["z"]
            tf.transform.rotation.w  = t["rotation"]["w"]
            transforms.append(tf)
        if not transforms:
            return
        if static:
            self.static_tf_broadcaster.sendTransform(transforms)
        else:
            self.tf_broadcaster.sendTransform(transforms)

    def _pub_marker(self, data: dict, publisher):
        msg = Marker()
        self._fill_header(msg, data)
        msg.ns     = data.get("ns", "")
        msg.id     = data.get("id", 0)
        msg.type   = data.get("type", Marker.LINE_STRIP)
        msg.action = data.get("action", Marker.ADD)
        msg.scale.x = data.get("scale_x", 0.05)
        msg.scale.y = data.get("scale_y", 0.05)
        msg.scale.z = data.get("scale_z", 0.05)
        c = data.get("color", {})
        msg.color.r = c.get("r", 1.0)
        msg.color.g = c.get("g", 1.0)
        msg.color.b = c.get("b", 1.0)
        msg.color.a = c.get("a", 1.0)
        for px, py, pz in data.get("points", []):
            p = Point()
            p.x = px; p.y = py; p.z = pz
            msg.points.append(p)
        publisher.publish(msg)

    def _pub_lap(self, data: dict):
        msg = Int32()
        msg.data = data["lap"]
        self.pub_mpc_lap.publish(msg)

    def _pub_drive(self, data: dict):
        msg = AckermannDriveStamped()
        self._fill_header(msg, data)
        msg.drive.steering_angle          = data.get("steering_angle", 0.0)
        msg.drive.steering_angle_velocity = data.get("steering_angle_velocity", 0.0)
        msg.drive.speed                   = data.get("speed", 0.0)
        msg.drive.acceleration            = data.get("acceleration", 0.0)
        msg.drive.jerk                    = data.get("jerk", 0.0)
        self.pub_drive.publish(msg)

    def _pub_map(self, data: dict):
        msg = OccupancyGrid()
        self._fill_header(msg, data)
        msg.info.resolution = data["resolution"]
        msg.info.width      = data["width"]
        msg.info.height     = data["height"]
        origin = data["origin"]
        p = origin["position"]
        msg.info.origin.position.x = p["x"]
        msg.info.origin.position.y = p["y"]
        msg.info.origin.position.z = p["z"]
        o = origin["orientation"]
        msg.info.origin.orientation.x = o["x"]
        msg.info.origin.orientation.y = o["y"]
        msg.info.origin.orientation.z = o["z"]
        msg.info.origin.orientation.w = o["w"]
        msg.data = data["data"]
        self.pub_map.publish(msg)
        self.get_logger().info(
            f"[MAP] Published {data['width']}×{data['height']} "
            f"@ {data['resolution']}m/cell")


# ─── WebSocket client ──────────────────────────────────────────────────────────

async def run_bridge(node: LaptopBridge, executor: SingleThreadedExecutor):

    # ── ROS2 spin (interleaved with asyncio) ──────────────────────────────────
    async def ros_spin_loop():
        while rclpy.ok():
            executor.spin_once(timeout_sec=0)
            await asyncio.sleep(0.001)

    # ── Main WebSocket loop with exponential-backoff reconnect ────────────────
    async def ws_loop():
        delay = RECONNECT_BASE
        while True:
            try:
                node.get_logger().info(f"[WS] Connecting to {node.uri} ...")
                async with websockets.connect(
                    node.uri,
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=2 ** 22,   # 4 MB — handles full LaserScan
                ) as ws:
                    node.get_logger().info("[WS] Connected!")
                    delay = RECONNECT_BASE  # reset backoff on successful connect

                    # Receive robot data and publish locally
                    async def recv_loop():
                        async for message in ws:
                            node.handle_message(message)

                    # Forward local RViz events to robot
                    async def send_loop():
                        try:
                            while True:
                                if node._outgoing:
                                    data = node._outgoing.popleft()
                                    await ws.send(data)
                                    node.get_logger().info(
                                        f"[TX] → {json.loads(data).get('topic')}")
                                else:
                                    await asyncio.sleep(0.01)
                        except websockets.exceptions.ConnectionClosed:
                            pass

                    await asyncio.gather(recv_loop(), send_loop())

            except (websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError, OSError) as e:
                node.get_logger().warn(
                    f"[WS] Disconnected: {e}. Retrying in {delay:.1f}s ...")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, RECONNECT_MAX)  # exponential backoff

    await asyncio.gather(ros_spin_loop(), ws_loop())


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Laptop-side WebSocket bridge (client)")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                        help="F1tenth compute IP (default: 192.168.123.160)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="WebSocket port (default: 9090)")
    args = parser.parse_args()

    rclpy.init()
    node = LaptopBridge(robot_ip=args.robot_ip, port=args.port)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        asyncio.run(run_bridge(node, executor))
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down laptop bridge...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
