#!/usr/bin/env python3
"""ROS2 node: receding-horizon Kinematic MPC for F1TENTH.

All tunable parameters live at the top of this file.
"""

import math
import os

import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory as _get_pkg_share
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.executors import MultiThreadedExecutor

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

from .mpc_solver import KinematicMPC
from .waypoint_utils import load_waypoints, compute_yaw_from_path, nearest_index, extract_horizon


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


# =====================================================================
#                       TUNABLE PARAMETERS
# =====================================================================

# --- ROS topics / frames ---
ODOM_TOPIC      = '/odometry/filtered'     # EKF output (velocity only; pose comes from TF)
DRIVE_TOPIC     = '/drive'
MAP_FRAME       = 'map'
BASE_FRAME      = 'base_link'

# --- Waypoints ---
def _resolve_waypoints_csv() -> str:
    for pkg in ('kinematic_mpc', 'ebot_nav2'):
        try:
            return os.path.join(_get_pkg_share(pkg), 'config', 'raceline.csv')
        except Exception:
            pass
    # Fallback: config/ sibling of this file's package root
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'raceline.csv')

WAYPOINTS_CSV   = _resolve_waypoints_csv()
WP_LOOP         = True
WP_REVERSE      = False    # set True if CSV is ordered against the car's driving direction

# --- MPC horizon ---
N               = 40        # was 15 — 40×0.05 = 2 s preview; car sees turns ~1.6 m ahead at 0.8 m/s
DT              = 0.05
CTRL_RATE_HZ    = 20.0


SPEED_SIGN  = -1.0   # flip if VESC erpm_gain is negative (default F1TENTH)
STEER_SIGN  =  1.0   # flip if car steers wrong way in auto mode


# --- Vehicle (from vesc.yaml hardware calibration) ---
# vesc_to_odom_node.wheelbase = 0.325 m
WHEELBASE       = 0.325
# servo_min=0.15 → (0.15-0.48)/(-1.2135) = 0.272 rad; confirmed by nav2 min_turning_radius = 1.17 m
MAX_STEER       = 0.272    # rad
# throttle_interpolator.max_servo_speed = 3.2 rad/s
MAX_STEER_VEL   = 1.0      # rad/s
# throttle_interpolator.max_acceleration = 2.5 m/s²
MAX_ACCEL       = 1.0      # m/s²
# speed_min=-23250 ERPM / |gain=4100| = 5.67 m/s; cap 0.17 m/s below for margin
MAX_SPEED       = 1.5      # m/s  (raise in 0.5 m/s steps once tracking is stable)
MIN_SPEED       = 0.5

# --- Reference ---
# Speed cap applied to every waypoint's vx_mps value from the CSV.
# Lower this to slow the entire raceline without regenerating the CSV.
# Raise it (up to MAX_SPEED) to let the car run at the CSV's optimised speeds.
TARGET_SPEED    = 1.5      # m/s  — start here; raise in 0.5 m/s steps once tracking is stable

# --- Cost weights ---
Q_X, Q_Y, Q_YAW, Q_V    = 20.0, 20.0, 20.0, 0.5  # Q_YAW raised to 20 — earlier yaw alignment in turns
R_STEER_VEL, R_ACCEL    = 0.01, 0.01           # raised from 0.001 — damps abrupt steer commands
RD_STEER_VEL, RD_ACCEL  = 1.0, 0.5             # raised RD_STEER_VEL — smoother steering rate changes
QF_SCALE                = 2.0                  # raised from 1.0 — stronger pull toward end of horizon

# --- Solver ---
IPOPT_PRINT_LEVEL = 0     # 0=silent (deployment), 5=verbose (debugging only)
IPOPT_MAX_ITER    = 100     # 50 is plenty at 20 Hz; raise to 100 only when debugging convergence

# --- Debug / visualization ---
# Set True to publish RViz markers: /mpc/raceline (green), /mpc/horizon (blue), /mpc/ref_horizon (yellow)
PUBLISH_MARKERS  = True

# --- Steering-only debug mode ---
# True  → MPC computes steering from map; speed is copied from the RC /teleop topic.
#         Keep the AUTO switch ON so the mux routes /drive to the VESC, then control
#         speed manually with the throttle stick.  Steering should track the raceline.
# False → normal MPC: both speed and steering are commanded by the solver.
STEER_ONLY_MODE  = True
TELEOP_TOPIC     = '/teleop'   # AckermannDriveStamped from crsf_teleop (mux.yaml)

# Warn when AMCL pose jumps more than this between control cycles (wheel-slip / EKF glitch).
MAX_POS_JUMP_M   = 0.3   # metres — anything above this in 50 ms is physically impossible

# =====================================================================


class KinematicMPCNode(Node):
    def __init__(self):
        super().__init__('kinematic_mpc_node')

        # Waypoints
        self.wps = load_waypoints(WAYPOINTS_CSV)
        if WP_REVERSE:
            self.wps = self.wps[::-1].copy()
        self.yaw_ref = compute_yaw_from_path(self.wps[:, :2], loop=WP_LOOP)
        self.last_idx = None
        self.get_logger().info(f'Loaded {len(self.wps)} waypoints from {WAYPOINTS_CSV}')

        # TF listener — provides map→base_link (AMCL must be running)
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # Lap state
        self._lap    = 0
        self._n_idx  = 0    # nearest waypoint index (current)
        self._p_idx  = 0    # nearest waypoint index (previous)
        self._lap_t0 = None

        # State (px, py, delta, v, theta) — delta is integrated from commanded steer_vel
        self.state = np.zeros(5)
        self.delta_cmd  = 0.0
        self._speed_cmd = 0.0   # integrated speed setpoint — accumulates like delta_cmd
        self._v    = 0.0    # velocity from EKF odometry
        self.u_prev = np.zeros(2)
        self.solver_failures = 0

        # Steer-only debug state
        self._rc_speed = 0.0    # latest speed seen on /teleop (from RC stick)
        self._last_xy  = None   # previous cycle's (x, y) — used for jump detection



        self._status_t0 = self.get_clock().now().nanoseconds * 1e-9
        self._status_period = 3.0
        self._tf_ready = False

 

        # MPC
        self.mpc = KinematicMPC(
            N=N, dt=DT, wheelbase=WHEELBASE,
            max_steer=MAX_STEER, max_steer_vel=MAX_STEER_VEL, max_accel=MAX_ACCEL,
            min_speed=MIN_SPEED, max_speed=MAX_SPEED,
            Q=[Q_X, Q_Y, Q_YAW, Q_V],
            R=[R_STEER_VEL, R_ACCEL],
            Rd=[RD_STEER_VEL, RD_ACCEL],
            qf_scale=QF_SCALE,
            ipopt_print_level=IPOPT_PRINT_LEVEL,
            ipopt_max_iter=IPOPT_MAX_ITER,
        )

        # ROS interfaces
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self._lap_pub  = self.create_publisher(Int32, '/mpc/lap', 5)
        self.create_timer(1.0 / CTRL_RATE_HZ, self.control_loop)

        if STEER_ONLY_MODE:
            self.create_subscription(AckermannDriveStamped, TELEOP_TOPIC, self._teleop_cb, 10)
            self.get_logger().warn(
                '[DEBUG] STEER_ONLY_MODE=True — steering from MPC, speed from RC /teleop. '
                'Keep AUTO switch ON on remote so mux routes /drive to VESC.'
            )

        if PUBLISH_MARKERS:
            self._raceline_pub   = self.create_publisher(Marker, '/mpc/raceline',    1)
            self._horizon_pub    = self.create_publisher(Marker, '/mpc/horizon',     1)
            self._ref_pub        = self.create_publisher(Marker, '/mpc/ref_horizon', 1)
            self._raceline_sent  = False
            self.get_logger().info('PUBLISH_MARKERS=True → /mpc/raceline  /mpc/horizon  /mpc/ref_horizon')

    def odom_cb(self, msg: Odometry):
        """Update velocity from EKF odometry. Pose comes from TF (map→base_link)."""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._v = float(np.hypot(vx, vy))

    def _get_map_pose(self):
        """Return (x, y, yaw) in map frame via TF lookup, or None if unavailable."""
        try:
            tf = self._tf_buf.lookup_transform(MAP_FRAME, BASE_FRAME, RclTime())
            t, r = tf.transform.translation, tf.transform.rotation
            return t.x, t.y, _quat_to_yaw(r.x, r.y, r.z, r.w)
        except Exception:
            return None

    def _teleop_cb(self, msg: AckermannDriveStamped):
        """Track RC speed so STEER_ONLY_MODE can mirror it in the /drive message."""
        # abs() because SPEED_SIGN handles polarity — we just need the magnitude
        self._rc_speed = abs(msg.drive.speed)

    def _check_lap(self):
        """Detect lap completion when waypoint index wraps from ≥85% to ≤15%."""
        n = len(self.wps)
        if self._p_idx > int(0.85 * n) and self._n_idx < int(0.15 * n):
            self._lap += 1
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self._lap_t0 is not None:
                self.get_logger().info(
                    f'>>> LAP {self._lap} complete — {now_s - self._lap_t0:.2f} s <<<')
            else:
                self.get_logger().info(f'>>> LAP {self._lap} <<<')
            self._lap_t0 = now_s
            msg = Int32()
            msg.data = self._lap
            self._lap_pub.publish(msg)

    def control_loop(self):
        # 1. Get pose in map frame from AMCL via TF
        pose = self._get_map_pose()
        if pose is None:
            self.get_logger().warn('Waiting for map→base_link TF (is AMCL running?)',
                                   throttle_duration_sec=5.0)
            return
        if not self._tf_ready:
            self.get_logger().info('[TF] map→base_link available — MPC active')
            self._tf_ready = True
        x, y, yaw = pose

        # Position-jump detector: wheel slip or AMCL glitch causes sudden teleport
        if self._last_xy is not None:
            jump = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if jump > MAX_POS_JUMP_M:
                self.get_logger().warn(
                    f'[POSE JUMP] {jump:.3f} m in one cycle! '
                    f'prev=({self._last_xy[0]:.2f},{self._last_xy[1]:.2f}) '
                    f'now=({x:.2f},{y:.2f}) — wheel slip / EKF divergence?'
                )
        self._last_xy = (x, y)

        self.state[0] = x
        self.state[1] = y
        self.state[2] = self.delta_cmd
        # self.state[3] = self._v
        self.state[3] = max(self._v, MIN_SPEED)
        self.state[4] = yaw

        # 2. Nearest waypoint (warm-started) + lap detection
        self._p_idx = self.last_idx if self.last_idx is not None else 0
        idx = nearest_index(
            self.wps[:, :2], self.state[:2],
            last_idx=self.last_idx, search_window=30, loop=WP_LOOP,
        )
        self.last_idx = idx
        self._n_idx  = idx
        self._check_lap()

        # 3. Horizon reference — pace the arc-length look-ahead at actual/commanded speed
        if STEER_ONLY_MODE:
            # Use measured velocity; fall back to 0.3 m/s so horizon doesn't collapse at standstill
            v_pace = max(self._v, 0.3)
        else:
            v_pace = max(self._speed_cmd if self._speed_cmd > 0 else MIN_SPEED, float(self.state[3]))
        x_ref = extract_horizon(
            self.wps, self.yaw_ref, idx, N, DT,
            v_target=v_pace, loop=WP_LOOP,
        )
        # Apply TARGET_SPEED cap so raising/lowering one number controls pace
        x_ref[3, :] = np.minimum(x_ref[3, :], TARGET_SPEED)

        # Align reference yaw to current yaw to avoid 2pi jumps
        x_ref[2, :] += np.round((self.state[4] - x_ref[2, 0]) / (2.0 * np.pi)) * 2.0 * np.pi

        # 4. Solve MPC
        u0, _X, _U, ok = self.mpc.solve(self.state, x_ref, self.u_prev)
        if not ok:
            self.solver_failures += 1
            if self.solver_failures >= 2:
                self.get_logger().warn(f'MPC solver failed {self.solver_failures}x; using fallback')
            # Fallback: zero increments → delta_cmd and _speed_cmd hold their last values
            u0 = np.array([0.0, 0.0])
        else:
            self.solver_failures = 0

        self.u_prev = u0.copy()

        # Visualization markers (only when PUBLISH_MARKERS=True)
        if PUBLISH_MARKERS:
            if not self._raceline_sent:
                self._pub_raceline()
                self._raceline_sent = True
            self._pub_markers(_X, x_ref)

        # 5. Convert (steer_vel, accel) into AckermannDrive (steering_angle, speed)
        self.delta_cmd  = float(np.clip(self.delta_cmd  + u0[0] * DT, -MAX_STEER, MAX_STEER))
        # Integrate speed command across cycles (same pattern as delta_cmd).
        # Using self._speed_cmd as the base — not measured _v — so the commanded speed
        # ramps up from 0 regardless of whether the VESC has moved the car yet.
        self._speed_cmd = float(np.clip(self._speed_cmd + u0[1] * DT, MIN_SPEED, MAX_SPEED))

        # 6. Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        if STEER_ONLY_MODE:
            # Speed mirrors what the RC stick is commanding; MPC only steers.
            msg.drive.speed = SPEED_SIGN * self._rc_speed
        else:
            msg.drive.speed = SPEED_SIGN * self._speed_cmd
        msg.drive.steering_angle = STEER_SIGN * self.delta_cmd

        self.drive_pub.publish(msg)

        # 7. Periodic status — tells you exactly what the MPC is doing
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._status_t0 >= self._status_period:
            dist_to_wp = float(np.linalg.norm(self.wps[idx, :2] - self.state[:2]))
            if STEER_ONLY_MODE:
                speed_str = f'rc_speed={self._rc_speed:.2f}m/s(sent={SPEED_SIGN*self._rc_speed:.2f})'
            else:
                speed_str = f'cmd_v={self._speed_cmd:.2f}m/s(sent={SPEED_SIGN*self._speed_cmd:.2f})'
            self.get_logger().info(
                f'[MPC{"*STEER-ONLY*" if STEER_ONLY_MODE else ""}] '
                f'lap={self._lap} wp={idx}/{len(self.wps)} dist={dist_to_wp:.2f}m | '
                f'{speed_str} steer={math.degrees(self.delta_cmd):.1f}° | '
                f'measured_v={self._v:.2f}m/s '
                f'pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}°) | '
                f'solver_fails={self.solver_failures}')
            self._status_t0 = now_s


    # ------------------------------------------------------------------
    # Marker helpers (only called when PUBLISH_MARKERS=True)
    # ------------------------------------------------------------------

    def _pub_raceline(self):
        """Publish the full raceline as a green LINE_STRIP once."""
        m = Marker()
        m.header.frame_id = MAP_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'mpc', 0
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 0.8
        for wp in self.wps:
            p = Point()
            p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
            m.points.append(p)
        # Close the loop
        p = Point()
        p.x, p.y, p.z = float(self.wps[0, 0]), float(self.wps[0, 1]), 0.0
        m.points.append(p)
        self._raceline_pub.publish(m)

    def _pub_markers(self, X_pred, x_ref):
        """Publish MPC predicted trajectory (blue) and reference horizon (yellow)."""
        now = self.get_clock().now().to_msg()

        def _line(mid, r, g, b, pts_xy, z=0.05):
            m = Marker()
            m.header.frame_id = MAP_FRAME
            m.header.stamp = now
            m.ns, m.id = 'mpc', mid
            m.type, m.action = Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9
            for (x, y) in pts_xy:
                p = Point()
                p.x, p.y, p.z = float(x), float(y), z
                m.points.append(p)
            return m

        # Blue — MPC predicted trajectory
        self._horizon_pub.publish(_line(
            1, 0.2, 0.4, 1.0,
            [(X_pred[0, k], X_pred[1, k]) for k in range(X_pred.shape[1])],
        ))
        # Yellow — reference horizon sent to the solver
        self._ref_pub.publish(_line(
            2, 1.0, 0.85, 0.0,
            [(x_ref[0, k], x_ref[1, k]) for k in range(x_ref.shape[1])],
        ))


def main():
    rclpy.init()
    node = KinematicMPCNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
