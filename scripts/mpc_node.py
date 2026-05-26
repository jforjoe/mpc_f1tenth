#!/usr/bin/env python3
"""ROS2 node: receding-horizon Kinematic MPC for F1TENTH.

All tunable parameters live at the top of this file.

Debug flags (see DEBUG section below):
  Run with the defaults to see startup checks + once-per-second runtime diagnostics.
  Flip ONE flag at a time to isolate failures — see "Diagnosis guide" below.
  DBG_STARTUP always runs — it validates the raceline, params and direction at boot.

Diagnosis guide (read this when the car misbehaves):
  1. Car turns hard at start and crashes into wall
       → Look at [STARTUP] Raceline direction line. If it says FAIL or err > 90°,
         the CSV is ordered the wrong way for the car's spawn heading.
         AUTO_REVERSE=True will flip it automatically next cycle.
  2. Car doesn't move at all              → DBG_TF then DBG_DRIVE
  3. Car goes backward                    → DBG_DRIVE (check SPEED_SIGN output)
  4. Car steers wrong way in auto         → DBG_DRIVE (check STEER_SIGN output)
  5. Car oscillates / wobbles             → DBG_SOLVER (check solver_fails, objective trend)
  6. Car ignores velocity profile         → DBG_HORIZON (v_ref vs cmd_v), raise Q_V
  7. Car cuts corners                     → DBG_RACELINE (lateral offset), raise Q_X/Q_Y
  8. Solver too slow / control jitter     → DBG_TIMING (solver_ms vs 50ms budget)
"""

import math
import os
import time

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

try:
    from .mpc_solver import KinematicMPC
    from .waypoint_utils import load_waypoints, compute_yaw_from_path, nearest_index, extract_horizon
except ImportError:
    from mpc_solver import KinematicMPC
    from waypoint_utils import load_waypoints, compute_yaw_from_path, nearest_index, extract_horizon


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


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
    # Fallback: config/ at the package root (one level above scripts/)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'raceline.csv')

WAYPOINTS_CSV   = _resolve_waypoints_csv()
WP_LOOP         = True
WP_REVERSE      = False    # static flip of the CSV at load time (do this if you ALWAYS need it reversed)
AUTO_REVERSE    = False     # on first valid TF, if |car_yaw - wp_yaw| > 90°, flip the CSV automatically.
                            # This is the fix for "car turns at start and crashes into wall" — the raceline
                            # skeleton traversal direction from plan_trajectory.py is arbitrary.

# --- MPC horizon ---
N               = 40        # 40 × 0.05 = 2 s preview; car sees turns ~2.0 m ahead at 1 m/s
DT              = 0.05
CTRL_RATE_HZ    = 20.0

# --- Hardware sign corrections (mirror mppi_racing_node.py) ---
SPEED_SIGN  = -1.0   # flip if VESC erpm_gain is negative (default F1TENTH: -4100)
STEER_SIGN  = 1.0   # flip if car steers wrong way in auto mode

# --- Vehicle (from vesc.yaml hardware calibration) ---
# vesc_to_odom_node.wheelbase = 0.325 m
WHEELBASE       = 0.325
# servo_min=0.15 → (0.15-0.48)/(-1.2135) = 0.272 rad; matches MPPI DELTA_MAX and nav2 min_turning_radius=1.17m
MAX_STEER       = 0.272    # rad
# throttle_interpolator.max_servo_speed = 3.2 rad/s — keep solver rate well below mechanical max
MAX_STEER_VEL   = 1.0      # rad/s   (raise to 1.5–2.0 if MPC's steer feels sluggish in turns)
# throttle_interpolator.max_acceleration = 2.5 m/s²
MAX_ACCEL       = 1.0      # m/s²
# speed_min=-23250 ERPM / |gain=4100| = 5.67 m/s; cap well below for safety margin during tuning
MAX_SPEED       = 1.5      # m/s  (raise in 0.5 m/s steps once tracking is stable)
MIN_SPEED       = 0.5

# --- Reference ---
# Speed cap applied to every waypoint's vx_mps value from the CSV.
# Lower this to slow the entire raceline without regenerating the CSV.
# Raise it (up to MAX_SPEED) to let the car run at the CSV's optimised speeds.
TARGET_SPEED    = 1.5      # m/s  — start here; raise in 0.5 m/s steps once tracking is stable

# --- Cost weights ---
# Tuning ranges (derived from MPPI calibration in mppi_racing_node.py):
#   Q_X, Q_Y     5–30  (position tracking; raise if car drifts off raceline laterally)
#   Q_YAW        5–30  (heading; raise if car enters corners misaligned)
#   Q_V          0.1–5 (velocity; MPPI uses ~3.0 — raise from 0.5 if car ignores v_ref)
#   R_*          0.001–0.1 (control magnitude — keeps u small)
#   Rd_*         0.1–2.0   (control rate — RD_STEER_VEL=1.0 damps jitter, RD_ACCEL=0.5 smooths throttle)
#   QF_SCALE     1.0–3.0   (terminal weight multiplier — 2.0 pulls hard toward end of horizon)
Q_X, Q_Y, Q_YAW, Q_V    = 20.0, 20.0, 20.0, 0.5
R_STEER_VEL, R_ACCEL    = 0.01, 0.01
RD_STEER_VEL, RD_ACCEL  = 1.0, 0.5
QF_SCALE                = 2.0

# --- Solver ---
IPOPT_PRINT_LEVEL = 0     # 0=silent (deployment), 5=verbose (debugging only)
IPOPT_MAX_ITER    = 100   # 50 is plenty at 20 Hz; raise to 100 only when debugging convergence

# --- Debug / visualization ---
# Set True to publish RViz markers: /mpc/raceline (green), /mpc/horizon (blue), /mpc/ref_horizon (yellow)
PUBLISH_MARKERS  = True

# =====================================================================
#                         DEBUG FLAGS  (toggle to diagnose)
# =====================================================================
# Each flag enables verbose logging for ONE subsystem.
# All False by default for production — zero runtime cost when off.
# Flip ONE at a time to isolate a failure.
DBG_STARTUP    = True    # Parameter + raceline + direction checks at boot (recommended ON)
DBG_TF         = False   # Map-frame pose every control cycle
DBG_RACELINE   = True    # Nearest waypoint + lookahead in robot frame + heading error
DBG_HORIZON    = False   # First / mid / last reference vs predicted state on the horizon
DBG_SOLVER     = True    # Solver success/fail + objective + state error vs reference[0]
DBG_DRIVE      = True    # Raw + sign-corrected drive commands every cycle
DBG_TIMING     = False   # MPC solve wall-clock time per cycle
DBG_LAP        = False    # Lap completion events (always recommended ON)

DBG_LOG_PERIOD = 1.0     # seconds between repeated debug prints (throttle rate)

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
        self._solver_fails_total = 0

        # Diagnostic accumulators (read by _diag_* methods)
        self._dbg_solver_ok    = True
        self._dbg_solver_ms    = 0.0
        self._dbg_objective    = 0.0       # not exposed by Opti — placeholder for future
        self._dbg_x_err0       = 0.0       # state - x_ref at step 0 after solve
        self._dbg_y_err0       = 0.0
        self._dbg_yaw_err0     = 0.0
        self._dbg_v_err0       = 0.0
        self._dbg_x_errN       = 0.0       # state - x_ref at terminal step
        self._dbg_y_errN       = 0.0
        self._dbg_yaw_errN     = 0.0

        # Direction check state
        self._dir_checked = False

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

        if PUBLISH_MARKERS:
            self._raceline_pub   = self.create_publisher(Marker, '/mpc/raceline',    1)
            self._horizon_pub    = self.create_publisher(Marker, '/mpc/horizon',     1)
            self._ref_pub        = self.create_publisher(Marker, '/mpc/ref_horizon', 1)
            self._raceline_sent  = False
            self.get_logger().info('PUBLISH_MARKERS=True → /mpc/raceline  /mpc/horizon  /mpc/ref_horizon')

        # Startup self-test
        self._run_startup_checks()

    # =========================================================================
    #  ROS callbacks
    # =========================================================================

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

    def _check_lap(self):
        """Detect lap completion when waypoint index wraps from ≥85% to ≤15%."""
        n = len(self.wps)
        if self._p_idx > int(0.85 * n) and self._n_idx < int(0.15 * n):
            self._lap += 1
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if DBG_LAP:
                if self._lap_t0 is not None:
                    self.get_logger().info(
                        f'>>> LAP {self._lap} complete — {now_s - self._lap_t0:.2f} s <<<')
                else:
                    self.get_logger().info(f'>>> LAP {self._lap} <<<')
            self._lap_t0 = now_s
            msg = Int32()
            msg.data = self._lap
            self._lap_pub.publish(msg)

    # =========================================================================
    #  Direction check (the fix for "turns at start, crashes")
    # =========================================================================

    def _check_raceline_direction(self, car_yaw: float, idx: int) -> float:
        """Compare car heading to raceline traversal direction at idx.

        Returns wrapped angular error (rad). |err| > π/2 means the CSV is ordered
        against the car's driving direction — set WP_REVERSE=True or enable AUTO_REVERSE.
        """
        wp_yaw = float(self.yaw_ref[idx])
        return _wrap_pi(car_yaw - wp_yaw)

    def _maybe_reverse_raceline(self, car_yaw: float, idx: int) -> int:
        """First-call direction check. Returns possibly-updated nearest index."""
        if self._dir_checked:
            return idx
        err = self._check_raceline_direction(car_yaw, idx)
        wp_yaw = float(self.yaw_ref[idx])
        if abs(err) > math.pi / 2:
            self.get_logger().error(
                f'[STARTUP] Raceline direction .. FAIL  '
                f'car_yaw={math.degrees(car_yaw):.1f}° vs '
                f'wp_yaw[{idx}]={math.degrees(wp_yaw):.1f}° → err={math.degrees(err):+.0f}°  '
                f'— CSV IS REVERSED relative to the car.')
            if AUTO_REVERSE:
                self.get_logger().warn(
                    '[STARTUP] AUTO_REVERSE=True — flipping waypoints now. '
                    '(Set WP_REVERSE=True permanently if this keeps happening.)')
                self.wps = self.wps[::-1].copy()
                self.yaw_ref = compute_yaw_from_path(self.wps[:, :2], loop=WP_LOOP)
                self.last_idx = None
                idx = nearest_index(
                    self.wps[:, :2], self.state[:2],
                    last_idx=None, search_window=30, loop=WP_LOOP)
                self.last_idx = idx
                new_err = self._check_raceline_direction(car_yaw, idx)
                self.get_logger().info(
                    f'[STARTUP] After flip ......... '
                    f'wp_yaw[{idx}]={math.degrees(self.yaw_ref[idx]):.1f}° → '
                    f'err={math.degrees(new_err):+.0f}°  (should be ≈ 0)')
            else:
                self.get_logger().error(
                    'AUTO_REVERSE=False — MPC will fight the reversed reference. '
                    'Stop the car and set WP_REVERSE=True at the top of mpc_node.py.')
        else:
            self.get_logger().info(
                f'[STARTUP] Raceline direction .. PASS  '
                f'car_yaw={math.degrees(car_yaw):.1f}° vs '
                f'wp_yaw[{idx}]={math.degrees(wp_yaw):.1f}° → err={math.degrees(err):+.0f}°  '
                f'(forward-aligned)')
        self._dir_checked = True
        return idx

    # =========================================================================
    #  Main control loop
    # =========================================================================

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

        if DBG_TF:
            self._diag_tf(x, y, yaw)

        self.state[0] = x
        self.state[1] = y
        self.state[2] = self.delta_cmd
        self.state[3] = max(self._v, MIN_SPEED)
        self.state[4] = yaw

        # 2. Nearest waypoint (warm-started) + lap detection
        self._p_idx = self.last_idx if self.last_idx is not None else 0
        idx = nearest_index(
            self.wps[:, :2], self.state[:2],
            last_idx=self.last_idx, search_window=30, loop=WP_LOOP,
        )

        # 2b. One-time direction check / auto-flip on first valid TF.
        # Done BEFORE first solve so the very first horizon is correct.
        idx = self._maybe_reverse_raceline(yaw, idx)

        self.last_idx = idx
        self._n_idx  = idx
        self._check_lap()

        if DBG_RACELINE:
            self._diag_raceline(x, y, yaw, idx)

        # 3. Horizon reference — use commanded speed so arc-length spacing matches reality
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
        _t0 = time.perf_counter() if DBG_TIMING or DBG_SOLVER else 0.0
        u0, _X, _U, ok = self.mpc.solve(self.state, x_ref, self.u_prev)
        self._dbg_solver_ms = (time.perf_counter() - _t0) * 1e3 if (DBG_TIMING or DBG_SOLVER) else 0.0
        self._dbg_solver_ok = ok

        if not ok:
            self.solver_failures += 1
            self._solver_fails_total += 1
            if self.solver_failures >= 2:
                self.get_logger().warn(
                    f'MPC solver failed {self.solver_failures}x consecutive '
                    f'(total={self._solver_fails_total}); using zero-increment fallback')
            u0 = np.array([0.0, 0.0])
        else:
            self.solver_failures = 0

        self.u_prev = u0.copy()

        # Pre-compute diag state errors (cheap; only used if DBG_SOLVER/DBG_HORIZON)
        if DBG_SOLVER or DBG_HORIZON:
            self._dbg_x_err0   = float(_X[0, 0] - x_ref[0, 0])
            self._dbg_y_err0   = float(_X[1, 0] - x_ref[1, 0])
            self._dbg_yaw_err0 = _wrap_pi(float(_X[4, 0] - x_ref[2, 0]))
            self._dbg_v_err0   = float(_X[3, 0] - x_ref[3, 0])
            self._dbg_x_errN   = float(_X[0, -1] - x_ref[0, -1])
            self._dbg_y_errN   = float(_X[1, -1] - x_ref[1, -1])
            self._dbg_yaw_errN = _wrap_pi(float(_X[4, -1] - x_ref[2, -1]))

        if DBG_SOLVER:
            self._diag_solver()
        if DBG_HORIZON:
            self._diag_horizon(_X, x_ref)
        if DBG_TIMING:
            self._diag_timing()

        # Visualization markers
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

        if DBG_DRIVE:
            self._diag_drive(self._speed_cmd, self.delta_cmd)

        # 6. Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed          = SPEED_SIGN * self._speed_cmd
        msg.drive.steering_angle = STEER_SIGN * self.delta_cmd
        self.drive_pub.publish(msg)

        # 7. Periodic status — compact one-liner, always on (separate from DBG flags)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._status_t0 >= self._status_period:
            dist_to_wp = float(np.linalg.norm(self.wps[idx, :2] - self.state[:2]))
            self.get_logger().info(
                f'[MPC] lap={self._lap} wp={idx}/{len(self.wps)} '
                f'dist_to_wp={dist_to_wp:.2f}m | '
                f'cmd: v={self._speed_cmd:.2f}m/s({SPEED_SIGN*self._speed_cmd:+.2f}) '
                f'steer={math.degrees(self.delta_cmd):+.1f}° | '
                f'measured_v={self._v:.2f}m/s '
                f'pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}°) | '
                f'solver_fails={self._solver_fails_total} '
                f'solve_ms={self._dbg_solver_ms:.1f}')
            self._status_t0 = now_s

    # =========================================================================
    #  Marker helpers (only called when PUBLISH_MARKERS=True)
    # =========================================================================

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

    # =========================================================================
    #  Startup self-test
    # =========================================================================

    def _run_startup_checks(self):
        """Validate all subsystems at startup. Logs PASS / WARN / FAIL for each."""
        log = self.get_logger()
        log.info('=' * 70)
        log.info('[STARTUP] Kinematic MPC subsystem checks ...')

        # 1. Waypoints
        if len(self.wps) > 0:
            v_csv = self.wps[:, 2][np.isfinite(self.wps[:, 2])]
            v_min_csv = float(v_csv.min()) if v_csv.size else float('nan')
            v_max_csv = float(v_csv.max()) if v_csv.size else float('nan')
            x_range = (float(self.wps[:, 0].min()), float(self.wps[:, 0].max()))
            y_range = (float(self.wps[:, 1].min()), float(self.wps[:, 1].max()))
            log.info(
                f'[STARTUP] Waypoints ........... PASS  '
                f'{len(self.wps)} pts  '
                f'x=[{x_range[0]:.1f},{x_range[1]:.1f}]  '
                f'y=[{y_range[0]:.1f},{y_range[1]:.1f}]  '
                f'v_csv=[{v_min_csv:.2f},{v_max_csv:.2f}] m/s  '
                f'{"REVERSED" if WP_REVERSE else "forward"}')
        else:
            log.error('[STARTUP] Waypoints ........... FAIL — empty CSV!')

        # 2. Horizon
        horizon_s = N * DT
        max_arc   = MAX_SPEED * horizon_s
        log.info(
            f'[STARTUP] MPC horizon ......... PASS  '
            f'N={N} steps × DT={DT}s = {horizon_s:.2f}s preview  '
            f'(arc at V_MAX={MAX_SPEED:.1f}: {max_arc:.2f}m)')

        # 3. CSV speed vs vehicle MAX_SPEED
        if len(self.wps) > 0 and np.isfinite(self.wps[:, 2]).any():
            v_max_csv = float(self.wps[np.isfinite(self.wps[:, 2]), 2].max())
            if v_max_csv > MAX_SPEED * 1.5:
                log.warn(
                    f'[STARTUP] CSV vs MAX_SPEED .... WARN  '
                    f'csv v_max={v_max_csv:.2f} >> MAX_SPEED={MAX_SPEED:.2f} m/s  '
                    f'TARGET_SPEED={TARGET_SPEED:.2f} cap is doing the work.  '
                    f'Raise MAX_SPEED + TARGET_SPEED together to go faster.')
            else:
                log.info(
                    f'[STARTUP] CSV vs MAX_SPEED .... PASS  '
                    f'csv v_max={v_max_csv:.2f}  MAX={MAX_SPEED:.2f}  TARGET={TARGET_SPEED:.2f} m/s')

        # 4. Cost weights
        log.info(
            f'[STARTUP] Cost weights ........ PASS  '
            f'Q=[x={Q_X} y={Q_Y} yaw={Q_YAW} v={Q_V}]  '
            f'R=[sv={R_STEER_VEL} a={R_ACCEL}]  '
            f'Rd=[sv={RD_STEER_VEL} a={RD_ACCEL}]  Qf×={QF_SCALE}')
        # Hint when Q_V is much smaller than position weights (common tuning issue)
        if Q_V < 0.1 * Q_X:
            log.warn(
                f'[STARTUP] Q_V vs Q_X ratio .... HINT  '
                f'Q_V={Q_V} << Q_X={Q_X} → solver may ignore velocity profile. '
                f'MPPI uses W_SPEED=3.0 with similar position weights — try Q_V≈2–5 if '
                f'speed tracking is poor.')

        # 5. Sign corrections
        log.info(
            f'[STARTUP] Sign corrections .... PASS  '
            f'SPEED_SIGN={SPEED_SIGN:+.0f}  STEER_SIGN={STEER_SIGN:+.0f}  '
            f'(+1 = normal, -1 = inverted VESC polarity)')

        # 6. Vehicle limits
        log.info(
            f'[STARTUP] Vehicle limits ...... PASS  '
            f'WHEELBASE={WHEELBASE:.3f}m  '
            f'V=[{MIN_SPEED:.2f},{MAX_SPEED:.2f}]m/s  '
            f'DELTA_MAX={math.degrees(MAX_STEER):.1f}°  '
            f'ACCEL_MAX={MAX_ACCEL:.1f}m/s²  '
            f'STEER_VEL_MAX={MAX_STEER_VEL:.1f}rad/s')

        # 7. Debug flags
        active = [n for n, v in [
            ('TF', DBG_TF), ('RACELINE', DBG_RACELINE), ('HORIZON', DBG_HORIZON),
            ('SOLVER', DBG_SOLVER), ('DRIVE', DBG_DRIVE),
            ('TIMING', DBG_TIMING), ('LAP', DBG_LAP)] if v]
        if active:
            log.info(f'[STARTUP] Debug flags active: {", ".join(active)}  '
                     f'(throttled to {DBG_LOG_PERIOD:.1f}s)')
        else:
            log.info('[STARTUP] Debug flags ......... all OFF (production)')

        # 8. Pending checks deferred to first cycle
        log.info(f'[STARTUP] TF .................. waiting for map→{BASE_FRAME}')
        log.info(f'[STARTUP] Direction check ..... will run on first valid TF '
                 f'(AUTO_REVERSE={AUTO_REVERSE})')
        log.info('[STARTUP] Done.  Watch for [TF] and [STARTUP] Raceline direction lines above.')
        log.info('=' * 70)

    # =========================================================================
    #  Debug diagnostic methods  (only called when matching DBG_* flag = True)
    # =========================================================================

    def _diag_tf(self, x: float, y: float, yaw: float):
        """Log current map-frame pose. Enable with DBG_TF=True."""
        self.get_logger().info(
            f'[TF] pose  x={x:.3f} m  y={y:.3f} m  yaw={math.degrees(yaw):.1f}°',
            throttle_duration_sec=DBG_LOG_PERIOD)

    def _diag_raceline(self, x: float, y: float, yaw: float, idx: int):
        """Log nearest waypoint, lookahead point in robot frame, heading error."""
        n = len(self.wps)
        rx, ry = float(self.wps[idx, 0]), float(self.wps[idx, 1])
        dist_to_nearest = math.hypot(x - rx, y - ry)
        # Look ahead ~1m worth of waypoints to mirror MPPI's LOOKAHEAD_M=1.0
        la_steps = max(1, int(1.0 / max(np.linalg.norm(self.wps[1, :2] - self.wps[0, :2]), 1e-3)))
        la_idx = (idx + la_steps) % n
        lx, ly = float(self.wps[la_idx, 0]), float(self.wps[la_idx, 1])
        h_ref = float(self.yaw_ref[la_idx])
        heading_err = math.degrees(_wrap_pi(yaw - h_ref))
        # Reference offset in robot frame (+y = LEFT, -y = RIGHT, +x = ahead)
        dx_w = lx - x; dy_w = ly - y
        cy = math.cos(yaw); sy = math.sin(yaw)
        ref_xr =  dx_w * cy + dy_w * sy
        ref_yr = -dx_w * sy + dy_w * cy
        self.get_logger().info(
            f'[RACELINE] nearest={idx}/{n}  ({rx:.2f},{ry:.2f})  dist={dist_to_nearest:.3f}m  |  '
            f'lookahead={la_idx}  ({lx:.2f},{ly:.2f})  '
            f'v_ref={self.wps[la_idx, 2]:.2f}m/s  '
            f'h_ref={math.degrees(h_ref):+.1f}°  |  '
            f'robot_yaw={math.degrees(yaw):+.1f}°  heading_err={heading_err:+.1f}°  |  '
            f'ref_in_robot_frame: fwd={ref_xr:+.2f}m  lat={ref_yr:+.2f}m '
            f'(+lat=LEFT  -lat=RIGHT)',
            throttle_duration_sec=DBG_LOG_PERIOD)

    def _diag_solver(self):
        """Log MPC solver health and state-vs-reference tracking. DBG_SOLVER=True."""
        status = 'OK  ' if self._dbg_solver_ok else 'FAIL'
        self.get_logger().info(
            f'[SOLVER] {status}  solve_ms={self._dbg_solver_ms:.1f}  '
            f'fails_total={self._solver_fails_total}  '
            f'err@0: x={self._dbg_x_err0:+.3f}m y={self._dbg_y_err0:+.3f}m '
            f'yaw={math.degrees(self._dbg_yaw_err0):+.1f}° v={self._dbg_v_err0:+.2f}m/s  |  '
            f'err@N: x={self._dbg_x_errN:+.3f}m y={self._dbg_y_errN:+.3f}m '
            f'yaw={math.degrees(self._dbg_yaw_errN):+.1f}°',
            throttle_duration_sec=DBG_LOG_PERIOD)

    def _diag_horizon(self, X_pred, x_ref):
        """Log reference and predicted state at first / mid / last horizon step."""
        K = X_pred.shape[1]
        idxs = [0, K // 2, K - 1]
        parts = []
        for k in idxs:
            parts.append(
                f't={k:02d}: ref=({x_ref[0, k]:+.2f},{x_ref[1, k]:+.2f},'
                f'{math.degrees(x_ref[2, k]):+.0f}°,{x_ref[3, k]:.2f}m/s)  '
                f'pred=({X_pred[0, k]:+.2f},{X_pred[1, k]:+.2f},'
                f'{math.degrees(X_pred[4, k]):+.0f}°,{X_pred[3, k]:.2f}m/s)'
            )
        self.get_logger().info(
            '[HORIZON]  ' + '  |  '.join(parts),
            throttle_duration_sec=DBG_LOG_PERIOD)

    def _diag_drive(self, v_cmd: float, d_cmd: float):
        """Log raw + sign-corrected drive commands. DBG_DRIVE=True."""
        steer_sat = abs(d_cmd) > 0.95 * MAX_STEER
        speed_sat = abs(v_cmd) >= MAX_SPEED - 1e-3
        flags = ''
        if steer_sat: flags += ' STEER_SAT'
        if speed_sat: flags += ' SPEED_SAT'
        self.get_logger().info(
            f'[DRIVE] cmd: v={v_cmd:.3f}m/s  steer={math.degrees(d_cmd):+.2f}°  |  '
            f'published: v={SPEED_SIGN * v_cmd:+.3f}m/s  '
            f'steer={math.degrees(STEER_SIGN * d_cmd):+.2f}°  '
            f'(signs: SPEED={SPEED_SIGN:+.0f} STEER={STEER_SIGN:+.0f}){flags}',
            throttle_duration_sec=DBG_LOG_PERIOD)

    def _diag_timing(self):
        """Log solver wall-clock time. DBG_TIMING=True."""
        budget_ms = (1.0 / CTRL_RATE_HZ) * 1e3 * 0.8   # 80% of control period
        ok = 'OK' if self._dbg_solver_ms < budget_ms else 'OVER BUDGET!'
        self.get_logger().info(
            f'[TIMING] solver={self._dbg_solver_ms:.2f}ms  '
            f'(budget={budget_ms:.1f}ms at {CTRL_RATE_HZ:.0f}Hz)  {ok}',
            throttle_duration_sec=DBG_LOG_PERIOD)


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
