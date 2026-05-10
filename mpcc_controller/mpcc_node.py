"""
ROS2 node for MPCC controller — DriftingCar / DynamicBicycle2D backend.

State convention (matches safe_control repo exactly):
    [x, y, theta, r, beta, V, delta, tau]
     0  1    2    3    4   5    6     7

    x, y    — global position          [m]       — from odom pose
    theta   — heading                  [rad]     — from odom quaternion
    r       — yaw rate                 [rad/s]   — from odom twist.angular.z
    beta    — side-slip angle          [rad]     — not in odom; held at 0
    V       — speed magnitude          [m/s]     — from odom twist.linear.x
    delta   — current steering angle   [rad]     — integrated from delta_dot output
    tau     — current drive torque     [Nm]      — integrated from tau_dot output

Control output from MPCC.solve_control_problem():
    U = [delta_dot, tau_dot]           (2x1 numpy array)

All error computation (e_c, e_l, e_theta, e_v) happens INSIDE the MPCC
optimizer as CasADi symbolic expressions. The node never computes errors.

Subscribes to:
    /ego_racecar/odom  (nav_msgs/Odometry)

Publishes to:
    /drive                  (ackermann_msgs/AckermannDriveStamped)
    /mpcc/predicted_path    (nav_msgs/Path)
    /mpcc/reference_path    (nav_msgs/Path)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import csv
import pathlib
import traceback

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

from drifting_car import DriftingCar
from mpcc import MPCC


class MPCCNode(Node):
    """ROS2 node for MPCC using DriftingCar + DynamicBicycle2D dynamics."""

    def __init__(self):
        super().__init__('mpcc_controller')

        # ── raw odom snapshots ─────────────────────────────────────────────
        # Populated by odom_callback; consumed by control_loop each tick.
        self._x:     float | None = None   # global x       [m]
        self._y:     float | None = None   # global y       [m]
        self._theta: float | None = None   # heading        [rad]
        self._r:     float = 0.0           # yaw rate       [rad/s]  — odom angular.z
        self._V:     float = 0.0           # speed          [m/s]    — odom linear.x

        # delta and tau are not observable from /odom.
        # Integrated forward each tick from MPCC output [delta_dot, tau_dot].
        self._delta_est: float = 0.0       # steering angle estimate  [rad]
        self._tau_est:   float = 0.0       # torque estimate          [Nm]

        # beta (side-slip) is not in /odom.
        # Held at 0 — valid at moderate speeds on high-friction surface (mu=1.0).
        self._beta: float = 0.0

        self.initialized = False

        # ── declare parameters ─────────────────────────────────────────────
        self.declare_parameter('waypoints_file', 'waypoints.csv')
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('control_frequency', 20.0)  # 20 Hz matches dt=0.05s
        self.declare_parameter('horizon_length', 30)
        self.declare_parameter('visualize', True)

        # Vehicle geometry — F1Tenth scale (wheelbase ~0.33 m)
        self.declare_parameter('a', 0.17)              # front axle to CG      [m]
        self.declare_parameter('b', 0.16)              # rear axle to CG       [m]
        self.declare_parameter('wheel_base', 0.3302)
        self.declare_parameter('body_length', 0.5)
        self.declare_parameter('body_width', 0.2032)   # from xacro
        self.declare_parameter('radius', 0.2)          # collision radius       [m]

        # Mass / inertia — F1Tenth 1:10 scale (Traxxas with electronics ~3.74 kg)
        self.declare_parameter('m', 3.74)              # vehicle mass           [kg]
        self.declare_parameter('Iz', 0.04)             # yaw moment of inertia  [kg*m^2]

        # Tire parameters — Fiala brush model, scaled for F1Tenth
        self.declare_parameter('Cc_f', 300.0)          # front cornering stiff. [N/rad]
        self.declare_parameter('Cc_r', 300.0)          # rear cornering stiff.  [N/rad]
        self.declare_parameter('mu', 1.0)              # friction coefficient
        self.declare_parameter('r_w', 0.0508)          # wheel radius from xacro [m]
        self.declare_parameter('gamma', 0.95)          # numeric stability param

        # Input limits
        self.declare_parameter('delta_max', np.deg2rad(24.0))     # [rad]
        self.declare_parameter('delta_dot_max', np.deg2rad(40.0)) # [rad/s] - reduced to prevent rapid oscillation
        self.declare_parameter('tau_max', 5.0)                    # [Nm]
        self.declare_parameter('tau_dot_max', 10.0)               # [Nm/s]

        # State limits
        self.declare_parameter('v_max', 3.0)
        self.declare_parameter('v_min', 0.1)
        self.declare_parameter('r_max', 6.0)
        self.declare_parameter('beta_max', np.deg2rad(45.0))      # [rad]
        self.declare_parameter('v_psi_max', 5.0)                  # max progress rate [m/s]

        # MPCC cost weights — balanced for F1Tenth stable path tracking
        self.declare_parameter('Q_c', 70.0)        # contouring error (perpendicular to path) - reduced to prevent oscillation
        self.declare_parameter('Q_l', 60.0)        # lag error (along path) - increased to prioritize forward progress
        self.declare_parameter('Q_theta', 5.0)    # heading error (reduced to prevent early turns)
        self.declare_parameter('Q_v', 10.0)        # velocity tracking weight
        self.declare_parameter('Q_r', 10.0)        # yaw rate damping - reduced to allow smooth turns
        self.declare_parameter('v_ref', 1.8)       # target speed [m/s] - increased for better momentum
        # R vector order: [delta_dot, tau_dot, v_psi] — matches MPCC.set_rterm order
        self.declare_parameter('R_delta_dot', 120.0) # higher → smoother steering (prevent wall-hitting oscillation)
        self.declare_parameter('R_tau_dot', 0.1)
        self.declare_parameter('R_vpsi', 0.0)
        self.declare_parameter('v_psi_ref', 1.5)  # desired progress rate [m/s]

        # ── read parameters ────────────────────────────────────────────────
        waypoints_file    = self.get_parameter('waypoints_file').value
        dt                = self.get_parameter('dt').value
        self.control_freq = self.get_parameter('control_frequency').value
        horizon           = self.get_parameter('horizon_length').value
        self.visualize    = self.get_parameter('visualize').value
        v_ref             = self.get_parameter('v_ref').value
        v_psi_ref         = self.get_parameter('v_psi_ref').value

        # robot_spec — key names match DynamicBicycle2D.__init__ setdefault() calls exactly
        robot_spec = {
            'a':             self.get_parameter('a').value,
            'b':             self.get_parameter('b').value,
            'wheel_base':    self.get_parameter('wheel_base').value,
            'body_length':   self.get_parameter('body_length').value,
            'body_width':    self.get_parameter('body_width').value,
            'radius':        self.get_parameter('radius').value,
            'm':             self.get_parameter('m').value,
            'Iz':            self.get_parameter('Iz').value,
            'Cc_f':          self.get_parameter('Cc_f').value,
            'Cc_r':          self.get_parameter('Cc_r').value,
            'mu':            self.get_parameter('mu').value,
            'r_w':           self.get_parameter('r_w').value,
            'gamma':         self.get_parameter('gamma').value,
            'delta_max':     self.get_parameter('delta_max').value,
            'delta_dot_max': self.get_parameter('delta_dot_max').value,
            'tau_max':       self.get_parameter('tau_max').value,
            'tau_dot_max':   self.get_parameter('tau_dot_max').value,
            'v_max':         self.get_parameter('v_max').value,
            'v_min':         self.get_parameter('v_min').value,
            'r_max':         self.get_parameter('r_max').value,
            'beta_max':      self.get_parameter('beta_max').value,
            'v_psi_max':     self.get_parameter('v_psi_max').value,
        }

        # R weight vector: order must match MPCC._create_mpc() set_rterm(u=self.R)
        # which applies to [delta_dot, tau_dot, v_psi] in that order
        R = np.array([
            self.get_parameter('R_delta_dot').value,
            self.get_parameter('R_tau_dot').value,
            self.get_parameter('R_vpsi').value,
        ])

        # ── load waypoints ─────────────────────────────────────────────────
        self.get_logger().info(f'Loading waypoints from: {waypoints_file}')
        path_x, path_y = self._load_waypoints(waypoints_file)
        self.get_logger().info(f'Loaded {len(path_x)} waypoints')

        # ── DriftingCar — ax=None suppresses all matplotlib code ───────────
        # Placeholder initial state; car.X is overwritten from odom before first solve.
        X0 = np.array([
            path_x[0], path_y[0],  # x, y      [m]
            0.0,                   # theta      [rad]  — overwritten from odom
            0.0,                   # r          [rad/s]
            0.0,                   # beta       [rad]
            v_ref,                 # V          [m/s]  — start near target speed
            0.0,                   # delta      [rad]
            0.0,                   # tau        [Nm]
        ])
        self.car = DriftingCar(X0, robot_spec, dt, ax=None)
        robot_spec['model'] = 'DriftingCar'

        # ── MPCC ───────────────────────────────────────────────────────────
        # MPCC.__init__ checks robot_spec['model'] == 'DriftingCar'
        # DriftingCar.__init__ sets robot_spec['model'] = 'DriftingCar' automatically
        self.mpcc = MPCC(self.car, robot_spec, show_mpc_traj=False, horizon=horizon)

        self.mpcc.set_reference_path(path_x, path_y)

        # set_cost_weights() rebuilds the entire MPC + simulator + estimator internally
        self.mpcc.set_cost_weights(
            Q_c=     self.get_parameter('Q_c').value,
            Q_l=     self.get_parameter('Q_l').value,
            Q_theta= self.get_parameter('Q_theta').value,
            Q_v=     self.get_parameter('Q_v').value,
            Q_r=     self.get_parameter('Q_r').value,
            v_ref=   v_ref,
            R=       R,
        )
        # set_progress_rate() sets self.v_psi_ref used inside the TVP lookahead function
        self.mpcc.set_progress_rate(v_psi_ref)

        # ── QoS ───────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, qos)

        # ── publishers ────────────────────────────────────────────────────
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)

        if self.visualize:
            self.pred_path_pub = self.create_publisher(Path, '/mpcc/predicted_path', qos)
            self.ref_path_pub  = self.create_publisher(Path, '/mpcc/reference_path',  qos)

        # ── control timer ─────────────────────────────────────────────────
        self.create_timer(1.0 / self.control_freq, self.control_loop)

        self.get_logger().info('MPCC node ready')
        self.get_logger().info(
            f'Control: {self.control_freq} Hz | dt={dt} s | horizon={horizon} steps'
        )

    # ── waypoint loader ────────────────────────────────────────────────────

    def _load_waypoints(self, filepath: str):
        """Load x_m, y_m columns from CSV. Returns (path_x, path_y) numpy arrays."""
        p = pathlib.Path(filepath)
        if not p.exists():
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('mpcc_controller')
            p = pathlib.Path(pkg_dir) / 'config' / filepath

        rows = []
        with open(p, 'r') as f:
            for row in csv.DictReader(f):
                rows.append([float(row['x_m']), float(row['y_m'])])

        rows = list(reversed(rows))   # Waypoints in correct direction - reversal disabled
        arr  = np.array(rows)
        return arr[:, 0], arr[:, 1]

    # ── odom callback ──────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        """
        Cache observable states from /ego_racecar/odom.

        Directly observable:
            x, y   — msg.pose.pose.position.x/y
            theta  — yaw from quaternion
            V      — msg.twist.twist.linear.x     (body-frame longitudinal speed)
            r      — msg.twist.twist.angular.z    (yaw rate, available in odom)

        Not in /odom (handled in control_loop):
            beta   — held at 0
            delta  — integrated from delta_dot
            tau    — integrated from tau_dot
        """
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y

        # Quaternion → yaw (theta)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self._theta = np.arctan2(siny_cosp, cosy_cosp)

        # Yaw rate — directly from odom, unused in old node
        self._r = msg.twist.twist.angular.z

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._V = np.sqrt(vx**2 + vy**2)  # true speed magnitude
        self._beta = np.arctan2(vy, vx) if np.sqrt(vx**2 + vy**2) > 0.5 else self._beta

        self.get_logger().info(
                f'odom: '
                f'x={self._x:.2f} m  y={self._y:.2f} m  '
                f'theta={np.rad2deg(self._theta):.1f} deg  '
                f'V={self._V:.2f} m/s  r={self._r:.3f} rad/s'
                f'beta={np.rad2deg(self._beta):.1f} deg'
            )
        if not self.initialized:
            self.initialized = True
            self.get_logger().info(
                f'First odom: '
                f'x={self._x:.2f} m  y={self._y:.2f} m  '
                f'theta={np.rad2deg(self._theta):.1f} deg  '
                f'V={self._V:.2f} m/s  r={self._r:.3f} rad/s'
                f'beta={np.rad2deg(self._beta):.1f} deg'
            )

    # ── control loop ───────────────────────────────────────────────────────

    def control_loop(self):
        """
        Main control loop at control_frequency Hz.

        Steps:
          1. Assemble 8-state vector from odom snapshots + running estimates.
          2. Inject into car.X  — NOT calling car.step(); the sim owns dynamics.
          3. Call mpcc.solve_control_problem(state) → U = [delta_dot, tau_dot].
          4. Integrate _delta_est and _tau_est forward by car.dt.
          5. Derive v_target from _tau_est and publish AckermannDriveStamped.
          6. Optionally publish predicted/reference paths for RViz.

        Error computation (e_c, e_l, e_theta, e_v) is entirely inside
        MPCC._create_model() as CasADi symbolic expressions baked into the
        optimizer. This node never computes errors explicitly.
        """
        if not self.initialized or self._x is None:
            return

        try:
            # ── 1. assemble full 8-state vector ───────────────────────────
            # Index mapping from DriftingCar (drifting_car.py line 50):
            #   X[0]=x  X[1]=y  X[2]=theta  X[3]=r  X[4]=beta  X[5]=V  X[6]=delta  X[7]=tau
            state = np.array([
                self._x,           # X[0]  x       [m]      — odom position
                self._y,           # X[1]  y       [m]      — odom position
                self._theta,       # X[2]  theta   [rad]    — odom quaternion
                self._r,           # X[3]  r       [rad/s]  — odom angular.z
                self._beta,        # X[4]  beta    [rad]    — held 0
                self._V,           # X[5]  V       [m/s]    — odom linear.x
                self._delta_est,   # X[6]  delta   [rad]    — integrated estimate
                self._tau_est,     # X[7]  tau     [Nm]     — integrated estimate
            ])

            # ── 2. inject real state into DriftingCar ─────────────────────
            # car.X is read by MPCC.solve_control_problem() via robot.get_state().
            # Writing directly bypasses car.step() — correct for real/sim deployment.
            self.car.X = state.reshape(-1, 1)

            # ── 3. solve ───────────────────────────────────────────────────
            # Internally: finds closest path point (_find_closest_path_point),
            # builds TVP reference over horizon (_set_tvp), runs IPOPT,
            # stores predictions (_store_predictions), steps do-mpc simulator.
            # Returns U[:2] — [delta_dot [rad/s], tau_dot [Nm/s]] as (2,1).
            U = self.mpcc.solve_control_problem(self.car.get_state())

            delta_dot = float(U[0, 0])   # [rad/s]
            tau_dot   = float(U[1, 0])   # [Nm/s]

            # ── 4. integrate echo states ───────────────────────────────────
            # delta and tau are rate-controlled in the DynamicBicycle2D model:
            #   delta_next = delta + delta_dot * dt
            #   tau_next   = tau   + tau_dot   * dt
            # We mirror that integration here so the next tick's state is consistent.
            dt = self.car.dt
            self._delta_est = float(np.clip(
                self._delta_est + delta_dot * dt,
                -self.car.robot_spec['delta_max'],
                 self.car.robot_spec['delta_max'],
            ))
            self._tau_est = float(np.clip(
                self._tau_est + tau_dot * dt,
                -self.car.robot_spec['tau_max'],
                 self.car.robot_spec['tau_max'],
            ))

            # ── 5. publish AckermannDriveStamped ──────────────────────────
            # steering_angle: absolute integrated delta_est [rad].
            # speed: use mpcc.v_ref directly — the MPCC already regulates
            #   velocity to v_ref via the e_v = V - v_ref cost term.
            #   Sending v_ref as the speed setpoint is consistent with what
            #   the optimizer assumes the vehicle should be doing.
            #   Clipped to [v_min, v_max] from robot_spec as a safety guard.
            v_target = float(np.clip(
                self.mpcc.v_ref,
                self.car.robot_spec['v_min'],
                self.car.robot_spec['v_max'],
            ))

            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp    = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.steering_angle = self._delta_est   # [rad]
            drive_msg.drive.speed          = v_target           # [m/s]
            self.drive_pub.publish(drive_msg)

            self.get_logger().info(
                f'x={self._x:.2f} y={self._y:.2f}  '
                f'V={self._V:.2f} m/s  '
                f'delta={np.rad2deg(self._delta_est):.1f} deg  '
                f'v_target={v_target:.2f} m/s  '
                f'solver={self.mpcc.status}',
                throttle_duration_sec=1.0,
            )

            # ── 6. visualization ───────────────────────────────────────────
            if self.visualize:
                self._publish_visualization()

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
            self.get_logger().error(traceback.format_exc())

    # ── visualization ──────────────────────────────────────────────────────

    def _publish_visualization(self):
        """
        Publish MPC predicted trajectory and reference lookahead as nav_msgs/Path.

        mpcc.get_predictions()       — returns (predicted_states, predicted_inputs)
                                       predicted_states shape: (2, horizon+1) — x,y only
                                       stored in mpcc.predicted_states by _store_predictions()

        mpcc.get_reference_horizon() — returns reference_horizon shape: (2, horizon+1)
                                       stored in mpcc.reference_horizon by _set_tvp()
        """
        stamp = self.get_clock().now().to_msg()

        pred_states, _ = self.mpcc.get_predictions()
        if pred_states is not None:
            pred_path = Path()
            pred_path.header.stamp    = stamp
            pred_path.header.frame_id = 'map'
            for i in range(pred_states.shape[1]):
                pose = PoseStamped()
                pose.header = pred_path.header
                pose.pose.position.x = float(pred_states[0, i])
                pose.pose.position.y = float(pred_states[1, i])
                pred_path.poses.append(pose)
            self.pred_path_pub.publish(pred_path)

        ref_horizon = self.mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_path = Path()
            ref_path.header.stamp    = stamp
            ref_path.header.frame_id = 'map'
            for i in range(ref_horizon.shape[1]):
                pose = PoseStamped()
                pose.header = ref_path.header
                pose.pose.position.x = float(ref_horizon[0, i])
                pose.pose.position.y = float(ref_horizon[1, i])
                ref_path.poses.append(pose)
            self.ref_path_pub.publish(ref_path)

    # ── cleanup ────────────────────────────────────────────────────────────

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPCCNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('Keyboard interrupt — shutting down')
    except Exception as e:
        print(f'Executor error: {e}')
        traceback.print_exc()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print('Shutdown complete')


if __name__ == '__main__':
    main()