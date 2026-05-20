"""Kinematic MPC NLP built once with CasADi Opti, solved each tick with IPOPT."""

import casadi as ca
import numpy as np

try:
    from .kinematic_model import rk4_step
except Exception:
    # Allow running both as a package or as a standalone script
    from kinematic_model import rk4_step


class KinematicMPC:
    def __init__(
        self,
        N,
        dt,
        wheelbase,
        max_steer,
        max_steer_vel,
        max_accel,
        min_speed,
        max_speed,
        Q,                # length 4: [Q_x, Q_y, Q_yaw, Q_v]
        R,                # length 2: [R_steer_vel, R_accel]
        Rd,               # length 2: [Rd_steer_vel, Rd_accel]
        qf_scale=2.0,
        ipopt_print_level=0,
        ipopt_max_iter=50,
    ):
        self.N = N
        self.dt = dt
        self.L = wheelbase

        opti = ca.Opti()

        X = opti.variable(5, N + 1)         # [px, py, delta, v, theta]
        U = opti.variable(2, N)             # [steer_vel, accel]

        x0 = opti.parameter(5)
        Xref = opti.parameter(4, N + 1)     # [x_ref, y_ref, yaw_ref, v_ref]
        u_prev = opti.parameter(2)

        # Initial state
        opti.subject_to(X[:, 0] == x0)

        # Dynamics constraints
        for k in range(N):
            x_next = rk4_step(X[:, k], U[:, k], dt, wheelbase)
            opti.subject_to(X[:, k + 1] == x_next)

        # State bounds — speed bound starts at k=1, not k=0.
        # X[:,0] is pinned by the equality X[:,0]==x0 (initial state).
        # If measured speed < min_speed (e.g., car stopped at startup), including k=0
        # creates a direct contradiction: X[3,0]==v_measured AND X[3,0]>=min_speed → infeasible.
        opti.subject_to(opti.bounded(-max_steer, X[2, :], max_steer))
        opti.subject_to(opti.bounded(min_speed, X[3, 1:], max_speed))

        # Input bounds
        opti.subject_to(opti.bounded(-max_steer_vel, U[0, :], max_steer_vel))
        opti.subject_to(opti.bounded(-max_accel, U[1, :], max_accel))

        # Cost
        cost = 0
        Qx, Qy, Qyaw, Qv = Q
        Rsv, Ra = R
        Rdsv, Rda = Rd

        for k in range(N):
            ex = X[0, k] - Xref[0, k]
            ey = X[1, k] - Xref[1, k]
            # Yaw error wrapped via atan2(sin, cos) for solver-safe continuity
            eyaw_raw = X[4, k] - Xref[2, k]
            eyaw = ca.atan2(ca.sin(eyaw_raw), ca.cos(eyaw_raw))
            ev = X[3, k] - Xref[3, k]
            cost += Qx * ex ** 2 + Qy * ey ** 2 + Qyaw * eyaw ** 2 + Qv * ev ** 2
            cost += Rsv * U[0, k] ** 2 + Ra * U[1, k] ** 2
            if k == 0:
                dsv = U[0, k] - u_prev[0]
                da = U[1, k] - u_prev[1]
            else:
                dsv = U[0, k] - U[0, k - 1]
                da = U[1, k] - U[1, k - 1]
            cost += Rdsv * dsv ** 2 + Rda * da ** 2

        # Terminal cost
        ex = X[0, N] - Xref[0, N]
        ey = X[1, N] - Xref[1, N]
        eyaw_raw = X[4, N] - Xref[2, N]
        eyaw = ca.atan2(ca.sin(eyaw_raw), ca.cos(eyaw_raw))
        ev = X[3, N] - Xref[3, N]
        cost += qf_scale * (Qx * ex ** 2 + Qy * ey ** 2 + Qyaw * eyaw ** 2 + Qv * ev ** 2)

        opti.minimize(cost)

        opts = {
            'ipopt.print_level': ipopt_print_level,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': ipopt_max_iter,
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }
        opti.solver('ipopt', opts)

        self.opti = opti
        self.X = X
        self.U = U
        self.x0_p = x0
        self.Xref_p = Xref
        self.u_prev_p = u_prev

        # Warm-start storage
        self._X_prev = np.zeros((5, N + 1))
        self._U_prev = np.zeros((2, N))

    def solve(self, x0, x_ref, u_prev):
        """Solve one MPC step.

        Returns: (u0, X_pred, U_pred, ok)
        """
        # Shift previous trajectory by one as warm start
        Xws = np.roll(self._X_prev, -1, axis=1)
        Xws[:, -1] = self._X_prev[:, -1]
        Xws[:, 0] = x0
        Uws = np.roll(self._U_prev, -1, axis=1)
        Uws[:, -1] = self._U_prev[:, -1]

        self.opti.set_value(self.x0_p, x0)
        self.opti.set_value(self.Xref_p, x_ref)
        self.opti.set_value(self.u_prev_p, u_prev)
        self.opti.set_initial(self.X, Xws)
        self.opti.set_initial(self.U, Uws)

        try:
            sol = self.opti.solve()
            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            self._X_prev = X_sol
            self._U_prev = U_sol
            return U_sol[:, 0], X_sol, U_sol, True
        except RuntimeError:
            # Recover debug values even on failure
            try:
                X_sol = self.opti.debug.value(self.X)
                U_sol = self.opti.debug.value(self.U)
                return U_sol[:, 0], X_sol, U_sol, False
            except Exception:
                return np.zeros(2), Xws, Uws, False
