"""Symbolic kinematic bicycle model (CasADi) matching f1tenth_gym vehicle_dynamics_ks.

State  x = [px, py, delta, v, theta]
Input  u = [steer_vel, accel]
"""

import casadi as ca


def vehicle_dynamics_ks(x, u, wheelbase):
    px, py, delta, v, theta = x[0], x[1], x[2], x[3], x[4]
    steer_vel, accel = u[0], u[1]
    return ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        steer_vel,
        accel,
        v / wheelbase * ca.tan(delta),
    )


def rk4_step(x, u, dt, wheelbase):
    k1 = vehicle_dynamics_ks(x, u, wheelbase)
    k2 = vehicle_dynamics_ks(x + 0.5 * dt * k1, u, wheelbase)
    k3 = vehicle_dynamics_ks(x + 0.5 * dt * k2, u, wheelbase)
    k4 = vehicle_dynamics_ks(x + dt * k3, u, wheelbase)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
