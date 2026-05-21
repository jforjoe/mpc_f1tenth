"""Waypoint loading, nearest-index search, and horizon reference extraction."""

import numpy as np
import pandas as pd


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_waypoints(path):
    """Load waypoints CSV into Nx3 array [x, y, v_ref].

    Supports columns: (x_m, y_m, vx_mps) or (x_m, y_m, velocity_mps[, heading_rad]).
    """
    df = pd.read_csv(path)
    cols = {c.strip(): c for c in df.columns}
    x = df[cols['x_m']].to_numpy(dtype=float)
    y = df[cols['y_m']].to_numpy(dtype=float)
    if 'vx_mps' in cols:
        v = df[cols['vx_mps']].to_numpy(dtype=float)
    elif 'velocity_mps' in cols:
        v = df[cols['velocity_mps']].to_numpy(dtype=float)
    else:
        v = np.full_like(x, np.nan)
    return np.stack([x, y, v], axis=1)


def compute_yaw_from_path(wps_xy, loop=True):
    """Finite-difference heading per waypoint."""
    n = len(wps_xy)
    yaw = np.zeros(n)
    for i in range(n):
        j = (i + 1) % n if loop else min(i + 1, n - 1)
        dx = wps_xy[j, 0] - wps_xy[i, 0]
        dy = wps_xy[j, 1] - wps_xy[i, 1]
        yaw[i] = np.arctan2(dy, dx)
    return yaw


def nearest_index(wps_xy, pos, last_idx=None, search_window=50, loop=True):
    """Windowed nearest-neighbor; falls back to global on big jumps."""
    n = len(wps_xy)
    if last_idx is None:
        d = np.linalg.norm(wps_xy - pos, axis=1)
        return int(np.argmin(d))
    if loop:
        idxs = np.array([(last_idx + k) % n for k in range(-5, search_window)])
    else:
        lo = max(0, last_idx - 5)
        hi = min(n, last_idx + search_window)
        idxs = np.arange(lo, hi)
    d = np.linalg.norm(wps_xy[idxs] - pos, axis=1)
    cand = int(idxs[int(np.argmin(d))])
    # Sanity: if jump is huge, redo global search.
    if np.linalg.norm(wps_xy[cand] - pos) > 5.0:
        d = np.linalg.norm(wps_xy - pos, axis=1)
        cand = int(np.argmin(d))
    return cand


def extract_horizon(wps, yaw, idx, N, dt, v_target, loop=True):
    """Build a reference of length N+1 by sampling waypoints along arc length.

    Each step advances by approximately v_target * dt meters along the path.
    Returns array shape (4, N+1): rows = [x_ref, y_ref, yaw_ref, v_ref].
    Yaw column is unwrapped along the horizon for continuity.
    """
    n = len(wps)
    ref = np.zeros((4, N + 1))
    step_dist = max(0.05, float(v_target) * float(dt))

    # Precompute cumulative arc length starting from idx
    cur = idx
    accumulated = 0.0
    target_dist = 0.0
    ref[0, 0] = wps[cur, 0]
    ref[1, 0] = wps[cur, 1]
    ref[2, 0] = yaw[cur]
    ref[3, 0] = wps[cur, 2] if not np.isnan(wps[cur, 2]) else v_target

    for k in range(1, N + 1):
        target_dist += step_dist
        # Walk forward along path until accumulated >= target_dist
        while accumulated < target_dist:
            nxt = (cur + 1) % n if loop else min(cur + 1, n - 1)
            seg = float(np.linalg.norm(wps[nxt, :2] - wps[cur, :2]))
            accumulated += seg
            cur = nxt
            if not loop and cur == n - 1:
                break
        ref[0, k] = wps[cur, 0]
        ref[1, k] = wps[cur, 1]
        ref[2, k] = yaw[cur]
        ref[3, k] = wps[cur, 2] if not np.isnan(wps[cur, 2]) else v_target

    # Unwrap yaw along horizon for solver continuity
    ref[2, :] = np.unwrap(ref[2, :])
    return ref
