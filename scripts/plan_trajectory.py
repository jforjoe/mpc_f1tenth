#!/usr/bin/env python3
"""
End-to-end offline trajectory planner for F1TENTH.

Input  : map PNG + YAML (ROS occupancy grid).
Output : raceline.csv  (x_m, y_m, vx_mps, heading_rad, kappa_radpm)
         + diagnostic PNGs in PLOT_DIR.

Pipeline (TUM `global_racetrajectory_optimization` two-step style):

  [1]  load_map ............ parse YAML + load PNG, return free-mask.
  [2]  safe_corridor ....... distance-transform → erode by MIN_WALL_DIST_M.
  [2b] isolate_track ....... floodfill from TRACK_SEED_XY → keep only the
                              connected component containing the seed.
                              (Otherwise free space *outside* the track is
                              included and the skeleton traces the wrong loop.)
  [3]  skeletonize_and_prune skimage skeletonize + iterative spur pruning.
  [4]  order_skeleton_loop . 8-connected graph traversal → ordered polyline.
  [5]  pixels_to_world ..... ROS map_server pixel → world transform.
  [6]  resample_arc ........ uniform arc-length resampling (RESAMPLE_DS).
  [7]  tangent_normal ...... centered-diff tangents + 90° CCW unit normals.
  [8]  compute_widths ...... ray-march along ±normal → asymmetric (w_l, w_r).
  [9]  solve_min_curvature . iterative re-linearized bounded QP with
                              Tikhonov regularization (LAMBDA_OFFSET·||α||²)
                              pulling toward the centerline so the QP cannot
                              collapse to a circle.
  [10] fit_spline / kappa .. periodic cubic spline + analytic curvature.
  [11] velocity_profile .... friction-circle cap + forward/backward pass.
  [12] write_csv ........... emit final raceline.csv.
  [13] visualize ........... 6-panel diagnostic summary.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.ndimage import label

# =============================================================================
#                              TUNABLE MACROS
# =============================================================================

# --- Files ---
MAP_YAML            = '/sim_ws/src/f1tenth_gym_ros/maps/map.yaml'
OUTPUT_CSV          = '/sim_ws/src/mpc_f1tenth/config/raceline.csv'
PLOT_DIR            = '/sim_ws/src/mpc_f1tenth/config/plots'

# --- Track isolation (handles maps where space outside the track is also free) ---
ISOLATE_TRACK       = True
TRACK_SEED_XY       = (0.0, 0.0)   # world-coord seed somewhere on the track (e.g. spawn pose)

# --- Centerline extraction ---
MIN_WALL_DIST_M     = 0.30        # erosion distance from walls when computing skeleton
SPUR_PRUNE_ITERS    = 200         # max passes of degree-1 removal
MIN_SAFE_PIXELS     = 500         # abort if corridor too small after erosion
CLOSED_LOOP         = True

# --- Resampling & raceline QP ---
RESAMPLE_DS         = 0.10        # m, uniform arc-length spacing
CAR_HALF_WIDTH      = 0.15        # m, car half-width + buffer
MAX_WIDTH_SEARCH    = 3.0         # m, cap on ray-march per side
N_REFINE            = 3           # min-curvature re-linearization passes
LAMBDA_OFFSET       = 0.01        # Tikhonov weight on ||alpha||^2 — pulls toward centerline.
                                  # Raise (e.g. 0.05) if the raceline still drifts unrealistically.
                                  # Lower (e.g. 1e-4) for aggressive corner-cutting on wide tracks.
# LAMBDA_OFFSET       = 1e-4

LBFGSB_MAXITER      = 300
LBFGSB_FTOL         = 1e-10

# --- Vehicle friction (velocity profile) ---
MU, G               = 1.0, 9.81
A_LAT_MAX           = MU * G
A_LONG_ACC_MAX      = 7.5
A_LONG_BRK_MAX      = 9.5
V_MAX, V_MIN        = 8.0, 1.5
V_START             = 2.0          # only used if CLOSED_LOOP=False

# --- Visualization ---
SHOW_PLOTS          = True
SAVE_PLOTS          = True
DPI                 = 150

# =============================================================================


# ---------- [1] Map I/O ----------
def load_map(yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_rel = meta['image']
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(os.path.dirname(yaml_path), img_rel)
    res = float(meta['resolution'])
    origin = meta['origin']
    free_thresh = float(meta.get('free_thresh', 0.196))
    negate = bool(meta.get('negate', 0))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')
    p = img.astype(np.float32) / 255.0 if negate else (255.0 - img.astype(np.float32)) / 255.0
    free = p < free_thresh
    return img, free, res, origin


def world_to_pixel(x, y, res, origin, H):
    c = int((x - origin[0]) / res)
    r = int(H - (y - origin[1]) / res)
    return r, c


# ---------- [2] Safe corridor ----------
def safe_corridor(free, res, min_wall_dist_m):
    dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    thresh_px = min_wall_dist_m / res
    corridor = dt > thresh_px
    return corridor, dt, thresh_px


# ---------- [2b] Track isolation via flood-fill from a seed ----------
def isolate_track(corridor, seed_xy, res, origin):
    """Keep only the connected component of `corridor` containing the seed.

    If the seed lands outside the corridor (e.g. because the spawn is on a wall
    pixel), snap to the nearest corridor pixel and warn.
    """
    H = corridor.shape[0]
    r0, c0 = world_to_pixel(seed_xy[0], seed_xy[1], res, origin, H)
    if not (0 <= r0 < corridor.shape[0] and 0 <= c0 < corridor.shape[1]):
        raise RuntimeError(f'Seed {seed_xy} is outside the map.')
    if not corridor[r0, c0]:
        true_pix = np.argwhere(corridor)
        if len(true_pix) == 0:
            raise RuntimeError('Corridor is empty — relax MIN_WALL_DIST_M.')
        d2 = (true_pix[:, 0] - r0) ** 2 + (true_pix[:, 1] - c0) ** 2
        r0, c0 = true_pix[int(np.argmin(d2))]
        print(f'    seed snapped to nearest corridor pixel ({r0}, {c0})')
    labeled, ncomp = label(corridor, structure=np.ones((3, 3), dtype=int))
    seed_label = labeled[r0, c0]
    track = (labeled == seed_label)
    print(f'    isolated component {seed_label}/{ncomp}: {track.sum()} px '
          f'(dropped {corridor.sum() - track.sum()} px outside track)')
    return track


# ---------- [3] Skeletonize + spur pruning ----------
def skeletonize_and_prune(mask, max_iters):
    skel = skeletonize(mask).astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    for _ in range(max_iters):
        nbr = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = skel & (nbr == 1)
        if not endpoints.any():
            break
        skel = skel & ~endpoints
    return skel


# ---------- [4] Skeleton loop ordering ----------
def order_skeleton_loop(skel, closed):
    pix = np.argwhere(skel)
    if len(pix) == 0:
        raise RuntimeError('Skeleton is empty.')
    pix_set = {tuple(p) for p in pix}
    nbrs8 = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1),          ( 0, 1),
             ( 1, -1), ( 1, 0), ( 1, 1)]

    def neighbors(p):
        return [(p[0] + dr, p[1] + dc) for dr, dc in nbrs8 if (p[0] + dr, p[1] + dc) in pix_set]

    if closed:
        start = tuple(pix[0])
    else:
        endpoints = [p for p in pix_set if len(neighbors(p)) == 1]
        if not endpoints:
            raise RuntimeError('No endpoints found but CLOSED_LOOP=False.')
        start = endpoints[0]

    ordered = [start]
    visited = {start}
    prev = None
    cur = start
    while True:
        nbs = [n for n in neighbors(cur)
               if n not in visited or (closed and n == start and len(ordered) > 3)]
        if not nbs:
            break
        if prev is None:
            nxt = nbs[0]
        else:
            d_prev = (cur[0] - prev[0], cur[1] - prev[1])
            def cont_score(n):
                d = (n[0] - cur[0], n[1] - cur[1])
                return -(d[0] * d_prev[0] + d[1] * d_prev[1])
            nxt = min(nbs, key=cont_score)
        if closed and nxt == start and len(ordered) > 3:
            break
        ordered.append(nxt)
        visited.add(nxt)
        prev = cur
        cur = nxt
        if len(ordered) > len(pix_set) + 5:
            break
    return np.array(ordered, dtype=int)


# ---------- [5] Pixel → world ----------
def pixels_to_world(rc, res, origin, H):
    r = rc[:, 0]; c = rc[:, 1]
    x = origin[0] + c * res
    y = origin[1] + (H - r) * res
    return np.stack([x, y], axis=1)


# ---------- [6] Resample uniformly ----------
def resample_arc(pts, ds, closed):
    if closed:
        pts_c = np.vstack([pts, pts[:1]])
    else:
        pts_c = pts
    seg = np.linalg.norm(np.diff(pts_c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(round(L / ds)), 20)
    s_new = np.linspace(0.0, L, n, endpoint=not closed)
    x = np.interp(s_new, s, pts_c[:, 0])
    y = np.interp(s_new, s, pts_c[:, 1])
    return np.stack([x, y], axis=1), L


# ---------- [7] Tangents + normals ----------
def tangent_normal(pts, closed):
    if closed:
        prev = np.roll(pts,  1, axis=0)
        nxt  = np.roll(pts, -1, axis=0)
    else:
        prev = np.vstack([pts[:1],  pts[:-1]])
        nxt  = np.vstack([pts[1:],  pts[-1:]])
    t = nxt - prev
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return t, n


# ---------- [8] Ray-march widths ----------
def ray_width(start, direction, free, res, origin, max_dist):
    H, W = free.shape
    step = 0.5 * res
    n_steps = int(max_dist / step) + 1
    for k in range(1, n_steps + 1):
        d = k * step
        r, c = world_to_pixel(start[0] + d * direction[0],
                              start[1] + d * direction[1],
                              res, origin, H)
        if r < 0 or r >= H or c < 0 or c >= W or not free[r, c]:
            return d
    return max_dist


def compute_widths(c, n_hat, free, res, origin, max_dist, car_half):
    w_l = np.zeros(len(c))
    w_r = np.zeros(len(c))
    for i, (p, n) in enumerate(zip(c, n_hat)):
        w_l[i] = max(ray_width(p,  n, free, res, origin, max_dist) - car_half, 0.05)
        w_r[i] = max(ray_width(p, -n, free, res, origin, max_dist) - car_half, 0.05)
    return w_l, w_r


# ---------- [9] Min-curvature QP with Tikhonov regularization ----------
def second_diff_matrix(N, ds, closed):
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N)
    S = diags([off[:-1], main, off[:-1]], [-1, 0, 1], shape=(N, N), format='lil')
    if closed:
        S[0, -1] = 1.0
        S[-1, 0] = 1.0
    else:
        S[0, 0] = 1.0; S[0, 1] = -2.0; S[0, 2] = 1.0
        S[-1, -1] = 1.0; S[-1, -2] = -2.0; S[-1, -3] = 1.0
    return (S / (ds * ds)).tocsr()


def solve_min_curvature(c, n_hat, ds, w_l, w_r, closed, n_refine, lam):
    """Solve: min  ||(c + alpha*n)''||^2 + lam*||alpha||^2
              s.t. -w_r <= alpha <= w_l, iteratively re-linearized.
    """
    N = len(c)
    S = second_diff_matrix(N, ds, closed)
    Nx = diags(n_hat[:, 0], 0, format='csr')
    Ny = diags(n_hat[:, 1], 0, format='csr')
    Ax = (S @ Nx).toarray()
    Ay = (S @ Ny).toarray()
    H_quad = Ax.T @ Ax + Ay.T @ Ay + lam * np.eye(N)

    alpha = np.zeros(N)
    for it in range(n_refine):
        r = c + n_hat * alpha[:, None]
        c2 = S @ r
        g = Ax.T @ c2[:, 0] + Ay.T @ c2[:, 1] + lam * alpha
        lb = -w_r - alpha
        ub =  w_l - alpha
        res = minimize(
            fun=lambda d: 0.5 * d @ H_quad @ d + g @ d,
            x0=np.zeros(N),
            jac=lambda d: H_quad @ d + g,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': LBFGSB_MAXITER, 'ftol': LBFGSB_FTOL},
        )
        alpha = alpha + res.x
        print(f'    QP iter {it+1}: cost={res.fun:.4f}, |alpha|_max={np.abs(alpha).max():.3f} m')
    return alpha


# ---------- [10] Spline + curvature ----------
def fit_spline(pts, closed):
    if closed:
        pts_c = np.vstack([pts, pts[:1]])
    else:
        pts_c = pts
    seg = np.linalg.norm(np.diff(pts_c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    sp = CubicSpline(s, pts_c, bc_type='periodic' if closed else 'natural')
    return sp, s[-1]


def curvature_from_spline(sp, s_query):
    d1 = sp(s_query, 1)
    d2 = sp(s_query, 2)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return num / (den + 1e-12), d1


# ---------- [11] Velocity profile ----------
def velocity_profile(kappa, ds, closed, v_max, v_min, v_start, a_lat, a_acc, a_brk):
    n = len(kappa)
    v_lim = np.sqrt(a_lat / (np.abs(kappa) + 1e-6))
    v = np.clip(v_lim, v_min, v_max)
    if not closed:
        v[0] = min(v[0], v_start)
    for _ in range(2 if closed else 1):
        for i in range(n - 1):
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_acc * ds))
        if closed:
            v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2 * a_acc * ds))
    for _ in range(2 if closed else 1):
        for i in range(n - 1, 0, -1):
            v[i - 1] = min(v[i - 1], np.sqrt(v[i] ** 2 + 2 * a_brk * ds))
        if closed:
            v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2 * a_brk * ds))
    return np.clip(v, v_min, v_max)


# ---------- [12] CSV writer ----------
def write_csv(path, pts, v, heading, kappa):
    pd.DataFrame({
        'x_m': pts[:, 0],
        'y_m': pts[:, 1],
        'vx_mps': v,
        'heading_rad': heading,
        'kappa_radpm': kappa,
    }).to_csv(path, index=False)


# ---------- [13] Visualization ----------
def visualize(map_img, free, dt_img, dt_thresh_px, corridor_raw, corridor_iso,
              skel, centerline_world, raceline, w_l, w_r, kappa, v, res, origin, plot_dir):
    H, W = map_img.shape
    extent = [origin[0], origin[0] + W * res, origin[1], origin[1] + H * res]
    s = np.arange(len(raceline)) * RESAMPLE_DS

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(map_img, cmap='gray', origin='upper', extent=extent)
    axes[0, 0].set_title('Map (free=white)')
    axes[0, 0].axis('equal')

    im = axes[0, 1].imshow(dt_img * res, cmap='hot', origin='upper', extent=extent)
    plt.colorbar(im, ax=axes[0, 1], label='dist to wall (m)')
    # show isolated corridor as cyan contour over the original
    axes[0, 1].contour(np.flipud(corridor_iso.astype(float)), levels=[0.5],
                       colors='cyan', extent=extent)
    axes[0, 1].set_title(f'Distance transform + isolated corridor (cyan)\n'
                         f'kept {corridor_iso.sum()}/{corridor_raw.sum()} px')
    axes[0, 1].axis('equal')

    axes[0, 2].imshow(map_img, cmap='gray', origin='upper', extent=extent)
    sy, sx = np.where(skel)
    axes[0, 2].scatter(origin[0] + sx * res, origin[1] + (H - sy) * res, s=1, c='red')
    axes[0, 2].set_title(f'Pruned skeleton ({skel.sum()} px)')
    axes[0, 2].axis('equal')

    axes[1, 0].imshow(map_img, cmap='gray', origin='upper', extent=extent)
    axes[1, 0].plot(centerline_world[:, 0], centerline_world[:, 1], 'g--', lw=1, label='centerline')
    axes[1, 0].plot(raceline[:, 0], raceline[:, 1], 'r-', lw=1.5, label='raceline')
    axes[1, 0].plot(raceline[0, 0], raceline[0, 1], 'yo', ms=8, label='start')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_title('Centerline vs. raceline')
    axes[1, 0].axis('equal')

    sc = axes[1, 1].scatter(raceline[:, 0], raceline[:, 1], c=v, s=6, cmap='turbo',
                            vmin=V_MIN, vmax=V_MAX)
    plt.colorbar(sc, ax=axes[1, 1], label='v (m/s)')
    axes[1, 1].set_title(f'Velocity ({v.min():.1f}–{v.max():.1f} m/s, mean {v.mean():.2f})')
    axes[1, 1].set_xlabel('x (m)'); axes[1, 1].set_ylabel('y (m)')
    axes[1, 1].axis('equal')

    ax6 = axes[1, 2]
    ax6.plot(s, kappa, 'b-', label=r'$\kappa(s)$')
    ax6.set_ylabel(r'curvature $\kappa$ [1/m]', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6b = ax6.twinx()
    ax6b.plot(s, w_l, 'g-', lw=0.8, label='w_left')
    ax6b.plot(s, -w_r, 'm-', lw=0.8, label='-w_right')
    ax6b.set_ylabel('lateral budget [m]')
    ax6b.legend(loc='upper right')
    ax6.set_xlabel('arc length s [m]')
    ax6.set_title('Curvature & widths')

    plt.tight_layout()
    if SAVE_PLOTS:
        os.makedirs(plot_dir, exist_ok=True)
        out = os.path.join(plot_dir, 'summary.png')
        plt.savefig(out, dpi=DPI)
        print(f'    saved: {out}')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =============================================================================
def main():
    print(f'[1] Load map: {MAP_YAML}')
    map_img, free, res, origin = load_map(MAP_YAML)
    H, W = map_img.shape
    print(f'    image {W}x{H}, res={res} m/px, origin={origin[:2]}, free px={free.sum()}')

    print(f'[2] Safe corridor (erode by {MIN_WALL_DIST_M} m)')
    corridor_raw, dt_img, dt_thresh_px = safe_corridor(free, res, MIN_WALL_DIST_M)
    print(f'    threshold={dt_thresh_px:.1f} px, corridor px={corridor_raw.sum()}')
    if corridor_raw.sum() < MIN_SAFE_PIXELS:
        print(f'    !! corridor too small. Try MIN_WALL_DIST_M ≈ {MIN_WALL_DIST_M*0.5:.2f} m')
        sys.exit(1)

    print(f'[2b] Isolate track from seed {TRACK_SEED_XY} (ISOLATE_TRACK={ISOLATE_TRACK})')
    corridor = isolate_track(corridor_raw, TRACK_SEED_XY, res, origin) if ISOLATE_TRACK else corridor_raw

    print(f'[3] Skeletonize + prune spurs (≤{SPUR_PRUNE_ITERS} iters)')
    skel = skeletonize_and_prune(corridor, SPUR_PRUNE_ITERS)
    print(f'    skeleton px = {skel.sum()}')

    print('[4] Order skeleton into polyline')
    pix_order = order_skeleton_loop(skel, CLOSED_LOOP)
    print(f'    ordered {len(pix_order)} pixels')

    print('[5] Pixel → world')
    centerline = pixels_to_world(pix_order, res, origin, H)

    print(f'[6] Resample @ ds={RESAMPLE_DS} m')
    c, L = resample_arc(centerline, RESAMPLE_DS, CLOSED_LOOP)
    print(f'    {len(c)} points,  L ≈ {L:.1f} m')

    print('[7] Tangents & normals')
    _, n_hat = tangent_normal(c, CLOSED_LOOP)

    print('[8] Per-side widths (ray-march)')
    w_l, w_r = compute_widths(c, n_hat, free, res, origin, MAX_WIDTH_SEARCH, CAR_HALF_WIDTH)
    print(f'    w_left  in [{w_l.min():.2f}, {w_l.max():.2f}] m')
    print(f'    w_right in [{w_r.min():.2f}, {w_r.max():.2f}] m')

    print(f'[9] Min-curvature QP  (lambda={LAMBDA_OFFSET}, {N_REFINE} re-lin passes)')
    alpha = solve_min_curvature(c, n_hat, RESAMPLE_DS, w_l, w_r,
                                CLOSED_LOOP, N_REFINE, LAMBDA_OFFSET)
    raceline = c + n_hat * alpha[:, None]

    print('[10] Spline fit & curvature')
    sp, Lr = fit_spline(raceline, CLOSED_LOOP)
    s_q = np.linspace(0.0, Lr, len(raceline), endpoint=not CLOSED_LOOP)
    pts = sp(s_q)
    kappa, d1 = curvature_from_spline(sp, s_q)
    kmax = float(np.max(np.abs(kappa)))
    print(f'    |kappa|_max = {kmax:.3f} 1/m  (R_min = {1.0/(kmax+1e-9):.2f} m)')

    print('[11] Velocity profile')
    v = velocity_profile(kappa, RESAMPLE_DS, CLOSED_LOOP,
                         V_MAX, V_MIN, V_START, A_LAT_MAX, A_LONG_ACC_MAX, A_LONG_BRK_MAX)
    print(f'    v: [{v.min():.2f}, {v.max():.2f}] m/s, mean {v.mean():.2f}')
    if CLOSED_LOOP:
        print(f'    est. lap time = {np.sum(RESAMPLE_DS / v):.2f} s')

    print(f'[12] Write {OUTPUT_CSV}')
    heading = np.arctan2(d1[:, 1], d1[:, 0])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_csv(OUTPUT_CSV, pts, v, heading, kappa)

    print('[13] Visualize')
    visualize(map_img, free, dt_img, dt_thresh_px, corridor_raw, corridor,
              skel, centerline, pts, w_l, w_r, kappa, v, res, origin, PLOT_DIR)

    print('Done.')


if __name__ == '__main__':
    main()
