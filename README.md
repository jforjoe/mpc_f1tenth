# kinematic_mpc — F1TENTH MPC Controller

Receding-horizon kinematic MPC built on CasADi + IPOPT.  
All tunable parameters live at the top of `kinematic_mpc/mpc_node.py`.

---

## Workflow overview

```
1. Generate raceline  →  2. Check direction  →  3. Run simulator  →  4. Run MPC
   plan_trajectory.py       summary.png /           gym_bridge            mpc_node
                            signed-area check        launch file
```

---

## Step 1 — Generate the raceline

Edit the two paths at the top of `scripts/plan_trajectory.py`:

```python
MAP_YAML   = '/sim_ws/src/f1tenth_gym_ros/maps/map.yaml'   # your map
OUTPUT_CSV = '/sim_ws/src/mpc_f1tenth/config/raceline.csv'  # output
```

Key tuning knobs (also near the top of the script):

| Parameter | Default | Effect |
|---|---|---|
| `MIN_WALL_DIST_M` | 0.30 m | Safety margin from walls. Increase if raceline clips walls. |
| `TRACK_SEED_XY` | (0.0, 0.0) | A world-coordinate point **on the track** (use your spawn pose). |
| `RESAMPLE_DS` | 0.10 m | Waypoint spacing. Smaller = smoother path, larger CSV. |
| `V_MAX / V_MIN` | 8.0 / 1.5 m/s | Speed limits for the velocity profile. |

Then run it:

```bash
cd /sim_ws
python3 src/mpc_f1tenth/scripts/plan_trajectory.py
```

This writes `config/raceline.csv` and saves diagnostic plots to `config/plots/summary.png`.

**Check `summary.png`** — the bottom-left panel shows the final raceline overlaid on the map with a red dot marking the start point. Make sure:
- The raceline stays inside the track walls
- The red start dot is near your spawn pose (`sx, sy` in `sim.yaml`)
- The path closes cleanly (no kinks at the loop join)

---

## Step 2 — Visualize waypoint direction in RViz

Use the **`waypoint_visualizer.py`** script in the `mpc_f1tenth/maps/` directory to visualize the waypoints and understand their direction in RViz.

**Run the visualizer:**

```bash
ros2 run kinematic_mpc waypoint_visualizer
```

This publishes markers to `/mpc/waypoints` that you can view in RViz:
- **Green line** — full raceline path
- **Arrows/direction indicators** — show the direction the path is traced

**In RViz:**
1. Add a `Marker` display and subscribe to `/mpc/waypoints`
2. Observe the direction arrows along the track
3. Compare with your spawn pose and driving direction (defined in `sim.yaml` by `stheta`)

**How to choose direction:**

- If arrows point **the same direction** the car should drive from spawn → `WP_REVERSE = False`
- If arrows point **opposite** to the desired driving direction → `WP_REVERSE = True`

**Set in `mpc_node.py`:**

```python
WP_REVERSE = False   # CSV order matches driving direction
# WP_REVERSE = True  # CSV is backwards — reverse it
```

---

## Step 3 — Run the simulator

```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

The spawn pose is set by `sx / sy / stheta` in `config/sim.yaml`.  
If you change the spawn, re-run the direction check (Step 2) since the nearest waypoint changes.

---

## Step 4 — Run the MPC node

```bash
ros2 run kinematic_mpc kinematic_mpc_node
```

or via the launch file (also starts a particle filter):

```bash
ros2 launch kinematic_mpc sim_with_pf.launch.py
```

**Key parameters in `mpc_node.py`:**

```python
ODOM_TOPIC   = '/ego_racecar/odom'  # sim ground truth (absolute map position)
                                     # swap to '/odometry/filtered' for hardware
TARGET_SPEED = 2.0                  # cap on reference velocity (m/s)
                                     # raceline CSV has 2–8 m/s; start low, raise when stable
N            = 15                   # MPC horizon steps (N × DT = look-ahead time)
DT           = 0.05                 # timestep (s)
```

---

## How the MPC gets its position

In simulation, `/ego_racecar/odom` has `frame_id = 'map'` and carries the **absolute ground-truth position** directly from the simulator physics. There is no drift.

On hardware, wheel-encoder odometry drifts over time. The position must come from a localization system (AMCL + EKF → `/odometry/filtered`, or a particle filter → `/pf/pose/odom`). Switch `ODOM_TOPIC` accordingly.

---

## Visualising the trajectory in RViz

While `mpc_node.py` is running, add these topics in RViz:

| Topic | Type | Colour | What it shows |
|---|---|---|---|
| `/mpc/raceline` | `Marker` | Green | Full optimised raceline |
| `/mpc/horizon` | `Marker` | Blue | MPC predicted trajectory |
| `/mpc/ref_horizon` | `Marker` | Yellow | Reference horizon sent to solver |

If `/mpc/raceline` does not appear, set `PUBLISH_MARKERS = True` in the node (only available in `mpc_node_hardware_test.py` by default).

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Car immediately turns hard and hits a wall | `WP_REVERSE` wrong — 180° heading mismatch at spawn | Run the direction check (Step 2) |
| `MPC solver failed Nx; using fallback` | Speed lower bound `MIN_SPEED > 0` while car is stationary → infeasible initial state | Keep `MIN_SPEED = 0.0` in `mpc_node.py` |
| Car follows track but overshoots corners | `TARGET_SPEED` too high or `Q_YAW` too low | Lower `TARGET_SPEED` first; if still bad, increase `Q_YAW` |
| Raceline clips a wall in `summary.png` | `MIN_WALL_DIST_M` too small | Increase to 0.35–0.40 m and re-run planner |
| Skeleton skips part of the track | `TRACK_SEED_XY` is outside the isolated loop | Set it to a point clearly on the driveable surface |
