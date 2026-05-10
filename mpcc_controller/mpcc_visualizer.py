"""
MPCC Visualization Module - Real-time debugging dashboard
Author: GitHub Copilot
Description:
    Provides matplotlib-based visualization of MPC controller state, tracking errors,
    control inputs, and reference path for real-time debugging and tuning.
    
    Displays:
    - Reference path + vehicle position + heading vector (live update)
    - Tracking errors (contouring, lag, heading) over last 100 steps
    - Control inputs (steering angle delta, throttle tau) history
    - Velocity profile (reference vs actual)
    
    Update frequency: 10 Hz (every 5 control steps) to minimize overhead.
"""

import numpy as np
from collections import deque
import threading

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
    print("[VISUALIZER] ✓ Matplotlib imported successfully")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"[VISUALIZER] ✗ Matplotlib import failed: {e}")
    print("[VISUALIZER] Visualizer will be disabled")


class MPCCVisualizer:
    """
    Real-time visualization dashboard for MPCC debugging.
    """
    
    def __init__(self, update_freq_hz=10, history_length=100, enable_display=True):
        """
        Initialize the visualizer.
        
        Args:
            update_freq_hz: Update frequency in Hz (default: 10)
            history_length: Number of steps to keep in history (default: 100)
            enable_display: Whether to display the plot (default: True)
        """
        self.update_freq_hz = update_freq_hz
        self.history_length = history_length
        self.enable_display = enable_display
        self.is_headless = False  # Will be set to True if in Docker/headless environment
        self.step_counter = 0
        self.update_interval = max(1, int(20 / update_freq_hz))  # Assume 20 Hz control loop
        self.render_count = 0  # Counter for saving plots periodically
        self.save_interval = 100  # Save plot every 100 renders (~20 seconds at 5 Hz)
        
        # Data storage (collections for efficient append/pop)
        self.time_history = deque(maxlen=history_length)
        self.x_history = deque(maxlen=history_length)
        self.y_history = deque(maxlen=history_length)
        self.e_c_history = deque(maxlen=history_length)  # Contouring error
        self.e_l_history = deque(maxlen=history_length)  # Lag error
        self.e_theta_history = deque(maxlen=history_length)  # Heading error
        self.delta_history = deque(maxlen=history_length)  # Steering angle
        self.tau_history = deque(maxlen=history_length)  # Throttle/torque
        self.v_history = deque(maxlen=history_length)  # Actual velocity
        self.v_ref_history = deque(maxlen=history_length)  # Reference velocity
        
        # Reference path (will be set externally)
        self.ref_path_x = None
        self.ref_path_y = None
        self.ref_horizon_x = None
        self.ref_horizon_y = None
        
        # Figure and axes
        self.fig = None
        self.ax_path = None
        self.ax_errors = None
        self.ax_controls = None
        self.ax_velocity = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        if enable_display:
            if not MATPLOTLIB_AVAILABLE:
                print("[VISUALIZER] ✗ Cannot setup figure: matplotlib not available")
                self.enable_display = False
            else:
                print("[VISUALIZER] Setting up matplotlib figure...")
                self._setup_figure()
                print("[VISUALIZER] ✓ Figure setup complete")
    
    def _setup_figure(self):
        """Setup the matplotlib figure with 4 subplots."""
        try:
            # Try to use a display-compatible backend, fallback to file-based if needed
            import matplotlib
            try:
                matplotlib.use('TkAgg')  # Try interactive first
                print("[VISUALIZER] Using TkAgg backend (interactive)")
            except Exception:
                try:
                    matplotlib.use('Qt5Agg')  # Try Qt5
                    print("[VISUALIZER] Using Qt5Agg backend (interactive)")
                except Exception:
                    # Fallback to Agg (non-interactive, file-based)
                    matplotlib.use('Agg')
                    print("[VISUALIZER] Using Agg backend (file-based) - likely Docker environment")
                    self.is_headless = True
        except Exception as e:
            print(f"Warning: Could not set matplotlib backend: {e}")
            self.is_headless = True
        
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('MPCC Controller Diagnostics - Real-time Dashboard', fontsize=14, fontweight='bold')
        
        # Subplot 1: Reference Path + Vehicle Position (top-left)
        self.ax_path = axes[0, 0]
        self.ax_path.set_xlabel('X [m]')
        self.ax_path.set_ylabel('Y [m]')
        self.ax_path.set_title('Reference Path & Vehicle Trajectory')
        self.ax_path.grid(True, alpha=0.3)
        self.ax_path.axis('equal')
        
        # Subplot 2: Tracking Errors (top-right)
        self.ax_errors = axes[0, 1]
        self.ax_errors.set_xlabel('Time Step')
        self.ax_errors.set_ylabel('Error [m] / [rad]')
        self.ax_errors.set_title('Tracking Errors Over Time')
        self.ax_errors.legend(['e_c (contouring) [m]', 'e_l (lag) [m]', 'e_θ (heading) [rad]'], loc='upper left')
        self.ax_errors.grid(True, alpha=0.3)
        
        # Subplot 3: Control Inputs (bottom-left)
        self.ax_controls = axes[1, 0]
        self.ax_controls.set_xlabel('Time Step')
        self.ax_controls.set_ylabel('Steering δ [rad]  /  Throttle τ [Nm]')
        self.ax_controls.set_title('Control Inputs History')
        self.ax_controls.legend(['δ (steering) [rad]', 'τ (throttle) [Nm×0.1]'], loc='upper left')
        self.ax_controls.grid(True, alpha=0.3)
        
        # Subplot 4: Velocity Profile (bottom-right)
        self.ax_velocity = axes[1, 1]
        self.ax_velocity.set_xlabel('Time Step')
        self.ax_velocity.set_ylabel('Velocity [m/s]')
        self.ax_velocity.set_title('Velocity Tracking')
        self.ax_velocity.legend(['V (actual)', 'V_ref (target)'], loc='upper left')
        self.ax_velocity.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def set_reference_path(self, path_x, path_y):
        """
        Set the reference path for visualization.
        
        Args:
            path_x: Array of x-coordinates
            path_y: Array of y-coordinates
        """
        with self.lock:
            self.ref_path_x = np.array(path_x)
            self.ref_path_y = np.array(path_y)
    
    def update(self, state, reference, errors, control, v_actual, v_ref, 
               ref_horizon=None, mpcc_pred_traj=None):
        """
        Update the visualizer with new data.
        
        Args:
            state: Robot state [x, y, theta, ...]
            reference: Reference state [x_ref, y_ref, theta_ref]
            errors: Dict with 'e_c', 'e_l', 'e_theta'
            control: Control input [delta, tau, ...]
            v_actual: Actual velocity
            v_ref: Reference velocity
            ref_horizon: Reference horizon for MPC lookahead (optional)
            mpcc_pred_traj: MPCC predicted trajectory (optional)
        """
        # Update counter and check if we should visualize this step
        self.step_counter += 1
        if self.step_counter % self.update_interval != 0:
            return
        
        with self.lock:
            # Extract state
            x, y = state[0], state[1]
            self.x_history.append(x)
            self.y_history.append(y)
            
            # Store errors
            self.e_c_history.append(errors.get('e_c', 0.0))
            self.e_l_history.append(errors.get('e_l', 0.0))
            self.e_theta_history.append(errors.get('e_theta', 0.0))
            
            # Store control inputs (scale tau for visibility)
            self.delta_history.append(control[0] if len(control) > 0 else 0.0)
            self.tau_history.append((control[1] * 0.1) if len(control) > 1 else 0.0)  # Scale by 0.1 for display
            
            # Store velocity
            self.v_history.append(v_actual)
            self.v_ref_history.append(v_ref)
            
            # Store reference horizon if provided
            if ref_horizon is not None and ref_horizon.shape[1] > 0:
                self.ref_horizon_x = ref_horizon[0, :]
                self.ref_horizon_y = ref_horizon[1, :]
            
            self.time_history.append(len(self.time_history))
    
    def render(self):
        """Render the current visualization (call this in the main loop or use animation)."""
        if not self.enable_display or self.fig is None:
            return
        
        with self.lock:
            # Clear previous plots
            self.ax_path.clear()
            self.ax_errors.clear()
            self.ax_controls.clear()
            self.ax_velocity.clear()
            
            # === Subplot 1: Path + Trajectory ===
            if self.ref_path_x is not None:
                self.ax_path.plot(self.ref_path_x, self.ref_path_y, 'b-', linewidth=2, label='Reference Path', alpha=0.7)
            
            if len(self.x_history) > 1:
                self.ax_path.plot(list(self.x_history), list(self.y_history), 'r-', linewidth=1.5, label='Vehicle Trajectory', alpha=0.8)
                # Plot current position
                self.ax_path.plot(self.x_history[-1], self.y_history[-1], 'ro', markersize=8, label='Current Position')
            
            # Plot MPC predicted trajectory if available
            if self.ref_horizon_x is not None and len(self.ref_horizon_x) > 0:
                self.ax_path.plot(self.ref_horizon_x, self.ref_horizon_y, 'g--', linewidth=1, 
                                 label='MPC Lookahead', alpha=0.6, marker='x')
            
            self.ax_path.set_xlabel('X [m]')
            self.ax_path.set_ylabel('Y [m]')
            self.ax_path.set_title('Reference Path & Vehicle Trajectory')
            self.ax_path.grid(True, alpha=0.3)
            self.ax_path.legend(loc='upper left', fontsize=8)
            self.ax_path.axis('equal')
            
            # === Subplot 2: Tracking Errors ===
            if len(self.time_history) > 0:
                time_axis = list(self.time_history)
                if len(self.e_c_history) > 0:
                    self.ax_errors.plot(time_axis[-len(self.e_c_history):], list(self.e_c_history), 
                                       'b-', linewidth=1.5, label='e_c (contouring) [m]', alpha=0.8)
                if len(self.e_l_history) > 0:
                    self.ax_errors.plot(time_axis[-len(self.e_l_history):], list(self.e_l_history), 
                                       'g-', linewidth=1.5, label='e_l (lag) [m]', alpha=0.8)
                if len(self.e_theta_history) > 0:
                    self.ax_errors.plot(time_axis[-len(self.e_theta_history):], list(self.e_theta_history), 
                                       'r-', linewidth=1.5, label='e_θ (heading) [rad]', alpha=0.8)
                
                # Add error bounds (safety threshold)
                self.ax_errors.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='±0.5m bound')
                self.ax_errors.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            
            self.ax_errors.set_xlabel('Time Step')
            self.ax_errors.set_ylabel('Error')
            self.ax_errors.set_title('Tracking Errors Over Time (keep |e_c|, |e_l| < 0.5m)')
            self.ax_errors.grid(True, alpha=0.3)
            self.ax_errors.legend(loc='upper left', fontsize=8)
            
            # === Subplot 3: Control Inputs ===
            if len(self.time_history) > 0:
                time_axis = list(self.time_history)
                if len(self.delta_history) > 0:
                    self.ax_controls.plot(time_axis[-len(self.delta_history):], list(self.delta_history), 
                                         'b-', linewidth=1, label='δ (steering) [rad]', alpha=0.8)
                if len(self.tau_history) > 0:
                    self.ax_controls.plot(time_axis[-len(self.tau_history):], list(self.tau_history), 
                                         'r-', linewidth=1, label='τ (throttle ×0.1) [Nm]', alpha=0.8)
            
            self.ax_controls.set_xlabel('Time Step')
            self.ax_controls.set_ylabel('Control Value')
            self.ax_controls.set_title('Control Inputs History')
            self.ax_controls.grid(True, alpha=0.3)
            self.ax_controls.legend(loc='upper left', fontsize=8)
            
            # === Subplot 4: Velocity ===
            if len(self.time_history) > 0:
                time_axis = list(self.time_history)
                if len(self.v_history) > 0:
                    self.ax_velocity.plot(time_axis[-len(self.v_history):], list(self.v_history), 
                                         'b-', linewidth=1.5, label='V (actual)', alpha=0.8)
                if len(self.v_ref_history) > 0:
                    self.ax_velocity.plot(time_axis[-len(self.v_ref_history):], list(self.v_ref_history), 
                                         'r--', linewidth=1.5, label='V_ref (target)', alpha=0.8)
            
            self.ax_velocity.set_xlabel('Time Step')
            self.ax_velocity.set_ylabel('Velocity [m/s]')
            self.ax_velocity.set_title('Velocity Tracking')
            self.ax_velocity.grid(True, alpha=0.3)
            self.ax_velocity.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        try:
            if self.is_headless:
                # Save to file instead of displaying
                self.render_count += 1
                if self.render_count % self.save_interval == 0:
                    filename = f"/tmp/mpcc_dashboard_{self.render_count}.png"
                    plt.savefig(filename, dpi=80, bbox_inches='tight')
                    print(f"[VISUALIZER] Saved plot to {filename}")
            else:
                # Interactive display
                plt.pause(0.001)  # Non-blocking display
        except Exception as e:
            print(f"[VISUALIZER] Warning: render failed: {e}")
    
    def get_statistics(self):
        """
        Get statistical summary of tracking performance.
        
        Returns:
            dict with mean/max errors and control values
        """
        stats = {}
        if len(self.e_c_history) > 0:
            e_c_array = np.array(list(self.e_c_history))
            stats['e_c_mean'] = np.mean(np.abs(e_c_array))
            stats['e_c_max'] = np.max(np.abs(e_c_array))
        
        if len(self.e_l_history) > 0:
            e_l_array = np.array(list(self.e_l_history))
            stats['e_l_mean'] = np.mean(np.abs(e_l_array))
            stats['e_l_max'] = np.max(np.abs(e_l_array))
        
        if len(self.e_theta_history) > 0:
            e_theta_array = np.array(list(self.e_theta_history))
            stats['e_theta_mean'] = np.mean(np.abs(e_theta_array))
            stats['e_theta_max'] = np.max(np.abs(e_theta_array))
        
        if len(self.v_history) > 0 and len(self.v_ref_history) > 0:
            v_error = np.array(list(self.v_history)) - np.array(list(self.v_ref_history))
            stats['v_error_mean'] = np.mean(np.abs(v_error))
            stats['v_error_max'] = np.max(np.abs(v_error))
        
        return stats
    
    def print_statistics(self):
        """Print statistics to console."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("MPCC TRACKING STATISTICS")
        print("="*60)
        if 'e_c_mean' in stats:
            print(f"Contouring Error (e_c):  mean={stats['e_c_mean']:.4f} m,  max={stats['e_c_max']:.4f} m")
        if 'e_l_mean' in stats:
            print(f"Lag Error (e_l):         mean={stats['e_l_mean']:.4f} m,  max={stats['e_l_max']:.4f} m")
        if 'e_theta_mean' in stats:
            print(f"Heading Error (e_θ):     mean={stats['e_theta_mean']:.4f} rad, max={stats['e_theta_max']:.4f} rad")
        if 'v_error_mean' in stats:
            print(f"Velocity Error:          mean={stats['v_error_mean']:.4f} m/s, max={stats['v_error_max']:.4f} m/s")
        print("="*60)
    
    def export_to_csv(self, filename="/tmp/mpcc_data.csv"):
        """
        Export all data to CSV for post-analysis (useful in Docker).
        
        Args:
            filename: Output CSV file path
        """
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['step', 'x', 'y', 'e_c', 'e_l', 'e_theta', 'delta', 'tau', 'v', 'v_ref'])
                
                # Data rows
                max_len = max(len(self.time_history), 1)
                for i in range(max_len):
                    row = [
                        i if i < len(self.time_history) else i,
                        self.x_history[i] if i < len(self.x_history) else 0,
                        self.y_history[i] if i < len(self.y_history) else 0,
                        self.e_c_history[i] if i < len(self.e_c_history) else 0,
                        self.e_l_history[i] if i < len(self.e_l_history) else 0,
                        self.e_theta_history[i] if i < len(self.e_theta_history) else 0,
                        self.delta_history[i] if i < len(self.delta_history) else 0,
                        self.tau_history[i] if i < len(self.tau_history) else 0,
                        self.v_history[i] if i < len(self.v_history) else 0,
                        self.v_ref_history[i] if i < len(self.v_ref_history) else 0,
                    ]
                    writer.writerow(row)
            print(f"[VISUALIZER] Exported data to {filename}")
        except Exception as e:
            print(f"[VISUALIZER] Error exporting to CSV: {e}")
