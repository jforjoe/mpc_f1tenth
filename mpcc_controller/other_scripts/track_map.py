#!/usr/bin/env python3
"""
Track representation using cubic spline interpolation and arc-length parameterization

Reference: ETH Zurich MPCC paper (arXiv:1711.07300v1)
- Section II-C: Track representation (Pages 7-8)
- Equation 7: Arc-length parameterization
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar


class TrackMap:
    """
    Track representation with centerline, boundaries, and reference computation
    
    Attributes:
        tck: Spline coefficients (from splprep)
        L: Total track length in meters (arc-length)
        theta_to_u: Function mapping arc-length θ to spline parameter u
        track_width: Track width in meters
    """
    
    def __init__(self, waypoints, track_width=1.0):
        """
        Initialize track from waypoints
        
        Args:
            waypoints: Nx2 or Nx3 array [[x, y], ...] or [[x, y, v], ...]
            track_width: Track width in meters
        """
        # Extract x, y coordinates
        waypoints_xy = waypoints[:, :2] if waypoints.shape[1] >= 2 else waypoints
        
        # Fit cubic spline (periodic for closed loop)
        self.tck, u = splprep([waypoints_xy[:, 0], waypoints_xy[:, 1]], 
                               s=0, k=3, per=True)
        
        # Compute arc-length parameterization
        # Sample spline densely
        u_fine = np.linspace(0, 1, 1000)
        points = np.array(splev(u_fine, self.tck)).T
        
        # Compute arc-length: s = ∫√(dx² + dy²) du
        dx_du = np.gradient(points[:, 0], u_fine)  # ← ADD u_fine parameter!
        dy_du = np.gradient(points[:, 1], u_fine)  # ← ADD u_fine parameter!
        ds = np.sqrt(dx_du**2 + dy_du**2)

        # Cumulative arc-length using trapezoidal integration
        s_cumsum = np.insert(cumulative_trapezoid(ds, u_fine), 0, 0)
        self.L = s_cumsum[-1]
        
        # Create mapping: θ (meters) → u (spline parameter)
        from scipy.interpolate import interp1d
        self.theta_to_u = interp1d(s_cumsum, u_fine, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        
        self.track_width = track_width
        
    def get_reference(self, theta):
        """
        Get reference point on centerline at progress θ
        
        Args:
            theta: Arc-length progress (meters)
            
        Returns:
            (X_ref, Y_ref, Phi): Position and heading on centerline
        """
        # Wrap theta to [0, L]
        theta = np.mod(theta, self.L)
        
        # Convert to spline parameter
        u = float(self.theta_to_u(theta))
        
        # Evaluate position
        point = splev(u, self.tck, der=0)
        X_ref = float(point[0])
        Y_ref = float(point[1])
        
        # Evaluate tangent (first derivative)
        deriv = splev(u, self.tck, der=1)
        Phi = np.arctan2(float(deriv[1]), float(deriv[0]))
        
        return X_ref, Y_ref, Phi
    
    def project_point(self, X, Y):
        """
        Project point (X, Y) onto centerline to find progress θ
        
        Args:
            X, Y: Point coordinates in world frame
            
        Returns:
            theta: Arc-length progress (meters) of closest point on centerline
        """
        # STAGE 1: Coarse global search
        # Sample every ~1 meter along entire track
        n_samples = max(500, int(self.L / 0.2))  # ~1700 samples for 337m track
        theta_samples = np.linspace(0, self.L, n_samples, endpoint=False)

        # Vectorized evaluation of all reference points at once.
        u_samples = self.theta_to_u(theta_samples)
        pts = np.array(splev(u_samples, self.tck, der=0))  # shape (2, n_samples)
        X_refs = pts[0]
        Y_refs = pts[1]
        distances_sq = (X - X_refs) ** 2 + (Y - Y_refs) ** 2

        # Find coarse minimum (globally)
        min_idx = int(np.argmin(distances_sq))
        theta_coarse = float(theta_samples[min_idx])
        # Result: θ ≈ 280m
        
        # STAGE 2: Fine local refinement
        # Search ±2m around the coarse minimum
        search_radius = 2.0
        theta_min = theta_coarse - 2.0  # 278m
        theta_max = theta_coarse + 2.0  # 282m

        def distance_squared(theta):
            X_ref, Y_ref, _ = self.get_reference(theta)
            return (X - X_ref)**2 + (Y - Y_ref)**2
        
        # Find minimum distance
        result = minimize_scalar(distance_squared, 
                              bounds=(theta_min, theta_max),
                              method='bounded')
        theta_refined = np.mod(result.x, self.L)
        
        return theta_refined
        
    
    def compute_errors(self, X, Y, theta):
        """
        Compute contouring and lag errors
        
        Reference: Equation 11a-11b (Page 11)
        
        Args:
            X, Y: Current position
            theta: Current progress on track
            
        Returns:
            (e_c, e_l): Contouring error (perpendicular), lag error (tangential)
        """
        # Get reference point
        X_ref, Y_ref, Phi = self.get_reference(theta)
        
        # Position error
        dx = X - X_ref
        dy = Y - Y_ref
        
        # Contouring error: perpendicular to track
        # e_c = sin(Φ)·dx - cos(Φ)·dy
        e_c = np.sin(Phi) * dx - np.cos(Phi) * dy
        
        # Lag error: along track direction
        # e_l = -cos(Φ)·dx - sin(Φ)·dy
        e_l = -np.cos(Phi) * dx - np.sin(Phi) * dy
        
        return e_c, e_l

    def get_halfspace_constraints(self, theta):
        """
        Get track boundary constraints as halfspace inequalities
        
        The constraint is: stay within ±half_w perpendicular to centerline
        
        Format: F · [X, Y]' - s ≤ f
        Where s is slack variable
        
        Args:
            theta: Arc-length progress
            
        Returns:
            (F, f): Constraint matrices where F·[X,Y]' - s ≤ f
        """
        # Get reference point and heading
        X_ref, Y_ref, Phi = self.get_reference(theta)
        
        # Normal vector (perpendicular to track, pointing LEFT)
        n_x = -np.sin(Phi)
        n_y = np.cos(Phi)
        
        # Half track width
        half_w = self.track_width / 2.0
        
        # The signed distance from centerline is:
        # d = n_x·(X - X_ref) + n_y·(Y - Y_ref)
        #
        # We want: -half_w ≤ d ≤ half_w
        #
        # Rearranging:
        #   d ≤ half_w  →  n_x·X + n_y·Y ≤ half_w + n_x·X_ref + n_y·Y_ref
        #  -d ≤ half_w  → -n_x·X - n_y·Y ≤ half_w - n_x·X_ref - n_y·Y_ref
        
        F = np.array([
            [n_x, n_y],      # Left boundary
            [-n_x, -n_y]     # Right boundary
        ])
        
        f = np.array([
            half_w + n_x * X_ref + n_y * Y_ref,    # Left: d ≤ +half_w
            half_w - n_x * X_ref - n_y * Y_ref     # Right: -d ≤ +half_w
        ])
        
        return F, f
