#!/usr/bin/env python3
"""
Kinematic bicycle model for F1TENTH vehicle

Reference: ETH Zurich MPCC paper (arXiv:1711.07300v1)
- Section II-A: Vehicle model (Pages 4-5)
- Equation 1: Kinematic bicycle model
"""

import numpy as np


class VehicleModel:
    """
    Kinematic bicycle model
    
    State: x = [X, Y, φ, v]
        X, Y: Position (m)
        φ: Heading angle (rad)
        v: Velocity (m/s)
    
    Input: u = [δ, a]
        δ: Steering angle (rad)
        a: Acceleration (m/s²)
    
    Dynamics: ẋ = f(x, u)
        Ẋ = v·cos(φ)
        Ẏ = v·sin(φ)
        φ̇ = (v/L)·tan(δ)
        v̇ = a
    
    Where L is the wheelbase
    """
    
    def __init__(self, wheelbase=0.33):
        """
        Initialize vehicle model
        
        Args:
            wheelbase: Distance between front and rear axles (meters)
        """
        self.L = wheelbase
    
    def dynamics(self, x, u):
        """
        Compute continuous-time dynamics: ẋ = f(x, u)
        
        Args:
            x: State [X, Y, φ, v]
            u: Input [δ, a]
            
        Returns:
            x_dot: State derivative [Ẋ, Ẏ, φ̇, v̇]
        """
        # Unpack state
        X, Y, phi, v = x
        
        # Unpack input
        delta, a = u
        
        # Compute derivatives (Equation 1)
        X_dot = v * np.cos(phi)
        Y_dot = v * np.sin(phi)
        phi_dot = (v / self.L) * np.tan(delta)
        v_dot = a
        
        return np.array([X_dot, Y_dot, phi_dot, v_dot])
    
    def linearize(self, x_ref, u_ref):
        """
        Linearize dynamics around reference trajectory
        
        Computes Jacobian matrices:
            A = ∂f/∂x at (x_ref, u_ref)
            B = ∂f/∂u at (x_ref, u_ref)
            g = f(x_ref, u_ref) - A·x_ref - B·u_ref
        
        Args:
            x_ref: Reference state [X, Y, φ, v]
            u_ref: Reference input [δ, a]
            
        Returns:
            (A, B, g): Linearized system matrices
        """
        # Unpack reference
        X, Y, phi, v = x_ref
        delta, a = u_ref
        
        # Jacobian A = ∂f/∂x (4x4)
        A = np.array([
            [0, 0, -v * np.sin(phi),  np.cos(phi)],
            [0, 0,  v * np.cos(phi),  np.sin(phi)],
            [0, 0,  0,                np.tan(delta) / self.L],
            [0, 0,  0,                0]
        ])
        
        # Jacobian B = ∂f/∂u (4x2)
        B = np.array([
            [0,                        0],
            [0,                        0],
            [v / (self.L * np.cos(delta)**2),  0],
            [0,                        1]
        ])
        
        # Compute offset: g = f - A·x - B·u
        f_ref = self.dynamics(x_ref, u_ref)
        g = f_ref - A @ x_ref - B @ u_ref
        
        return A, B, g
    
    def discretize(self, A, B, g, dt):
        """
        Discretize linearized system using Euler integration
        
        Continuous: ẋ = A·x + B·u + g
        Discrete: x[k+1] = Ad·x[k] + Bd·u[k] + gd
        
        Where:
            Ad = I + dt·A
            Bd = dt·B
            gd = dt·g
        
        Args:
            A: Continuous state matrix (4x4)
            B: Continuous input matrix (4x2)
            g: Continuous offset (4,)
            dt: Timestep (seconds)
            
        Returns:
            (Ad, Bd, gd): Discrete-time system matrices
        """
        # Euler discretization
        Ad = np.eye(4) + dt * A
        Bd = dt * B
        gd = dt * g
        
        return Ad, Bd, gd
