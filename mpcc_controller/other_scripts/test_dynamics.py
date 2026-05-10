import numpy as np
from vehicle_model import VehicleModel

def test_linearization():
    vm = VehicleModel(wheelbase=0.33)
    dt = 0.05
    
    # Test cases spanning typical operating conditions
    test_cases = [
        # (state, input, description)
        (np.array([0.0, 0.0, 0.0, 0.0]),    np.array([0.0, 0.0]),  "Stationary, no input"),
        (np.array([0.0, 0.0, 0.0, 1.0]),    np.array([0.0, 0.0]),  "Moving straight, no input"),
        (np.array([0.0, 0.0, 0.0, 1.0]),    np.array([0.1, 0.0]),  "Straight, small steer"),
        (np.array([0.0, 0.0, 0.0, 1.0]),    np.array([0.0, 1.0]),  "Straight, accelerating"),
        (np.array([0.0, 0.0, 0.5, 2.0]),    np.array([0.3, 0.5]),  "Turning + accelerating"),
        (np.array([0.0, 0.0, -0.5, 3.0]),   np.array([-0.2, 0.0]), "Negative heading, right steer"),
    ]
    
    print(f"{'Test':<35} {'max |err|':<12} {'status'}")
    print("-" * 60)
    
    all_passed = True
    for x0, u0, desc in test_cases:
        # True nonlinear dynamics via Euler step
        x_dot_true = vm.dynamics(x0, u0)
        x_next_true = x0 + dt * x_dot_true
        
        # Linearized prediction (what the solver uses)
        A, B, g = vm.linearize(x0, u0)
        Ad, Bd, gd = vm.discretize(A, B, g, dt)
        x_next_lin = Ad @ x0 + Bd @ u0 + gd
        
        err = np.max(np.abs(x_next_true - x_next_lin))
        status = "✓" if err < 1e-9 else "✗ FAIL"
        if err > 1e-9: all_passed = False
        print(f"{desc:<35} {err:<12.2e} {status}")
    
    print("-" * 60)
    print(f"{'ALL PASSED' if all_passed else 'SOMETHING FAILED'}")

if __name__ == '__main__':
    test_linearization()