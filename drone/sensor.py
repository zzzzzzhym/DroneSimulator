import numpy as np

import dynamics
import dynamics_state
import imu_model

class Sensor:
    """Converts ground-truth dynamics into simulated sensor outputs."""
    def __init__(self):
        self.imu_model = imu_model.ImuModel()

    def get_sensor_data(self, state: dynamics.DroneDynamics, t: float):
        """Get all sensor data"""
        return SensorData(
            position=state.state.get_position_in_inertial_frame(),
            v=state.state.get_velocity_in_inertial_frame(),
            pose=state.state.get_pose_in_inertial_frame(),
            omega=self.imu_model.create_noisified_omega(state.state.get_omega_in_body_frame(), t),
            v_dot=self.imu_model.create_noisified_accel(state.v_dot, t),
            rotors=state.rotors,
            omega_dot=state.omega_dot
        )
    
class SensorData(dynamics_state.State):
    """Container for all sensor data, follow FRD convention."""
    def __init__(self, 
                 position: np.ndarray, 
                 v: np.ndarray,
                 pose: np.ndarray, 
                 omega: np.ndarray,
                 v_dot: np.ndarray,
                 rotors: dynamics.rotor.RotorSet,
                 omega_dot: np.ndarray) -> None:
        super().__init__(position, v, pose, omega)
        self.v_dot = v_dot
        self.rotors = rotors
        self.omega_dot = omega_dot