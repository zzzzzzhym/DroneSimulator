import numpy as np
import rotation_updater
import drone_parameters as params
import drone_utils as utils


class DroneDynamics:
    def __init__(self, position: np.ndarray, v: np.ndarray,
                 pose: np.ndarray, omega: np.ndarray, dt: float = 0.01) -> None:
        """
        pose is a 3x3 rotation matrix from body to inertial frame
        omega is in body fix frame
        """
        self.position = position    # in inertial frame
        self.pose = pose
        self.v = v  # in inertial frame
        self.omega = omega  # omega in body fix frame
        self.dt = dt

        self.v_dot = np.array([0.0, 0.0, 0.0])
        self.omega_dot = np.array([0.0, 0.0, 0.0])  # in body fix frame

        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force opposite to self.pose[:,2]
        self.torque = np.array([0.0, 0.0, 0.0])

    def step_position(self) -> None:
        self.position += self.dt*self.v

    def step_pose(self) -> None:
        rotation_instance = rotation_updater.Rotation(self.pose)
        omega_in_iniertial_frame = self.pose@self.omega
        rotation_instance.step_rotation_matrix(
            omega_in_iniertial_frame, self.dt)
        self.pose = rotation_instance.rotation_matrix

    def step_derivatives(self):
        self.v_dot = params.g*np.array([0.0, 0.0, 1.0]) - \
            self.pose@self.f/params.m
        self.omega_dot = params.inertia_inv@(self.torque -
                                          utils.get_hat_map(self.omega)@params.inertia@self.omega)
        self.v += self.v_dot*self.dt
        """
        to do: rotation matrix is not constant, improve omega accuracy
        """
        self.omega += self.omega_dot*self.dt

    def step_dynamics(self) -> None:
        self.step_derivatives()
        self.step_position()
        self.step_pose()


if __name__ == "__main__":
    x = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 2.0, 3.0])
    pose1 = np.identity(3)
    omega1 = np.array([np.pi/6, 0.0, 0.0])
    dt1 = 1.0
    kinematics = DroneDynamics(x, v1, pose1, omega1, dt1)
    kinematics.step_dynamics()
    pose_answer = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                       [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
    position_answer = np.array([v1[0]*dt1, v1[1]*dt1, v1[2]*dt1 + 0.5*params.g*dt1*dt1])
    print(kinematics.pose - pose_answer)
    print(kinematics.position - position_answer)
