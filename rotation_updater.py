import numpy as np
import quaternion
import drone.utils as utils


class Rotation:
    def __init__(self, rotation_matrix: np.ndarray = np.eye(3)):
        self.rotation_matrix = rotation_matrix

    def step_rotation_matrix(self, omega: np.ndarray, dt: float):
        r_rotated = self.rotation_matrix
        q = utils.get_quaternion_from_angular_displacement(omega*dt)
        for i in range(self.rotation_matrix.shape[1]):
            e_q = utils.convert_vector_to_quaternion(self.rotation_matrix[:, i])
            e_rotated = q*e_q*q.conjugate()
            r_rotated[:, i] = utils.convert_quaternion_to_vector(e_rotated)
        self.rotation_matrix = r_rotated




if __name__ == "__main__":
    r = Rotation()
    omega = np.array([np.pi/6, 0, 0])
    dt = 1
    r.step_rotation_matrix(omega, dt)
    print(r.rotation_matrix)
