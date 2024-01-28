import numpy as np
import quaternion


class Rotation:
    def __init__(self, rotation_matrix: np.ndarray = np.identity(3)):
        self.rotation_matrix = rotation_matrix

    def step_rotation_matrix(self, omega: np.ndarray, dt: float):
        r_rotated = self.rotation_matrix
        q = get_quaternion_from_angular_displacement(omega*dt)
        for i in range(self.rotation_matrix.shape[1]):
            e_q = convert_vector_to_quaternion(self.rotation_matrix[:, i])
            e_rotated = q*e_q*q.conjugate()
            r_rotated[:, i] = convert_quaternion_to_vector(e_rotated)
        self.rotation_matrix = r_rotated


def get_quaternion_from_angular_displacement(angle_3d: np.ndarray) -> quaternion.quaternion:
    theta = np.sqrt(angle_3d@angle_3d)
    result = np.quaternion()
    result.real = np.cos(0.5*theta)
    result.imag = 0.5*np.sinc(0.5*theta/np.pi)*angle_3d
    return result


def convert_vector_to_quaternion(vector_3d: np.ndarray) -> quaternion.quaternion:
    result = np.quaternion()
    result.imag = vector_3d
    return result


def convert_quaternion_to_vector(quat: quaternion.quaternion) -> np.ndarray:
    result = quat.imag
    if quat.real != 0.0:
        print(
            'Warning: rotation_updater: quaternion to vector conversion, non-zero quaternion real part is', quat.real)
    return result


def normalize_rotation_matrix(self):
    for i in range(self.rotation_matrix.shape[1]):
        self.rotation_matrix[:, i] = self.rotation_matrix[:, i] / \
            np.sqrt(self.rotation_matrix[:, i]@self.rotation_matrix[:, i])


if __name__ == "__main__":
    r = Rotation()
    omega = np.array([np.pi/6, 0, 0])
    dt = 1
    r.step_rotation_matrix(omega, dt)
    print(r.rotation_matrix)
