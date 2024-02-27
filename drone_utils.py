import numpy as np
import quaternion


def get_vector_norm_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    pass


def get_hat_map(v: np.ndarray) -> np.ndarray:
    m = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return m


def get_vee_map(m: np.ndarray) -> np.ndarray:
    v = np.array([0.5*(-m[1, 2] + m[2, 1]),
                  0.5*(m[0, 2] - m[2, 0]),
                  0.5*(-m[0, 1] + m[1, 0])])
    return v


def get_vector_norm_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    v_norm = np.linalg.norm(v)
    if v_norm < 0.001:
        print("Warning: get_vector_norm_derivatives: vector norm too small to get unit vector result")
    v_norm_dot = v@v_dot/v_norm
    v_norm_dot2 = v_dot@v_dot/v_norm + \
        (v@v_dot2)/v_norm - (v@v_dot*v_norm_dot)/v_norm**2
    return (v_norm, v_norm_dot, v_norm_dot2)


def get_unit_vector_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    (v_norm, v_norm_dot, v_norm_dot2) = get_vector_norm_derivatives(v, v_dot, v_dot2)
    b = v/v_norm
    b_dot = v_dot/v_norm - (v*v_norm_dot)/v_norm**2
    b_dot2 = v_dot2/v_norm - (v*v_norm_dot2)/v_norm**2 - \
        (2*v_dot*v_norm_dot)/v_norm**2 + (2*v*v_norm_dot**2)/v_norm**3
    return (b, b_dot, b_dot2)

def get_signal_derivative(t: np.ndarray, v: np.ndarray, dt) -> (np.ndarray, np.ndarray) :
     
    t_dv = t[1:]
    dim = np.ndim(v)
    if 1 == dim:
        dv = (v[1:] - v[:-1])/dt
    elif 2 == dim:
        dv = (v[1:, :] - v[:-1, :])/dt
    elif 3 == dim:
        dv = (v[1:, :, :] - v[:-1, :, :])/dt
    else:
        dv = t_dv - t_dv
        print("Warning: get_signal_derivative: input dimension not in range, which is ", dim)
    return t_dv, dv


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
    if quat.real > 0.000001:
        print(
            'Warning: rotation_updater: quaternion to vector conversion, non-zero quaternion real part is', quat.real)
    return result


def normalize_rotation_matrix(self):
    for i in range(self.rotation_matrix.shape[1]):
        self.rotation_matrix[:, i] = self.rotation_matrix[:, i] / \
            np.sqrt(self.rotation_matrix[:, i]@self.rotation_matrix[:, i])


if __name__ == "__main__":
    v_test = np.array([1, 2, 3])
    m_test = get_hat_map(v_test)
    v_test = np.array([1, 1, 1])
    (b_test, b_dot_test, b_dot2_test) = get_unit_vector_derivatives(
        v_test, 1*v_test, 0.1*v_test)
    print(m_test)
    print(get_vee_map(m_test))
    print((b_test, b_dot_test, b_dot2_test))
