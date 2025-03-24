import numpy as np
import quaternion
from enum import Enum
from scipy.spatial.transform import Rotation as R


class Dim(Enum):
    """Dimension of a 3D trajectory"""
    X = 0
    Y = 1
    Z = 2
    
class KinematicVars(Enum):
    """Kinematic variables"""
    Position = 0
    Velocity = 1
    Acceleration = 2
    Jerk = 3
    Snap = 4

def get_vector_norm_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def get_vector_norm_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_norm = np.linalg.norm(v)
    if v_norm < 0.001:
        print("Warning: get_vector_norm_derivatives: vector norm too small to get unit vector result")
    v_norm_dot = v@v_dot/v_norm
    v_norm_dot2 = v_dot@v_dot/v_norm + \
        (v@v_dot2)/v_norm - (v@v_dot*v_norm_dot)/v_norm**2
    return (v_norm, v_norm_dot, v_norm_dot2)


def get_unit_vector_derivatives(v: np.ndarray, v_dot: np.ndarray, v_dot2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (v_norm, v_norm_dot, v_norm_dot2) = get_vector_norm_derivatives(v, v_dot, v_dot2)
    b = v/v_norm
    b_dot = v_dot/v_norm - (v*v_norm_dot)/v_norm**2
    b_dot2 = v_dot2/v_norm - (v*v_norm_dot2)/v_norm**2 - \
        (2*v_dot*v_norm_dot)/v_norm**2 + (2*v*v_norm_dot**2)/v_norm**3
    return (b, b_dot, b_dot2)

def get_signal_derivative(t: np.ndarray, v: np.ndarray, dt) -> tuple[np.ndarray, np.ndarray] :
     
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


def get_quaternion_from_angular_displacement(v: np.ndarray) -> quaternion.quaternion:
    """convert a rotation represented by a vector v to a quaternion

    Args:
        v (np.ndarray): the direction of v is the rotation axis, the norm of v is the angle to rotate

    Returns:
        quaternion.quaternion: the quaternion
    """
    theta = np.sqrt(v@v)
    result = np.quaternion()
    result.real = np.cos(0.5*theta)
    result.imag = 0.5*np.sinc(0.5*theta/np.pi)*v
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


def normalize_rotation_matrix(rotation_matrix):
    """make sure the rotation matrix is still represented by 3 unit vectors
    """
    for i in range(rotation_matrix.shape[1]):
        rotation_matrix[:, i] = rotation_matrix[:, i] / \
            np.sqrt(rotation_matrix[:, i]@rotation_matrix[:, i])


def convert_rotation_matrix_to_quaternion(rotation_matrix) -> np.ndarray:
    """convert a rotation matrix to a quaternion
    """
    rotation = R.from_matrix(rotation_matrix)
    q = rotation.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])   # Rearrange from [x, y, z, w] to [w, x, y, z]

def convert_quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion (in [w, x, y, z] format) to a rotation matrix."""
    q_compatible = [q[1], q[2], q[3], q[0]] # Rearrange from [w, x, y, z] to [x, y, z, w]
    rotation = R.from_quat(q_compatible)
    return rotation.as_matrix() 

class FrdFluConverter:
    m_frd_flu = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])  # m_frd_flu.T = m_frd_flu so both direction of conversion use the same matrix
    
    @staticmethod
    def flip(x: np.ndarray):
        """x can be a vector or rotation matrix
        if x is in FRD frame, convert it to FLU frame
        if x is in FLU frame, convert it to FRD frame
        """
        return FrdFluConverter.m_frd_flu@x
    
def convert_rpm_to_radps(rpm):
    return rpm/60*2*np.pi

def convert_radps_to_rpm(radps):
    return radps*60/(2*np.pi)


if __name__ == "__main__":
    v_test = np.array([1, 2, 3])
    m_test = get_hat_map(v_test)
    v_test = np.array([1, 1, 1])
    (b_test, b_dot_test, b_dot2_test) = get_unit_vector_derivatives(
        v_test, 1*v_test, 0.1*v_test)
    print(m_test)
    print(get_vee_map(m_test))
    print((b_test, b_dot_test, b_dot2_test))
