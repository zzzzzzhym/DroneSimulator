import numpy as np
import drone_parameters as params

class DisturbanceForce:
    """Generate a force in inertial frame"""
    def initialize_force(t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        result = const_force(params.m_payload)
        return result

    def get_force(t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        """API of disturbance force

        Args:
            t (float): time for disturbance force explicitly dependent on time
            state (np.ndarray): np.array([  position[0],  # 0
                                            position[1],  # 1
                                            position[2],  # 2
                                            v[0],         # 3
                                            v[1],         # 4
                                            v[2],         # 5
                                            q[0],         # 6
                                            q[1],         # 7
                                            q[2],         # 8
                                            q[3],         # 9
                                            omega[0],     # 10
                                            omega[1],     # 11
                                            omega[2]])    # 12
        Returns:
            np.ndarray: (3,) array to represent (f_x, f_y, f_z)
        """
        result = const_force(params.m_payload)
        return result
    
    def get_force_derivatives(t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        """API of disturbance derivatives

        Args:
            t (float): time for disturbance force explicitly dependent on time
            state (np.ndarray): np.array([  position[0],  # 0
                                            position[1],  # 1
                                            position[2],  # 2
                                            v[0],         # 3
                                            v[1],         # 4
                                            v[2],         # 5
                                            q[0],         # 6
                                            q[1],         # 7
                                            q[2],         # 8
                                            q[3],         # 9
                                            omega[0],     # 10
                                            omega[1],     # 11
                                            omega[2]])    # 12
        Returns:
            np.ndarray: (3,) array to represent (f_x, f_y, f_z)
        """
        result = np.zeros(3)
        return result

def const_force(weight):
    # typical payload of a drone ranges from 0.2kg to 1kg
    # of course, to properly simulate payload, we should change mass instead
    return np.array([0, 0, params.g*weight])
