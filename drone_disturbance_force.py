import numpy as np
import drone_parameters as params

class DisturbanceForce:
    """Generate a force in inertial frame"""
    def __init__(self) -> None:
        self.f_implicit = np.zeros(3)
        self.f_explicit = np.zeros(3)

    def initialize_implicit_force(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        raise NotImplementedError("This function need to be implemented by subclasses")

    def update_explicit_force(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> None:
        """API of explicit disturbance force

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
        raise NotImplementedError("This function need to be implemented by subclasses")
    
    def get_implicit_force_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        """API of implicit disturbance derivatives

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
        raise NotImplementedError("This function need to be implemented by subclasses")


def const_force(weight):
    # typical payload of a drone ranges from 0.2kg to 1kg
    # of course, to properly simulate payload, we should change mass instead
    return np.array([0, 0, params.g*weight])

class AirDrag(DisturbanceForce):
    def __init__(self) -> None:
        super().__init__()

    def update_explicit_force(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> None:
        self.f_explicit = self.get_air_drag(state[3:6])

    def get_implicit_force_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3)
    
    @staticmethod
    def get_air_drag(v: np.ndarray) -> np.ndarray:
        """air drag force in inertial frame
        f = 0.5*c_d*area*v_norm^2*(v/v_norm)

        Args:
            v (np.ndarray): speed np.array([vx, vy, vz])

        Returns:
            np.ndarray: (3,) array to represent (f_x, f_y, f_z)
        """

        f = 0.5*params.c_d*params.area_frontal*np.sqrt(v[0]**2+v[1]**2+v[2]**2)*v
        return f

class WallEffect(DisturbanceForce):
    def __init__(self) -> None:
        self.wall_location = None
        super().__init__()

