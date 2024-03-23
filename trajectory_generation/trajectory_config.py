import numpy as np
from dataclasses import dataclass
import waypoint

@dataclass
class TrajectoryConfig:
    order_of_polynomial: int
    initial_velocity: np.ndarray
    initial_acceleration: np.ndarray
    terminal_velocity: np.ndarray
    terminal_acceleration: np.ndarray

