import numpy as np
from dataclasses import dataclass
import waypoint_creator

@dataclass
class TrajectoryConfig:
    order_of_polynomial: int
    initial_state: np.ndarray
    terminal_state: np.ndarray

