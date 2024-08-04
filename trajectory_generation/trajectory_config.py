import numpy as np
from dataclasses import dataclass
import waypoint

@dataclass
class TrajectoryConfig:
    order_of_polynomial: int=7
    initial_velocity: np.ndarray=np.array([0,0,0])
    initial_acceleration: np.ndarray=np.array([0,0,0])
    initial_jerk: np.ndarray=np.array([0,0,0])
    terminal_velocity: np.ndarray=np.array([0,0,0])
    terminal_acceleration: np.ndarray=np.array([0,0,0])
    terminal_jerk: np.ndarray=np.array([0,0,0])

if __name__ == "__main__":
    config = TrajectoryConfig()
    for attr, value in config.__dict__.items():
        print(f"{attr}: {value}")