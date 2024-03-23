import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryConfig:
    waypoints: np.ndarray
    order_of_polynomial: int = 5
    time: float = 5.0
    initial_state: np.ndarray = np.array([0, 0])
    terminal_state: np.ndarray = np.array([0, 0])


demo1 = TrajectoryConfig(np.array([[0, 0],
                                   [1, 2],
                                   [2, -1],
                                   [4, 8],
                                   [5, 2]]))


if __name__ == "__main__":
    print(demo1)
    demo1.time = 10
    print(np.shape(demo1.waypoints)[0])
    for waypoint in demo1.waypoints:
        print(waypoint)
