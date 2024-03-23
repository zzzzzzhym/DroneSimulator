import numpy as np
import waypoint

class WaypointGenerator:
    def __init__(self, adjacent_distance: float, number_of_points: int, total_time: float) -> None:
        self.waypoints = waypoint.Waypoint(self.generate_coordinates(adjacent_distance, number_of_points), 
                                           total_time)

    def generate_random_vector(self, distance: float) -> np.ndarray:
        x_weight = np.random.rand()
        y_weight = np.random.rand()
        z_weight = np.sqrt(1.0 - x_weight**2 + y_weight**2)

        dx = distance*x_weight*np.random.choice([1, -1])
        dy = distance*y_weight*np.random.choice([1, -1])
        dz = distance*z_weight*np.random.choice([1, -1])
        return np.array([dx, dy, dz])
        
    def generate_coordinates(self, distance, number_of_points) -> np.ndarray:
        coordinate_list = np.zeros((1, 3))
        for i in range(number_of_points):
            vector_to_next = self.generate_random_vector(distance)
            coordinate = coordinate_list[-1] + vector_to_next
            coordinate_list = np.vstack((coordinate_list, coordinate))
        return coordinate_list


if __name__ == "__main__":
    random_waypoint = WaypointGenerator(1.0, 10, 10.0)
    print(random_waypoint.waypoints.coordinates)
