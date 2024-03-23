import numpy as np


class Waypoint:
    def __init__(self, coordinate_list: np.ndarray, total_time: float) -> None:
        '''
        self.section_time: a vector of m element, the i^th element is the i^th section time span
        self.section_time_stamp: arrival time at each waypoint
        '''        
        self.coordinates = coordinate_list
        self.total_time = total_time
        self.dimensions = np.shape(coordinate_list[0])[0]

        self.number_of_waypoints = 0
        self.number_of_sections = 0
        self.section_distances = np.empty(self.number_of_sections)
        self.total_distance = 0.0
        self.section_time = np.empty(self.number_of_sections)
        self.waypoint_time_stamp = np.empty(self.number_of_sections + 1)
        self.initialize_waypoint(self.coordinates, self.total_time)

    def initialize_waypoint(self, coordinate_list: np.ndarray, total_time: float):
        self.number_of_waypoints = np.shape(coordinate_list)[0]
        self.number_of_sections = self.number_of_waypoints - 1
        self.section_distances, self.total_distance = self.get_waypoint_distances(coordinate_list, self.number_of_sections)
        self.section_time, self.waypoint_time_stamp = self.initialize_time_distribution(total_time, self.total_distance, self.section_distances)

    def initialize_time_distribution(self, total_time: float, total_distance: float, section_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """initialize time distribution such that section time is proportional to section distance

        Args:
            total_time (float): _description_
            total_distance (float): _description_
            section_distances (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        section_time = total_time / \
            total_distance*section_distances
        waypoint_time_stamp = self.generate_time_stamps(section_time)
        return section_time, waypoint_time_stamp
    
    def change_time_distribution(self, section_time: np.ndarray) -> None:
        self.section_time = section_time
        self.waypoint_time_stamp = self.generate_time_stamps(section_time)

    def generate_time_stamps(self, section_time: np.ndarray) -> np.ndarray:
        return np.cumsum(np.hstack((0, section_time)))

    def get_waypoint_distances(self, coordinates, number_of_sections):
        section_vectors = coordinates[1:] - coordinates[:-1]
        section_distances = np.empty(number_of_sections)
        total_distance = 0.0
        for i, section_vector in enumerate(section_vectors):
            section_distances[i] = np.sqrt(section_vector@section_vector)
            total_distance += section_distances[i]
        return section_distances, total_distance

    def insert_middle_waypoints(self, sample_distance):
        self.coordinates = self.interpolate_sections(self.coordinates, sample_distance, self.dimensions)
        self.initialize_waypoint(self.coordinates, self.total_time)

    def interpolate_sections(self, coordinate_list: np.ndarray, sample_distance: float, dimensions: int):
        delta_distance = coordinate_list[1:] - coordinate_list[:-1]
        new_coordinate_list = np.empty((0,dimensions))
        for i, _ in enumerate(coordinate_list[0:-1]):
            number_of_samples = int(np.ceil(np.sqrt(delta_distance[i]@delta_distance[i]) / sample_distance)) + 1
            new_coordinate_list = np.vstack((new_coordinate_list, interpolate_coordinate(coordinate_list[i], coordinate_list[i+1], number_of_samples, dimensions)))
        new_coordinate_list = np.vstack((new_coordinate_list, coordinate_list[-1]))
        return new_coordinate_list

def interpolate_coordinate(start_coordinate, stop_coordinate, num, dim):
    samples = np.empty((num, 0))
    for i in range(dim):
        new_coordinate = np.linspace(start_coordinate[i], stop_coordinate[i], num)
        samples = np.hstack((samples, new_coordinate.reshape(-1, 1)))
    return samples[:-1]

if __name__ == "__main__":
    waypoint_list = np.array([[0, 0],
                              [1, 1],
                              [2, 0]])
    time = 5
    waypoint_instance = Waypoint(waypoint_list, time)
    print(waypoint_instance.section_distances)
    print(waypoint_instance.waypoint_time_stamp)
    print(waypoint_instance.section_time)
    waypoint_instance.insert_middle_waypoints(0.5)
    print(waypoint_instance.coordinates)

    a = interpolate_coordinate(np.array([1, 1, 1]), np.array([2, 2, 2]), 5, 3)
    print(a)



