import numpy as np


class Waypoint:
    def __init__(self, coordinate_list: np.ndarray, total_time: float) -> None:
        '''
        self.section_time: a vector of m element, the i^th element is the i^th section time span
        self.section_time_stamp: arrival time at each waypoint
        '''        
        self.coordinates = coordinate_list
        self.number_of_waypoints = np.shape(coordinate_list)[0]
        self.number_of_sections = self.number_of_waypoints - 1
        self.section_distances = np.empty(self.number_of_sections)
        self.total_distance = 0.0
        self.total_time = total_time
        self.section_time = np.empty(self.number_of_sections)
        self.section_time_stamp = np.empty(self.number_of_sections + 1)
        self.get_time_and_distance()

    def get_time_and_distance(self):
        self.get_waypoint_distances()
        self.get_time_distribution()

    def get_time_distribution(self) -> None:
        self.section_time = self.total_time / \
            self.total_distance*self.section_distances
        self.section_time_stamp = np.cumsum(np.hstack((0, self.section_time)))

    def get_waypoint_distances(self):
        section_vector_list = self.coordinates[1:] - self.coordinates[:-1]
        for i, section_vector in enumerate(section_vector_list):
            self.section_distances[i] = np.sqrt(section_vector@section_vector)
            self.total_distance += self.section_distances[i]


if __name__ == "__main__":
    waypoint_list = np.array([[0, 0],
                              [1, 1],
                              [2, 0]])
    time = 5
    waypoint_instance = Waypoint(waypoint_list, time)
    print(waypoint_instance.section_distances)
    print(waypoint_instance.section_time_stamp)
    print(waypoint_instance.section_time)



