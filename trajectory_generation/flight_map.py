import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import trajectory_generator_qp
import trajectory_logger

class FlightMap:
    def __init__(self, subtrajs: list[trajectory_generator_qp.TrajectoryGenerator]) -> None:
        self.subtrajs = subtrajs
        self.t_offset = np.array([0.0])
        self.coordinate_offset = [np.array([0.0, 0.0, 0.0])]
        self.total_num_of_subtrajs = len(self.subtrajs)
        self.total_time = 0.0
        self.stitch_subtrajs()

    def read_data_by_time(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_clamped = min(self.subtrajs[-1].waypoints.waypoint_time_stamp[-1] + self.t_offset[-1], t)
        i_subtraj = self.find_subtrajs(t_clamped)
        selected_subtraj = self.subtrajs[i_subtraj]
        t_adjusted = t_clamped - self.t_offset[i_subtraj]
        i_section = self.find_sections(t_adjusted, selected_subtraj)
        x = np.array([selected_subtraj.profiles[0][i_section].sample_polynomial(0, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[1][i_section].sample_polynomial(0, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[2][i_section].sample_polynomial(0, t_adjusted - selected_subtraj.time_shift[i_section])])
        v = np.array([selected_subtraj.profiles[0][i_section].sample_polynomial(1, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[1][i_section].sample_polynomial(1, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[2][i_section].sample_polynomial(1, t_adjusted - selected_subtraj.time_shift[i_section])])
        x_dot2 = np.array([selected_subtraj.profiles[0][i_section].sample_polynomial(2, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[1][i_section].sample_polynomial(2, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[2][i_section].sample_polynomial(2, t_adjusted - selected_subtraj.time_shift[i_section])])
        x_dot3 = np.array([selected_subtraj.profiles[0][i_section].sample_polynomial(3, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[1][i_section].sample_polynomial(3, t_adjusted - selected_subtraj.time_shift[i_section]),
                      selected_subtraj.profiles[2][i_section].sample_polynomial(3, t_adjusted - selected_subtraj.time_shift[i_section])])
        x_dot4 = np.array([selected_subtraj.profiles[0][i_section].sample_polynomial(4, t_adjusted - selected_subtraj.time_shift[i_section]),
                           selected_subtraj.profiles[1][i_section].sample_polynomial(4, t_adjusted - selected_subtraj.time_shift[i_section]),
                           selected_subtraj.profiles[2][i_section].sample_polynomial(4, t_adjusted - selected_subtraj.time_shift[i_section])])
        x += self.coordinate_offset[i_subtraj]
        return x, v, x_dot2, x_dot3, x_dot4
        
    def find_subtrajs(self, t) -> int:
        i = np.searchsorted(self.t_offset, t) - 1
        i = min(self.total_num_of_subtrajs - 1, i)
        i = max(0, i)
        return i
        
    def find_sections(self, t, trajectory: trajectory_generator_qp.TrajectoryGenerator) -> int:
        i = np.searchsorted(trajectory.waypoints.waypoint_time_stamp, t) - 1
        i = min(trajectory.waypoints.number_of_sections - 1, i)
        i = max(0, i)
        return i
    
    def stitch_subtrajs(self):
        for subtraj in self.subtrajs:
            self.total_time += subtraj.waypoints.total_time
            for dim in subtraj.profiles:
                for section in dim:
                    section.extend_to_derivative_order(4)            
        for subtraj in self.subtrajs[:-1]:
            self.t_offset = np.hstack((self.t_offset, subtraj.waypoints.waypoint_time_stamp[-1]))
            self.coordinate_offset = np.vstack((self.coordinate_offset, subtraj.waypoints.coordinates[-1]))
        self.coordinate_offset = np.cumsum(self.coordinate_offset, axis=0)
        self.t_offset = np.cumsum(self.t_offset)

    def plot_stitched_traj(self):
        t_span = np.arange(0, self.total_time, 0.1)
        fig, axs = plt.subplots(3, 5, sharex=True)
        x_trace = np.empty((0, 3))
        v_trace = np.empty((0, 3))
        x_dot2_trace = np.empty((0, 3))
        x_dot3_trace = np.empty((0, 3))
        x_dot4_trace = np.empty((0, 3))
        for t in t_span:
            x, v, x_dot2, x_dot3, x_dot4 = self.read_data_by_time(t)
            x_trace = np.vstack((x_trace, x))
            v_trace = np.vstack((v_trace, v))
            x_dot2_trace = np.vstack((x_dot2_trace, x_dot2))
            x_dot3_trace = np.vstack((x_dot3_trace, x_dot3))
            x_dot4_trace = np.vstack((x_dot4_trace, x_dot4))
        axs[0, 0].plot(t_span, x_trace[:, 0])
        axs[1, 0].plot(t_span, x_trace[:, 1])
        axs[2, 0].plot(t_span, x_trace[:, 2])
        axs[0, 1].plot(t_span, v_trace[:, 0])
        axs[1, 1].plot(t_span, v_trace[:, 1])
        axs[2, 1].plot(t_span, v_trace[:, 2])
        axs[0, 2].plot(t_span, x_dot2_trace[:, 0])
        axs[1, 2].plot(t_span, x_dot2_trace[:, 1])
        axs[2, 2].plot(t_span, x_dot2_trace[:, 2])
        axs[0, 3].plot(t_span, x_dot3_trace[:, 0])
        axs[1, 3].plot(t_span, x_dot3_trace[:, 1])
        axs[2, 3].plot(t_span, x_dot3_trace[:, 2])
        axs[0, 4].plot(t_span, x_dot4_trace[:, 0])
        axs[1, 4].plot(t_span, x_dot4_trace[:, 1])
        axs[2, 4].plot(t_span, x_dot4_trace[:, 2])

        fig1, axs1 = plt.subplots(1, 1, sharex=True)
        axs1 = fig1.add_subplot(111, projection='3d')
        axs1.plot3D(x_trace[:, 0], x_trace[:, 1], x_trace[:, 2])
        for point in self.coordinate_offset:
            axs1.plot3D(point[0], point[1], point[2], 'o', c='green', label='Points')
        axs1.set_xlabel('X')
        axs1.set_ylabel('Y')
        axs1.set_zlabel('Z')

def construct_map_file_path(file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_folder_dir = os.path.join(os.path.dirname(current_dir), "data")
    map_folder_dir = os.path.join(map_folder_dir, "map")    
    file_path = os.path.join(map_folder_dir, file_name)
    return file_path

def write_flight_map(map: FlightMap, file_name: str) -> None:
    """Write trajectory information into file"""
    file_path = construct_map_file_path(file_name)
    if os.path.exists(file_path):
        raise ValueError("File exist: " + file_path)
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(map, file)
        print("Map data is written into:\n" + file_path)

def read_flight_map(file_name: str) -> FlightMap:
    file_path = construct_map_file_path(file_name)
    if os.path.exists(file_path):
        print("Map data read from:\n" + file_path)
        with open(file_path, 'rb') as file:
            loaded_map = pickle.load(file)
        return loaded_map
    else:
        raise ValueError("File not exist: " + file_path)
        
def construct_map_from_subtrajs(file_name_prefix: str, num_of_subtrajs: int) -> FlightMap:
    file_name_suffix = ".pkl"
    subtrajs = []
    for i in range(num_of_subtrajs):
        file_name = file_name_prefix + str(i) + file_name_suffix
        subtrajs.append(trajectory_logger.read_trajectory(file_name))
    return FlightMap(subtrajs)

if __name__ == "__main__":
    file_name_prefix = "5_random_wp_map_"
    constructed_map = construct_map_from_subtrajs(file_name_prefix, 24)
    write_flight_map(constructed_map, "random_wp_map_2min.pkl")
    map_reload = read_flight_map("random_wp_map_2min.pkl")
    map_reload.plot_stitched_traj()
    plt.show()