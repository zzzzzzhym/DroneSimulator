import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import trajectory_generator_qp
import trajectory_logger
import traj_gen_utils
import common_utils.robust_pickle

class FlightMap:
    def __init__(self, subtrajs: list[trajectory_generator_qp.TrajectoryGenerator], x_limit=None, y_limit=None, z_limit=None) -> None:
        self.subtraj_candidates = subtrajs
        self.subtraj_selected = []
        self.t_offset = np.array([0.0])
        self.coordinate_offset = [np.array([0.0, 0.0, 0.0])]
        self.num_of_subtrajs_in_use = 0
        self.total_time = 0.0
        self.stitch_subtrajs(x_limit, y_limit, z_limit)

    def read_data_by_time(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_clamped = np.clip(t, a_max=self.total_time, a_min=None)
        i_subtraj = self.find_subtrajs(t_clamped)
        selected_subtraj = self.subtraj_selected[i_subtraj]
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
        """to properly find the subtrajectory, t need to be clamped"""
        i = np.searchsorted(self.t_offset, t) - 1
        i = np.clip(i, a_min=0, a_max=self.num_of_subtrajs_in_use - 1)
        return i
        
    def find_sections(self, t, trajectory: trajectory_generator_qp.TrajectoryGenerator) -> int:
        """subtrajectory is divided into sections by sampled waypoints. Search which section to use."""
        i = np.searchsorted(trajectory.waypoints.waypoint_time_stamp, t) - 1
        i = np.clip(i, 0, trajectory.waypoints.number_of_sections - 1)
        return i
    
    def stitch_subtrajs(self, x_limit=None, y_limit=None, z_limit=None) -> None:
        """Compute the coordinate offset and time offset of each subtrajectory.
        read_data_by_time() will use this to connect the subtrajectories.
        If limits are not specify, the stitching will be done in free space. 
        If limits are specify, subtrajectory will be skipped if end point is out of bound.
        Note that limitting does not guarantee the whole trajecoty is in bound. Leave enough margin."""
        # make sure the subtrajs have enough derivative order
        for subtraj in self.subtraj_candidates:
            for dim in subtraj.profiles:
                for section in dim:
                    section.extend_to_derivative_order(4)          
        # compute the coordinate offset and time offset of each subtrajectory
        t_list = [0.0]
        self.coordinate_offset = [np.array([0.0, 0.0, 0.0])]    # initialize in case of redo stitching
        self.subtraj_selected = []    # initialize in case of redo stitching
        for subtraj in self.subtraj_candidates:
            end_point = self.coordinate_offset[-1]+subtraj.waypoints.coordinates[-1]
            if (not self.check_out_of_bound(end_point[0], x_limit)) and \
               (not self.check_out_of_bound(end_point[1], y_limit)) and \
               (not self.check_out_of_bound(end_point[2], z_limit)):
                t_list.append(subtraj.waypoints.waypoint_time_stamp[-1])
                self.coordinate_offset.append(end_point)
                self.subtraj_selected.append(subtraj)
        self.t_offset = np.cumsum(t_list)    # t_offset need to be a np array because np.searchsorted is used later
        self.total_time = self.t_offset[-1]
        self.num_of_subtrajs_in_use = len(t_list) - 1

    def check_out_of_bound(self, coordinate, limit=None) -> bool:
        result = False
        if limit is not None:
            if coordinate < -limit or coordinate > limit:
                result = True
        return result

    def plot_stitched_traj(self):
        t_span = np.arange(0, self.total_time, 0.1)
        fig, axs = plt.subplots(3, 5, sharex=True)
        x_trace = []
        v_trace = []
        x_dot2_trace = []
        x_dot3_trace = []
        x_dot4_trace = []
        for t in t_span:
            x, v, x_dot2, x_dot3, x_dot4 = self.read_data_by_time(t)
            x_trace.append(x)
            v_trace.append(v)
            x_dot2_trace.append(x_dot2)
            x_dot3_trace.append(x_dot3)
            x_dot4_trace.append(x_dot4)
        x_trace = np.array(x_trace)
        v_trace = np.array(v_trace)
        x_dot2_trace = np.array(x_dot2_trace)
        x_dot3_trace = np.array(x_dot3_trace)
        x_dot4_trace = np.array(x_dot4_trace)
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
        axs[0, 0].set_ylabel('X')
        axs[1, 0].set_ylabel('Y')
        axs[2, 0].set_ylabel('Z')
        axs[2, 0].set_xlabel('x')
        axs[2, 1].set_xlabel('x_dot')
        axs[2, 2].set_xlabel('x_dot2')
        axs[2, 3].set_xlabel('x_dot3')
        axs[2, 4].set_xlabel('x_dot4')

        fig1, axs1 = plt.subplots(1, 1, sharex=True)
        axs1 = fig1.add_subplot(111, projection='3d')
        axs1.plot3D(x_trace[:, 0], x_trace[:, 1], x_trace[:, 2])
        points = np.array(self.coordinate_offset)
        axs1.scatter(points[:, 0], points[:, 1], points[:, 2], color='green', label='Stitch Points')
        axs1.set_xlabel('X')
        axs1.set_ylabel('Y')
        axs1.set_zlabel('Z')
        set_axes_equal(axs1)

def set_axes_equal(ax):
    '''Make 3D axes have equal scale for all axes (x, y, z).'''
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(xlim)
    y_middle = np.mean(ylim)
    z_middle = np.mean(zlim)

    ax.set_xlim([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim([z_middle - max_range / 2, z_middle + max_range / 2])

def construct_map_file_path(file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "data", "map", file_name)
    return file_path

def write_flight_map(map: FlightMap, file_name: str) -> None:
    """Write trajectory information into file"""
    file_path = construct_map_file_path(file_name)
    relative_path = os.path.relpath(file_path, os.getcwd())
    if os.path.exists(file_path):
        raise ValueError("File exist: " + relative_path)
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(map, file)
        print("Map data is written into:\n" + relative_path)

def read_flight_map(file_name: str) -> FlightMap:
    file_path = construct_map_file_path(file_name)
    relative_path = os.path.relpath(file_path, os.getcwd())
    if os.path.exists(file_path):
        print("Map data read from:\n" + relative_path)
        with open(file_path, 'rb') as file:
            loaded_map = pickle.load(file)
        return loaded_map
    else:
        raise ValueError("File not exist: " + relative_path)
        
def construct_map_with_subtrajs(is_random=True, num_of_subtrajs: int=0, subtraj_id: list[int]=[], x_limit=None, y_limit=None, z_limit=None) -> FlightMap:
    subtrajs = []
    if is_random:
        for i in range(num_of_subtrajs):
            file_name = trajectory_logger.pick_a_trajectory()
            subtrajs.append(trajectory_logger.read_trajectory(file_name))
    else:
        for i in subtraj_id:
            file_name = trajectory_logger.pick_a_trajectory(i)
            subtrajs.append(trajectory_logger.read_trajectory(file_name))
    return FlightMap(subtrajs, x_limit, y_limit, z_limit)

def construct_map_with_specific_subtrajs(subtraj_names: list[str], x_limit=None, y_limit=None, z_limit=None) -> FlightMap:
    subtrajs = []
    for name in subtraj_names:
        file_name = traj_gen_utils.get_dir_from_traj_gen(["data", "map", name])
        check_trajectory(file_name)
        subtrajs.append(trajectory_logger.read_trajectory(file_name))
    return FlightMap(subtrajs, x_limit, y_limit, z_limit)

def check_trajectory(file_name: str) -> None:
    file_path = construct_map_file_path(file_name)
    if os.path.exists(file_path):
        print("Map data read from: " + os.path.relpath(file_path, os.getcwd()))
    else:
        raise ValueError("File not exist: " + os.path.relpath(file_path, os.getcwd()))

if __name__ == "__main__":
    # constructed_map = construct_map_with_subtrajs(is_random=False, subtraj_id=[21,22,23])
    # write_flight_map(constructed_map, "specified_wp_map_3sub.pkl")
    constructed_map = construct_map_with_subtrajs(num_of_subtrajs=3)
    write_flight_map(constructed_map, "random_wp_map_3sub.pkl")
    map_reload = read_flight_map("specified_wp_map_3sub.pkl")
    map_reload.plot_stitched_traj()
    plt.show()