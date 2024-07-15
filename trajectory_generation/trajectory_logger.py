import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import trajectory_config
import trajectory_generator_qp as trajectory_generator
import trajectory_time_optimizer
import waypoint_generator



def get_trajecotry(section_time_span, number_of_points, point_to_point_distance) -> trajectory_generator.TrajectoryGenerator:
    waypoint_generator_instance = waypoint_generator.WaypointGenerator(point_to_point_distance, 
                                                                    number_of_points, 
                                                                    section_time_span*number_of_points)

    waypoints = waypoint_generator_instance.waypoints
    print('before opt')
    print(waypoints.section_time)
    config = trajectory_config.TrajectoryConfig(order_of_polynomial=5, 
                                                initial_velocity=np.array([0,0,0]), 
                                                initial_acceleration=np.array([0,0,0]), 
                                                terminal_velocity=np.array([0,0,0]), 
                                                terminal_acceleration=np.array([0,0,0]))

    waypoints, trajectory = trajectory_time_optimizer.find_trajectory_with_optimal_time(waypoints, config)
    print('after opt')
    print(waypoints.section_time)
    return trajectory

def write_trajectory(trajectory: trajectory_generator.TrajectoryGenerator, file_name: str) -> None:
    """Write trajectory information into file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_folder_dir = os.path.join(os.path.dirname(current_dir), "data")
    map_folder_dir = os.path.join(map_folder_dir, "map")    
    file_path = os.path.join(map_folder_dir, file_name)
    if os.path.exists(file_path):
        raise ValueError("File exist: " + file_path)
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(trajectory, file)
        print("Trajectory data is written into:\n" + file_path)

def read_trajectory(file_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_folder_dir = os.path.join(os.path.dirname(current_dir), "data")
    map_folder_dir = os.path.join(map_folder_dir, "map")   
    map_path = os.path.join(map_folder_dir, file_name)    
    if os.path.exists(map_path):
        with open(map_path, 'rb') as file:
        # Deserialize each object from the file
            loaded_trajectory = pickle.load(file)
            return loaded_trajectory
    else:
        raise ValueError("No such file: " + map_path)

def make_random_trajectories(num_of_traj: int):
    file_name_prefix = "5_random_wp_map_"
    file_name_suffix = ".pkl"
    section_time_span = 1.0
    number_of_points = 5
    point_to_point_distance = 3.0    
    for i in range(num_of_traj):
        file_name = file_name_prefix + str(i) + file_name_suffix
        trajectory = get_trajecotry(section_time_span, number_of_points, point_to_point_distance)
        write_trajectory(trajectory, file_name)    

if __name__ == "__main__":
    # file_name = "5_random_wp_map.pkl"
    # section_time_span = 1.0
    # number_of_points = 5
    # point_to_point_distance = 3.0    
    # trajectory = get_trajecotry(section_time_span, number_of_points, point_to_point_distance)
    # write_trajectory(trajectory, file_name)

    # """To load file, use the following code
    # with open('trajectory.pkl', 'rb') as file:
    # # Deserialize each object from the file
    #     loaded_trajectory = pickle.load(file)
    # loaded_trajectory.plot_trajectory()
    # """
    # loaded_trajectory = read_trajectory(file_name)
    # loaded_trajectory.plot_trajectory()
    # plt.show()

    make_random_trajectories(24)
    loaded_trajectory = read_trajectory("5_random_wp_map_2.pkl")
    loaded_trajectory.plot_trajectory()    
    plt.show()