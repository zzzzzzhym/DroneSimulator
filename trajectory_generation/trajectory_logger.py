import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import trajectory_config
import trajectory_generator_qp as trajectory_generator
import trajectory_time_optimizer
import waypoint_generator
import traj_gen_utils


def get_trajecotry(section_time_span, number_of_points, point_to_point_distance) -> trajectory_generator.TrajectoryGenerator:
    waypoint_generator_instance = waypoint_generator.WaypointGenerator(point_to_point_distance, 
                                                                    number_of_points, 
                                                                    section_time_span*number_of_points)

    waypoints = waypoint_generator_instance.waypoints
    print('before opt')
    print(waypoints.section_time)
    config = trajectory_config.TrajectoryConfig()
    waypoints, trajectory = trajectory_time_optimizer.find_trajectory_with_optimal_time(waypoints, config)
    print('after opt')
    print(waypoints.section_time)
    return trajectory

def write_trajectory(trajectory: trajectory_generator.TrajectoryGenerator, file_name: str) -> None:
    """Write trajectory information into file"""
    file_path = traj_gen_utils.get_dir_from_traj_gen(["data", "map", file_name])
    if os.path.exists(file_path):
        raise ValueError("File exist: " + file_path)
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(trajectory.__dict__, file)
        print("Trajectory data is written into:\n" + os.path.relpath(file_path))

def read_trajectory(file_name: str):
    map_path = traj_gen_utils.get_dir_from_traj_gen(["data", "map", file_name])
    if os.path.exists(map_path):
        with open(map_path, 'rb') as file:
            state_dict = pickle.load(file)
            # Create a new instance without calling __init__
            loaded_trajectory = trajectory_generator.TrajectoryGenerator.__new__(trajectory_generator.TrajectoryGenerator)
            # Update the new instance with the saved state
            loaded_trajectory.__dict__.update(state_dict)
            return loaded_trajectory
    else:
        raise ValueError("No such file: " + map_path)

def make_random_trajectories(num_of_traj: int):
    section_time_span = 2.0
    number_of_points = 5
    point_to_point_distance = 3.0
    target_dir = traj_gen_utils.get_dir_from_traj_gen(["data", "map"])
    for i in range(num_of_traj):
        file_name = get_valid_path_to_write("5_random_wp_map_", ".pkl", i, target_dir)
        trajectory = get_trajecotry(section_time_span, number_of_points, point_to_point_distance)
        write_trajectory(trajectory, file_name)    

def get_valid_path_to_write(prefix, suffix, num, file_dir):
    j = num
    while j <= 100000:
        file_name = prefix + str(j) + suffix
        file_path = traj_gen_utils.get_dir_from_traj_gen([file_dir, file_name])
        if not os.path.exists(file_path):
            return file_path
        j += 1
    raise ValueError(f"Map data file index too large: {j}")

def pick_a_trajectory(file_id: int=None):
    map_dir = traj_gen_utils.get_dir_from_traj_gen(["data", "map"])
    files = os.listdir(map_dir)
    if files:
        if file_id is None:
            i = np.random.randint(0, len(files))
            result = files[i]
        else:
            result = files[file_id]
    else:
        raise ValueError("map directory is empty")
    return result


if __name__ == "__main__":
    make_random_trajectories(196)
    loaded_trajectory = read_trajectory("5_random_wp_map_199.pkl")
    loaded_trajectory.plot_trajectory()    
    plt.show()
