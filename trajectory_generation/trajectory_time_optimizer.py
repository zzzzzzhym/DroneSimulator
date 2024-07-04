import numpy as np
import cvxopt
import cvxpy
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import copy

import trajectory_config
import waypoint
import trajectory_generator_qp


def has_constraint_violation(section_time: np.ndarray) -> bool:
    return (np.min(section_time) < 0)

def get_total_time_invariant_vectors(number_of_sections: int) -> np.ndarray:
    vector_base = np.ones(number_of_sections)*( - 1 / (number_of_sections - 1))
    vectors = np.zeros((0, np.shape(vector_base)[0] ))
    for i in range(number_of_sections):
        vector = np.copy(vector_base)
        vector[i] = 1
        vectors = np.vstack((vectors, vector))
    return vectors

def get_total_cost(waypoints: waypoint.Waypoint, config: trajectory_config.TrajectoryConfig):
    trajectory = trajectory_generator_qp.TrajectoryGenerator(config, waypoints)
    trajectory.get_trajectory()
    return np.sum(trajectory.cost) 

def get_directional_derivative(waypoints: waypoint.Waypoint, config: trajectory_config.TrajectoryConfig, vector_trial: np.ndarray, base_cost) -> np.ndarray:
    step_size = 0.1
    waypoints_trial = copy.deepcopy(waypoints)
    waypoints_trial.change_time_distribution(waypoints_trial.section_time + vector_trial*step_size)
    directional_derivative = (get_total_cost(waypoints_trial, config) - base_cost)/step_size 
    return directional_derivative

def get_all_directional_derivatives(waypoints: waypoint.Waypoint, config: trajectory_config.TrajectoryConfig, vectors: np.ndarray, base_cost) -> np.ndarray:
    step_size = 1
    directional_derivatives = np.zeros(waypoints.number_of_sections)
    
    for i in range(waypoints.number_of_sections):
        directional_derivatives[i] = get_directional_derivative(waypoints, config, vectors[i], base_cost)

    return directional_derivatives

def optimize_time_distribution(waypoints: waypoint.Waypoint, config: trajectory_config.TrajectoryConfig):
    """backtracking line search to find optimal section time, the algorithm stops on hitting constraint or max iteration

    Args:
        waypoints (waypoint.Waypoint): _description_
        config (trajectory_config.TrajectoryConfig): _description_

    Returns:
        _type_: _description_
    """
    step_size = 1
    max_steps = 10
    alpha = 0.5
    beta = 0.5

    vectors = get_total_time_invariant_vectors(waypoints.number_of_sections)
    new_waypoints = copy.deepcopy(waypoints)
    for i in range(max_steps):
        print('step:' + str(i))
        total_cost = get_total_cost(new_waypoints, config)
        directional_derivatives = get_all_directional_derivatives(new_waypoints, config, vectors, total_cost)
        # need a zero check
        norm_of_directional_derivatives = np.sqrt(directional_derivatives@directional_derivatives)
        delta_section_time = - (directional_derivatives/norm_of_directional_derivatives).T @ vectors*step_size
        current_section_time = np.copy(new_waypoints.section_time)
        new_section_time = new_waypoints.section_time + delta_section_time
        if has_constraint_violation(new_section_time) or (np.max(np.abs(directional_derivatives)) < 100):
            # stop
            break
        else:
            # apply backtracking line search
            new_waypoints.change_time_distribution(new_section_time)
            while get_total_cost(new_waypoints, config) > total_cost + alpha*directional_derivatives@delta_section_time:
                delta_section_time *= beta
                new_section_time = current_section_time + delta_section_time
                new_waypoints.change_time_distribution(new_section_time)

    return new_waypoints

def find_trajectory_with_optimal_time(waypoints: waypoint.Waypoint, config: trajectory_config.TrajectoryConfig) -> tuple[waypoint.Waypoint, trajectory_generator_qp.TrajectoryGenerator]:
    """entry point to optimize trajectory time distribution in all sections

    Args:
        waypoints (waypoint.Waypoint): _description_
        config (trajectory_config.TrajectoryConfig): _description_

    Returns:
        tuple[waypoint.Waypoint, trajectory_generator_qp.TrajectoryGenerator]: _description_
    """
    new_waypoints = optimize_time_distribution(waypoints, config)
    new_trajectory = trajectory_generator_qp.TrajectoryGenerator(config, new_waypoints)
    new_trajectory.get_trajectory()
    return new_waypoints, new_trajectory


if __name__ == "__main__":
    points = (np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 2, 0],
                    [0, 2, 0]]))
    time_span = 5.0
    waypoints = waypoint.Waypoint(points, time_span)

    init_velocity = np.array([0,0,0])
    init_acceleration = np.array([0,0,0])
    end_velocity = np.array([0,0,0])
    end_acceleration = np.array([0,0,0])
    config = trajectory_config.TrajectoryConfig(5, init_velocity, init_acceleration, end_velocity, end_acceleration)

    new_waypoints, new_trajectory = find_trajectory_with_optimal_time(waypoints, config)
    new_trajectory.plot_trajectory()

    trajectory = trajectory_generator_qp.TrajectoryGenerator(config, waypoints)
    trajectory.get_trajectory()
    trajectory.plot_trajectory()
    plt.show()

    print('original section time: ')
    print(waypoints.section_time)
    print('proposed section time: ')
    print(new_waypoints.section_time)
    
