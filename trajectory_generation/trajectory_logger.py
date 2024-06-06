import numpy as np
import matplotlib.pyplot as plt

import trajectory_config
import trajectory_generator_qp as trajectory_generator
import trajectory_time_optimizer
import waypoint_generator


section_time_span = 1.0
number_of_points = 10
point_to_point_distance = 1.0

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


if __name__ == "__main__":
    '''
    optimization check
    '''
    with open('waypoint_list.csv', 'w') as file:
        print(waypoints.coordinates, file=file)
    # print(trajectory.position_profiles)
    trajectory.plot_trajectory()
    plt.show()
    

