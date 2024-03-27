import numpy as np

import trajectory_config
import waypoint_creator as waypoint_creator
import trajectory_generator as trajectory_generator

# points = (np.array([[0, 0],
#                     [1, 2],
#                     [2, -1],
#                     [4, 8],
#                     [5, 2]]))
# time_span = 5.0
points = (np.array([[0, 0, 0],
                    [1, 1, 0],
                    [2, 2, 0]]))
time_span = 2.0
waypoints = waypoint_creator.Waypoint(points, time_span)
config = trajectory_config.TrajectoryConfig(5, np.array([0,0,0]), np.array([0,0,0]))
trajecotry = trajectory_generator.TrajectoryGenerator(config, waypoints)


if __name__ == "__main__":
    trajecotry.section_position_polynomials[0][0][1] = 1
    trajecotry.section_position_polynomials[0][1][1] = 1
    trajecotry.section_position_polynomials[1][0][1] = 1
    trajecotry.section_position_polynomials[1][1][1] = 1
    # print(trajecotry.section_position_polynomials)

    # result = trajecotry.get_reach_waypoint_constraint(trajecotry.section_position_polynomials[1], 0)
    # print(result)
    # result = trajecotry.get_continuity_constraint(trajecotry.section_position_polynomials[1])
    # print(result)
    result = trajecotry.get_equality_constraint(trajecotry.section_position_polynomials[1], 1)
    print(result)

    # trajecotry.get_trajectory()
    # print(trajecotry.section_position_polynomials)