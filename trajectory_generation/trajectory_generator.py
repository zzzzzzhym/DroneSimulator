import numpy as np
from scipy.optimize import minimize
import scipy.integrate as spi
import matplotlib.pyplot as plt

import trajectory_config
import waypoint_creator

dimensions = {'x': 0,
              'y': 1,
              'z': 2}
# todo:
# separate x y z in waypoints
# calculate differantiate_polynomial() once only
# check matrix transpose related to polynomials @
class TrajectoryGenerator:
    '''
    Args:
        section_position_polynomials: 3 by m by n array of position function. 
                                      3 is dimension index of (x, y ,z). 
                                      each row is a section.
                                      each column is the coefficient of polynomial from 0 to n-1
    '''
    def __init__(self, config: trajectory_config.TrajectoryConfig, waypoints: waypoint.Waypoint) -> None:
        self.config = config
        self.waypoints = waypoints
        self.section_position_polynomials = np.empty((3, self.waypoints.number_of_sections, self.config.order_of_polynomial + 1))

    def initialize_solution(self, i_dim):
        # todo: overlap waypoint divided by 0 protection
        for i, each_section_polynomial in enumerate(self.section_position_polynomials[i_dim]):
            each_section_polynomial[1] = (self.waypoints.coordinates[i+1][i_dim] - self.waypoints.coordinates[i][i_dim])/(self.waypoints.waypoint_time_stamp[i+1] - self.waypoints.waypoint_time_stamp[i])
            each_section_polynomial[0] = self.waypoints.coordinates[i][i_dim] - each_section_polynomial[1]*self.waypoints.waypoint_time_stamp[i]

    def get_trajectory(self):
        for dim in dimensions:
            self.initialize_solution(dimensions[dim])
            self.find_trajectory(dimensions[dim])
        self.plot_trajectory()

    def pack_section_polynomial_to_vector(self, polynimial: np.ndarray) -> np.ndarray:
        result = np.reshape(polynimial, -1)
        return result

    def unpack_vector_to_polynomial(self, var_vector: np.ndarray) -> np.ndarray:
        '''
        convert vector of decision vars into section_polynomials
        '''
        result = np.reshape(var_vector, 
                            (self.waypoints.number_of_sections, self.config.order_of_polynomial + 1))
        return result

    def find_trajectory(self, i_dim):
        '''
        i_dim: dimension_index (0: x, 1: y, 2: z)
        '''
        inital_state = self.pack_section_polynomial_to_vector(self.section_position_polynomials[i_dim])
        cons = {'type': 'eq', 'fun': self.wrap_equality_constraint, 'args':(i_dim, )}
        # print('start optimize')
        optimization_result = minimize(self.wrap_cost_function, inital_state, constraints=cons)
        print(optimization_result)
        packed_result = optimization_result.x 
        self.section_position_polynomials[i_dim] = self.unpack_vector_to_polynomial(packed_result)

    def wrap_cost_function(self, x) -> float:
        position_polynomials = self.unpack_vector_to_polynomial(x)
        cost = self.get_cost_of_all_sections(position_polynomials)
        return cost

    def get_cost_of_all_sections(self, position_polynomials: np.ndarray) -> float:
        '''
        return cost of snap^2 for all sections
        '''
        snap_polynomials = np.empty(np.shape(position_polynomials))
        for i, polynomial in enumerate(position_polynomials):
            snap_polynomials[i] = differantiate_polynomial(polynomial, 4)
        cost = 0.0
        for i, row in enumerate(snap_polynomials):
            cost += self.get_cost_of_a_section(self.waypoints.waypoint_time_stamp[i], self.waypoints.waypoint_time_stamp[i+1], row)
        return cost

    def get_cost_of_a_section(self, t_start, t_stop, snap_polynomial) -> float:
        result, error = spi.quad(self.return_derivative_of_cost_function, t_start, t_stop, args = (snap_polynomial))
        # todo: error check
        return result 
    
    def return_derivative_of_cost_function(self, t, polynomial) -> float:
        result = sample_polynomial(polynomial, t)**2
        return result

    def wrap_equality_constraint(self, x, i_dim) -> np.ndarray:
        position_polynomials = self.unpack_vector_to_polynomial(x)
        result = self.get_equality_constraint(position_polynomials, i_dim)
        return result

    def get_equality_constraint(self, position_polynomials: np.ndarray, i_dim: int) -> np.ndarray:
        velocity_polynomials = np.zeros(np.shape(position_polynomials))
        acceleration_polynomials = np.zeros(np.shape(velocity_polynomials))
        for i_section in range(self.waypoints.number_of_sections):
            velocity_polynomials[i_section] = differantiate_polynomial(position_polynomials[i_section], 1)
            acceleration_polynomials[i_section] = differantiate_polynomial(velocity_polynomials[i_section], 1)
        constraint_delta_0 = self.get_reach_waypoint_constraint(position_polynomials, i_dim)
        constraint_delta_1_0 = self.get_continuity_constraint(position_polynomials)
        constraint_delta_1_1 = self.get_continuity_constraint(velocity_polynomials)
        constraint_delta_1_2 = self.get_continuity_constraint(acceleration_polynomials)
        constraint_delta_2 = self.get_terminal_constraint(position_polynomials, i_dim)
        constraint_delta = np.append(constraint_delta_0, constraint_delta_1_0)
        constraint_delta = np.append(constraint_delta, constraint_delta_1_1)
        constraint_delta = np.append(constraint_delta, constraint_delta_1_2)
        constraint_delta = np.append(constraint_delta, constraint_delta_2)
        return constraint_delta

    def get_reach_waypoint_constraint(self, position_component_polynomials, i_dim) -> np.ndarray:
        result = np.empty((self.waypoints.number_of_sections, 2))
        for i, polynomial in enumerate(position_component_polynomials):
            result[i][0] = sample_polynomial(polynomial, self.waypoints.waypoint_time_stamp[i]) - self.waypoints.coordinates[i][i_dim]
            result[i][1] = sample_polynomial(polynomial, self.waypoints.waypoint_time_stamp[i+1]) - self.waypoints.coordinates[i+1][i_dim]
        reshaped_result = np.reshape(result, -1)
        return reshaped_result
    
    def get_terminal_constraint(self, position_component_polynomials, i_dim) -> np.ndarray:
        result = np.empty((0,))
        first_velocity_component_polynomial = differantiate_polynomial(position_component_polynomials[0], 1)
        first_acceleration_component_polynomial = differantiate_polynomial(first_velocity_component_polynomial, 1)
        last_velocity_component_polynomial = differantiate_polynomial(position_component_polynomials[-1], 1)
        last_acceleration_component_polynomial = differantiate_polynomial(last_velocity_component_polynomial, 1)

        temp_result = sample_polynomial(first_velocity_component_polynomial, self.waypoints.waypoint_time_stamp[0]) - self.config.initial_state[0][i_dim]
        result = np.append(result, temp_result)
        temp_result = sample_polynomial(first_acceleration_component_polynomial, self.waypoints.waypoint_time_stamp[0]) - self.config.initial_state[1][i_dim]
        result = np.append(result, temp_result)
        temp_result = sample_polynomial(last_velocity_component_polynomial, self.waypoints.waypoint_time_stamp[-1]) - self.config.terminal_state[0][i_dim]
        result = np.append(result, temp_result)
        temp_result = sample_polynomial(last_acceleration_component_polynomial, self.waypoints.waypoint_time_stamp[-1]) - self.config.terminal_state[1][i_dim]
        result = np.append(result, temp_result)
        return result

    def get_continuity_constraint(self, polynomials) -> np.ndarray:
        combined_result = np.empty((0, ))
        for i in range(self.waypoints.number_of_sections - 1):
            # protection: range(0) and range(-1) will skip this loop
            result = sample_polynomial(polynomials[i], self.waypoints.waypoint_time_stamp[i]) - \
                     sample_polynomial(polynomials[i+1], self.waypoints.waypoint_time_stamp[i])
            combined_result = np.append(combined_result, result)
        return combined_result
    
    def plot_trajectory(self):
        fig, axs = plt.subplots(1, 1, sharex=True)
        # Use projection='3d' for 3D plots
        axs = fig.add_subplot(111, projection='3d')
        for i in range(self.waypoints.number_of_sections):
            t_samples = np.linspace(self.waypoints.waypoint_time_stamp[i], self.waypoints.waypoint_time_stamp[i+1], 11)
            x = np.empty((0,))
            y = np.empty((0,))
            z = np.empty((0,))
            for t in t_samples:
                x = np.append(x, sample_polynomial(self.section_position_polynomials[0][i], t))
                y = np.append(y, sample_polynomial(self.section_position_polynomials[1][i], t))
                z = np.append(z, sample_polynomial(self.section_position_polynomials[2][i], t))
            axs.plot3D(x,y,z)
        axs.plot3D(self.waypoints.coordinates[:, 0],self.waypoints.coordinates[:, 1],self.waypoints.coordinates[:, 2], '.', c='blue', label='Points')
        axs.set_title('3D Plot')
        axs.set_xlabel('X')
        axs.set_ylabel('Y')
        axs.set_zlabel('Z')
        axs.axis('equal')        
        plt.show()


def differantiate_polynomial(polynomial: np.ndarray, order_of_derivative: int) -> np.ndarray:
    number_of_terms = np.shape(polynomial)[0]
    derivative = np.zeros(number_of_terms)
    if number_of_terms - order_of_derivative >= 0:
        for i in range(order_of_derivative, number_of_terms, 1):
            new_order, coefficient = differantiate_monomial(i, order_of_derivative)
            derivative[new_order] = coefficient*polynomial[i]
    return derivative
        

def differantiate_monomial(order_of_monomial: int, order_of_derivative: int) -> tuple[int, int]:
    '''
    calculate i^th order of t^n
    Args:
        order_of_monomial: n
        order_of_derivative: number of attempts to differantiate
    Returns:
        differantiated_order: order of monomial after differantiation
        coefficient: coefficient generated from differantiation
    '''
    differantiated_order = 0
    coefficient = 0
    if order_of_derivative <= order_of_monomial:
        coefficient = np.prod(range(order_of_monomial, order_of_monomial - order_of_derivative, -1))
        differantiated_order = order_of_monomial - order_of_derivative
    return (differantiated_order, coefficient)

def sample_polynomial(polynomial: np.ndarray, x: float) -> float:
    '''
    given polynomial and x, calculate sum(polynomial[i]*x^i)
    '''
    number_of_terms = np.shape(polynomial)[0]
    exponent = np.array(range(number_of_terms))
    result = polynomial @ (x**exponent)
    return result



if __name__ == "__main__":
    # x^3 2th derivative to be 6x
    diff_order, coeff = differantiate_monomial(3, 2)
    print(diff_order, coeff)
    # x^3 3th derivative to be 6
    diff_order, coeff = differantiate_monomial(3, 3)
    print(diff_order, coeff)
    
    coeff = differantiate_polynomial(np.array([3, 1, 1, 2]), 2)
    print(coeff)






