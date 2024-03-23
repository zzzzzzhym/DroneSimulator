import numpy as np
import trajectory_config as config
from scipy.optimize import minimize
import scipy.integrate as spi

# todo:
# separate x y z in waypoints
# calculate differantiate_polynomial() once only
# check matrix transpose related to polynomials @
class TrajectoryGenerator:
    def __init__(self, settings: config.TrajectoryConfig) -> None:
        '''
        self.section_time: a vector of m element, the i^th element is the i^th section time span
        self.section_time_stamp: arrival time at each waypoint
        self.section_polynomial: m by n array, each row is a section, each column is the coefficient of polynomial from 0 to n-1
        '''
        self.settings = settings
        self.number_of_waypoints = np.shape(self.settings.waypoints)[0]
        self.number_of_sections = self.number_of_waypoints - 1
        self.section_distances = np.empty(self.number_of_sections)
        self.total_distance = 0.0
        self.section_time = np.empty(self.number_of_sections)
        self.section_time_stamp = np.empty(self.number_of_sections + 1)
        self.section_position_polynomials = np.empty(self.number_of_sections, self.settings.order_of_polynomial + 1)
        self.section_velocity_polynomials = np.empty(self.number_of_sections, self.settings.order_of_polynomial + 1)
        self.section_acceleration_polynomials = np.empty(self.number_of_sections, self.settings.order_of_polynomial + 1)
        self.section_jerk_polynomials = np.empty(self.number_of_sections, self.settings.order_of_polynomial + 1)
        self.section_snap_polynomials = np.empty(self.number_of_sections, self.settings.order_of_polynomial + 1)

    def get_trajectory(self):
        self.get_waypoint_distances()
        self.get_time_distribution()
        self.find_trajectory()

    def get_time_distribution(self) -> None:
        self.section_time = self.settings.time/self.total_distance*self.section_distances
        self.section_time_stamp = np.cumsum([0, self.section_time])

    def get_waypoint_distances(self):
        diff_of_waypoints = self.settings.waypoints[1:] - self.settings.waypoints[:-1] 
        for i, each_diff in enumerate(diff_of_waypoints):
            self.section_distances[i] = np.sqrt(each_diff@each_diff)
            self.total_distance += self.section_distances[i]

    def pack_section_polynomial_to_vector(self, polynimial: np.ndarray) -> np.ndarray:
        result = np.reshape(polynimial, -1)
        return result

    def unpack_vector_to_polynomial(self, var_vector: np.ndarray) -> np.ndarray:
        '''
        convert vector of decision vars into section_polynomials
        '''
        result = np.reshape(var_vector, 
                            (self.number_of_sections, self.settings.order_of_polynomial + 1))
        return result

    def find_trajectory(self):
        inital_state = self.pack_section_polynomial_to_vector(self.section_position_polynomials)
        cons = {'type': 'eq', 'fun': self.get_equality_constraint}
        packed_result = minimize(self.get_cost_of_all_sections, inital_state, constraints=cons)
        self.section_position_polynomials = self.unpack_vector_to_polynomial(packed_result)

    def get_cost_of_all_sections(self, position_polynomials: np.ndarray) -> float:
        '''
        return cost of snap^2 for all sections
        '''
        snap_polynomials = position_polynomials @ differantiate_polynomial(self.settings.order_of_polynomial, 4).T
        cost = 0.0
        for i, row in enumerate(snap_polynomials):
            cost += self.get_cost_of_a_section(self.section_time_stamp[i], self.section_time_stamp[i+1], row)
        return cost

    def get_cost_of_a_section(self, t_start, t_stop, snap_polynomial):
        spi.quad(self.return_derivative_of_cost_function, t_start, t_stop, args = (snap_polynomial))

    def return_derivative_of_cost_function(self, polynomial, t) -> float:
        vector_of_t = sample_polynomial(self.settings.order_of_polynomial, t)
        result = (vector_of_t @ polynomial)**2
        return result

    def get_equality_constraint(self, x):
        position_polynomials = self.unpack_vector_to_polynomial(x)
        velocity_polynomials = position_polynomials @ differantiate_polynomial(self.settings.order_of_polynomial, 1)
        acceleration_polynomials = velocity_polynomials @ differantiate_polynomial(self.settings.order_of_polynomial, 1)
        constraint_delta_0 = self.get_reach_waypoint_constraint(position_polynomials)
        constraint_delta_1_0 = self.get_continuity_constraint(position_polynomials)
        constraint_delta_1_1 = self.get_continuity_constraint(velocity_polynomials)
        constraint_delta_1_2 = self.get_continuity_constraint(acceleration_polynomials)
        constraint_delta = np.append(constraint_delta_0, constraint_delta_1_0)
        constraint_delta = np.append(constraint_delta, constraint_delta_1_1)
        constraint_delta = np.append(constraint_delta, constraint_delta_1_2)
        return constraint_delta

    def get_reach_waypoint_constraint(self, polynomials):
        result = np.empty(self.number_of_waypoints)
        for i, polynomial in enumerate(polynomials):
            result[i] = polynomial @ sample_polynomial(self.settings.order_of_polynomial, self.section_time_stamp[i]) - self.settings.waypoints[i]
        return result
        # todo: x y and z dimension
    
    def get_continuity_constraint(self, polynomials):
        result = np.empty(self.number_of_waypoints)
        for i in range(polynomials):
            result[i] = (polynomials[i] - polynomials[i+1]) @ sample_polynomial(self.settings.order_of_polynomial, self.section_time_stamp[i])


def differantiate_polynomial(order_of_polynomial: int, order_of_derivative: int) -> np.ndarray:
    '''
    i^th derivative of [t^0, t^1, ... t^n]
    '''
    derivative = np.zeros(order_of_polynomial + 1)
    if order_of_polynomial - order_of_derivative >= 0:
        for i in range(order_of_derivative, order_of_polynomial + 1, 1):
            new_order, coefficient = differantiate_monomial(i, order_of_derivative)
            derivative[new_order] = coefficient
    return derivative
        

def differantiate_monomial(order_of_monomial: int, order_of_derivative: int) -> tuple[int, int]:
    '''
    calculate i^th order of t^n
    '''
    differantiated_order = 0
    coefficient = 0
    if order_of_derivative <= order_of_monomial:
        coefficient = np.prod(range(order_of_monomial, order_of_monomial - order_of_derivative, -1))
        differantiated_order = order_of_monomial - order_of_derivative
    return (differantiated_order, coefficient)

def sample_polynomial(order: int, x: int) -> np.ndarray:
    '''
    calculate [t^0, t^1, ... t^n]
    '''    
    exponent = np.array(range(order + 1))
    return (x**exponent)

if __name__ == "__main__":
    # x^3 2th derivative to be 6x
    diff_order, coeff = differantiate_monomial(3, 2)
    print(diff_order, coeff)
    # x^3 3th derivative to be 6
    diff_order, coeff = differantiate_monomial(3, 3)
    print(diff_order, coeff)
    
    coeff = differantiate_polynomial(3, 2)
    print(coeff)
    coeff = differantiate_polynomial(3, 3)
    print(coeff)
    coeff = differantiate_polynomial(3, 4)
    print(coeff)

    print(sample_polynomial(3,2))
    print(sample_polynomial(3,0))


