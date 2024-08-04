import numpy as np
import cvxopt
import cvxpy
import torch
import qpth
from qpth.qp import QPFunction
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import copy
from enum import Enum, auto

import trajectory_config
import waypoint
import polynomial


solver_collection = {'cvxopt', 'cvxpy', 'qpfunction'}
which_solver = 'cvxopt'

class Dim(Enum):
    """Dimension of a 3D trajectory"""
    X = 0
    Y = 1
    Z = 2

class KinematicVars(Enum):
    """Kinematic variables"""
    Position = 0
    Velocity = 1
    Acceleration = 2
    Jerk = 3
    Snap = 4


class UnitCoeffPolynomialUtils:
    def __init__(self, order_of_polynomial: int) -> None:
        self.position = polynomial.Polynomial(np.ones(order_of_polynomial + 1), 4)

class TrajectoryGenerator:
    '''
    Args:
        position_profiles: position_profiles[dim][section_num][coeff]
                           dim is dimension index of (x, y ,z). 
                           section_num is a section id.
                           coeff is the coefficient of polynomial from 0 to n-1,
                           namely p_0, p_1 ... p_{n-1}
                           the corresponding polynomial profile is 
                           p_0 + p_1*(t - t_i) + ... p_{n-1}*(t-t_i)^(n-1)
    '''
    def __init__(self, config: trajectory_config.TrajectoryConfig, waypoints: waypoint.Waypoint) -> None:
        self.config = config
        self.waypoints = copy.deepcopy(waypoints)
        self.corridor_width = 0.2
        # self.waypoints.insert_middle_waypoints(self.corridor_width)
        self.number_of_polynomial_terms = self.config.order_of_polynomial + 1
        self.position_profiles = np.zeros((3, self.waypoints.number_of_sections, self.number_of_polynomial_terms))
        self.profiles: list[list[polynomial.Polynomial]] = []
        self.cost = np.array([-1, -1, -1])
        self.time_shift = self.get_time_shift(self.waypoints)
        # private utils
        self.coeff_utils = UnitCoeffPolynomialUtils(self.config.order_of_polynomial)

    def get_time_shift(self, waypoints: waypoint.Waypoint):
        return waypoints.waypoint_time_stamp[:-1]

    def initialize_profiles(self):
        """for each dimension, initialize the polynomial to a linear profile connect adjacent waypoints
        """
        for i_dim, component_profile in enumerate(self.position_profiles):
            for i, polynomial in enumerate(component_profile):
                dt = (self.waypoints.waypoint_time_stamp[i+1] - self.waypoints.waypoint_time_stamp[i])
                if dt != 0:
                    polynomial[1] = (self.waypoints.coordinates[i+1][i_dim] - self.waypoints.coordinates[i][i_dim])/dt
                    polynomial[0] = self.waypoints.coordinates[i][i_dim] - polynomial[1]*self.waypoints.waypoint_time_stamp[i]

    def get_trajectory(self):
        self.initialize_profiles()
        for dim in Dim:
            self.position_profiles[dim.value], self.cost[dim.value] = self.optimize_1d_profile(self.waypoints.coordinates[:,dim.value],
                                                                                   self.config.initial_velocity[dim.value],
                                                                                   self.config.initial_acceleration[dim.value],
                                                                                   self.config.initial_jerk[dim.value],
                                                                                   self.config.terminal_velocity[dim.value],
                                                                                   self.config.terminal_acceleration[dim.value],
                                                                                   self.config.terminal_jerk[dim.value])
        self.construct_profiles(self.position_profiles)

    def construct_profiles(self, position_profiles):
        """get the derivatives of the given profile and save them to the attribute 

        Args:
            position_profiles (_type_): _description_
        """
        for component in [0, 1, 2]:
            sub_list = []
            for position_polynomial in position_profiles[component]:
                profile = polynomial.Polynomial(position_polynomial, 3)
                sub_list.append(profile)
            self.profiles.append(sub_list)

    def pack_component_profile_to_vector(self, polynomial: np.ndarray) -> np.ndarray:
        result = np.reshape(polynomial, -1)
        return result

    def unpack_vector_to_component_profile(self, var_vector: np.ndarray) -> np.ndarray:
        '''
        convert vector of decision vars into component profile
        '''
        result = np.reshape(var_vector, 
                            (self.waypoints.number_of_sections, self.config.order_of_polynomial + 1))
        return result

    def get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        1/2*x.T*Q*x + p*x
        q: Q
        p: p
        Q = diag([q_sub_0, q_sub_1, ...])
        q_sub_i = 

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        derivative = 4
        q = np.empty((0,0))
        for i in range(self.waypoints.number_of_sections):
            q_sub = np.zeros((self.number_of_polynomial_terms, self.number_of_polynomial_terms))
            for row in range(self.number_of_polynomial_terms):
                for col in range(self.number_of_polynomial_terms):
                    if (row >= derivative) and (col >= derivative):
                        q_sub[row, col] = (np.prod(range(row, row - derivative, -1))*np.prod(range(col, col - derivative, -1))/(row + col - 2*derivative + 1))* \
                            (self.waypoints.waypoint_time_stamp[i+1]- self.waypoints.waypoint_time_stamp[i])**(row + col - 2*derivative + 1)
            q = block_diag(q, q_sub)
        p = np.zeros(self.waypoints.number_of_sections*(self.config.order_of_polynomial + 1))     
        return q, p

    def get_equality_constraint_matrices(self, component_of_waypoints: np.ndarray,
                                         init_velocity: float, 
                                         init_acceleration: float, 
                                         init_jerk: float, 
                                         end_velocity: float, 
                                         end_acceleration: float,
                                         end_jerk: float,) -> tuple[np.ndarray, np.ndarray]:
        '''
        Ax + b = 0
        a: A
        b: b
        '''
        row_length_of_a = self.waypoints.number_of_sections*(self.config.order_of_polynomial + 1)
        a = np.empty((0,row_length_of_a))
        b = np.empty((0,))
        # initial state and terminal state constraint
        # initial state
        coeff = self.coeff_utils.position.get_lumped_coefficient(0, self.waypoints.waypoint_time_stamp[0]) # polynomial(t0) = p0 similar below
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, 0)))
        b = np.append(b, component_of_waypoints[0])
        coeff = self.coeff_utils.position.get_lumped_coefficient(1, self.waypoints.waypoint_time_stamp[0])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, 0)))
        b = np.append(b, init_velocity)
        coeff = self.coeff_utils.position.get_lumped_coefficient(2, self.waypoints.waypoint_time_stamp[0])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, 0)))
        b = np.append(b, init_acceleration)
        coeff = self.coeff_utils.position.get_lumped_coefficient(3, self.waypoints.waypoint_time_stamp[0])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, 0)))
        b = np.append(b, init_jerk)
        # terminal state
        # position
        coeff = self.coeff_utils.position.get_lumped_coefficient(0, self.waypoints.waypoint_time_stamp[-1] - self.time_shift[-1])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, self.waypoints.number_of_sections - 1)))
        b = np.append(b, component_of_waypoints[-1])  
        # velocity      
        coeff = self.coeff_utils.position.get_lumped_coefficient(1, self.waypoints.waypoint_time_stamp[-1] - self.time_shift[-1])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, self.waypoints.number_of_sections - 1)))
        b = np.append(b, end_velocity)
        # accel
        coeff = self.coeff_utils.position.get_lumped_coefficient(2, self.waypoints.waypoint_time_stamp[-1] - self.time_shift[-1])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, self.waypoints.number_of_sections - 1)))
        b = np.append(b, end_acceleration)
        coeff = self.coeff_utils.position.get_lumped_coefficient(3, self.waypoints.waypoint_time_stamp[-1] - self.time_shift[-1])
        a = np.vstack((a, self.load_row_of_a(row_length_of_a, coeff, self.waypoints.number_of_sections - 1)))
        b = np.append(b, end_jerk)

        # continuity constraint (position)
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff_lh = self.coeff_utils.position.get_lumped_coefficient(0, self.waypoints.section_time[section_index])   # left hand side coeff
            coeff_rh = self.coeff_utils.position.get_lumped_coefficient(0, 0) # right hand side coeff
            a = np.vstack((a, 
                           self.load_row_of_a(row_length_of_a, coeff_lh, section_index) + self.load_row_of_a(row_length_of_a, -coeff_rh, section_index+1)
                           ))
            b = np.append(b, 0.0)        
        # continuity constraint (velocity)
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff_lh = self.coeff_utils.position.get_lumped_coefficient(1, self.waypoints.section_time[section_index])
            coeff_rh = self.coeff_utils.position.get_lumped_coefficient(1, 0)
            a = np.vstack((a, 
                           self.load_row_of_a(row_length_of_a, coeff_lh, section_index) + self.load_row_of_a(row_length_of_a, -coeff_rh, section_index+1)
                           ))
            b = np.append(b, 0.0)
        # continuity constraint (acceleration)
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff_lh = self.coeff_utils.position.get_lumped_coefficient(2, self.waypoints.section_time[section_index])
            coeff_rh = self.coeff_utils.position.get_lumped_coefficient(2, 0)
            a = np.vstack((a, 
                           self.load_row_of_a(row_length_of_a, coeff_lh, section_index) + self.load_row_of_a(row_length_of_a, -coeff_rh, section_index+1)
                           ))
            b = np.append(b, 0.0)        
        # continuity constraint (jerk)
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff_lh = self.coeff_utils.position.get_lumped_coefficient(3, self.waypoints.section_time[section_index])
            coeff_rh = self.coeff_utils.position.get_lumped_coefficient(3, 0)
            a = np.vstack((a, 
                           self.load_row_of_a(row_length_of_a, coeff_lh, section_index) + self.load_row_of_a(row_length_of_a, -coeff_rh, section_index+1)
                           ))
            b = np.append(b, 0.0)        
        # continuity constraint (snap)
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff_lh = self.coeff_utils.position.get_lumped_coefficient(4, self.waypoints.section_time[section_index])
            coeff_rh = self.coeff_utils.position.get_lumped_coefficient(4, 0)
            a = np.vstack((a, 
                           self.load_row_of_a(row_length_of_a, coeff_lh, section_index) + self.load_row_of_a(row_length_of_a, -coeff_rh, section_index+1)
                           ))
            b = np.append(b, 0.0)        
        return a, b

    def get_inequality_constraint_matrices(self, 
                                           component_of_waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row_length_of_g = self.waypoints.number_of_sections*(self.config.order_of_polynomial + 1)
        g = np.empty((0,row_length_of_g))
        h = np.empty((0,))
        # waypoint corridor constraint 
        for section_index in range(self.waypoints.number_of_sections - 1):
            coeff = self.coeff_utils.position.get_lumped_coefficient(0, self.waypoints.section_time[section_index]) # polynomial(t_i - t_{i-1}) < p_i + corridor_width
            new_row_of_g = self.load_row_of_a(row_length_of_g, coeff, section_index)
            g = np.vstack((g, new_row_of_g))
            h = np.append(h, component_of_waypoints[section_index + 1] + self.corridor_width)
            new_row_of_g = self.load_row_of_a(row_length_of_g, -coeff, section_index) # -polynomial(t_i) < -(p_i - corridor_width)
            g = np.vstack((g, new_row_of_g))
            h = np.append(h, -(component_of_waypoints[section_index + 1] - self.corridor_width))
        return g, h

    def load_row_of_a(self, row_length_of_a: int, coeff: np.ndarray, section_index: int) -> np.ndarray:
        row_of_a = np.zeros((1,row_length_of_a))
        col_start, col_end = self.get_column_range_of_polynomial_in_a(section_index)
        # todo: error check on coeff length mismatch
        row_of_a[0][col_start: col_end] = coeff
        return row_of_a

    def get_column_range_of_polynomial_in_a(self, section_index: int) -> tuple[int, int]:
        '''
        find the column indices of a section in matrix A
        a[row][col_start:col_end] gets the corresponding elements of the polynomial
        '''
        col_start = (self.config.order_of_polynomial + 1)*section_index
        col_end = (self.config.order_of_polynomial + 1)*(section_index + 1)
        return col_start, col_end

    def optimize_vectorized_1d_profile(self, component_of_waypoints: np.ndarray,
                                   init_velocity: float, 
                                   init_acceleration: float, 
                                   init_jerk: float, 
                                   end_velocity: float, 
                                   end_acceleration: float,
                                   end_jerk: float) -> tuple[np.ndarray, float]:
        """core optimization method to optimize a 1D profile

        Args:
            component_of_waypoints (np.ndarray): _description_
            init_velocity (float): _description_
            init_acceleration (float): _description_
            init_jerk (float): _description_
            end_velocity (float): _description_
            end_acceleration (float): _description_
            end_jerk (float): _description_

        Returns:
            tuple[np.ndarray, float]: vectorized result, cost
        """
        # arg_x min(1/2*x.T*Q*x + p*x)
        q, p = self.get_cost_matrices()
        a, b = self.get_equality_constraint_matrices(component_of_waypoints,
                                                     init_velocity, 
                                                     init_acceleration,
                                                     init_jerk,
                                                     end_velocity, 
                                                     end_acceleration,
                                                     end_jerk)
        g, h = self.get_inequality_constraint_matrices(component_of_waypoints)
        # g, h = (None, None)
        epsilon = 1e-6
        q_regularized = q + epsilon * np.eye(q.shape[0])

        if which_solver == 'cvxopt':
            # use cvxopt
            # https://cvxopt.org/userguide/coneprog.html#algorithm-parameters
            cvxopt.solvers.options['maxiters'] = 200
            cvxopt.solvers.options['abstol'] = 1e-8
            cvxopt.solvers.options['reltol'] = 1e-7
            cvxopt.solvers.options['show_progress'] = False

            sol = cvxopt.solvers.qp(cvxopt.matrix(q_regularized), 
                                    cvxopt.matrix(p), 
                                    cvxopt.matrix(g), 
                                    cvxopt.matrix(h),  
                                    cvxopt.matrix(a), 
                                    cvxopt.matrix(b))

            # todo: warning of ill solution
            x = np.array(sol['x'])
            return x, (x.T@q@x)[0][0]
        elif which_solver == 'cvxpy':
            # use cvxpy
            n, _ = np.shape(q)
            sol = cvxpy.Variable(n)
            prob = cvxpy.Problem(cvxpy.Minimize((1/2)*cvxpy.quad_form(sol, q_regularized) + p.T @ sol),
                            [a @ sol == b])
            options = {'eps_abs': 1e-8, 'eps_rel' : 1e-7, 'max_iter': 1000}
            prob.solve(solver=cvxpy.OSQP, **options)
            x = sol.value
            return x, (x.T@q@x)
        else:
            solver_options = {'eps':1e-24, 'maxIter': 100, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
            solver = QPFunction(verbose=0, **solver_options)
            solution = solver(torch.tensor(q_regularized), 
                              torch.tensor(p), 
                              torch.tensor(g), 
                              torch.tensor(h), 
                              torch.tensor(a), 
                              torch.tensor(b))
            x = np.array(solution[0])
            return x, (x.T@q@x)
            

    def optimize_1d_profile(self, component_of_waypoints: np.ndarray, 
                        init_velocity: float, 
                        init_acceleration: float, 
                        init_jerk: float, 
                        end_velocity: float, 
                        end_acceleration: float,
                        end_jerk: float) -> tuple[np.ndarray, float]:
        """optimize a 1D profile

        Args:
            component_of_waypoints (np.ndarray): _description_
            init_velocity (float): _description_
            init_acceleration (float): _description_
            end_velocity (float): _description_
            end_acceleration (float): _description_

        Returns:
            tuple[np.ndarray, float]: optimized profile, cost
        """
        packed_result, cost = self.optimize_vectorized_1d_profile(component_of_waypoints,
                                                        init_velocity, 
                                                        init_acceleration,
                                                        init_jerk,
                                                        end_velocity, 
                                                        end_acceleration, 
                                                        end_jerk) 
        # todo check constraint violation

        return self.unpack_vector_to_component_profile(packed_result), cost

    def plot_trajectory(self):
        fig0, axs0 = plt.subplots(1, 1, sharex=True)
        # Use projection='3d' for 3D plots
        axs0 = fig0.add_subplot(111, projection='3d')
        for section in range(self.waypoints.number_of_sections):
            t_samples = np.linspace(self.waypoints.waypoint_time_stamp[section], self.waypoints.waypoint_time_stamp[section+1], 21)
            x = np.empty((0,))
            y = np.empty((0,))
            z = np.empty((0,))
            for t in t_samples:
                x = np.append(x, self.profiles[0][section].sample_polynomial(0, t - self.time_shift[section]))
                y = np.append(y, self.profiles[1][section].sample_polynomial(0, t - self.time_shift[section]))
                z = np.append(z, self.profiles[2][section].sample_polynomial(0, t - self.time_shift[section]))
            axs0.plot3D(x,y,z)
        axs0.plot3D(self.waypoints.coordinates[:, 0],
                   self.waypoints.coordinates[:, 1],
                   self.waypoints.coordinates[:, 2], '.', c='blue', label='Points')
        # for coordinate in self.waypoints.coordinates:
        #     cubic_profile = get_cubic_profile(coordinate, self.corridor_width)
        #     axs.plot3D(cubic_profile[:, 0],
        #                cubic_profile[:, 1],
        #                cubic_profile[:, 2], 
        #                'o', c='green', label='Points')
        axs0.set_title('3D Plot')
        axs0.set_xlabel('X')
        axs0.set_ylabel('Y')
        axs0.set_zlabel('Z')
        axs0.axis('equal')            
        fig1, axs1 = plt.subplots(len(Dim), len(KinematicVars)-1, sharex=True)
        for section in range(self.waypoints.number_of_sections):
            t_samples = np.linspace(self.waypoints.waypoint_time_stamp[section], self.waypoints.waypoint_time_stamp[section+1], 11)
            for kin_var in list(KinematicVars)[:-1]:
                data = np.empty((len(Dim),0))
                for t in t_samples:
                    new_data_col = np.zeros((len(Dim),1))
                    for dim in Dim:
                        new_data_col[dim.value] = self.profiles[dim.value][section].sample_polynomial(kin_var.value, t - self.time_shift[section])
                    data = np.hstack((data, new_data_col))
                for dim in Dim:
                    axs1[dim.value, kin_var.value].plot(t_samples, data[dim.value])
                    axs1[dim.value, kin_var.value].set_ylabel(kin_var.name + dim.name)

def get_cubic_profile(center_coordinate: np.ndarray, half_side_length: float) -> np.ndarray:
    result = np.empty((0,3))
    for i in range(9):
        b_0 = i & 1
        b_1 = (i >> 1) & 1
        b_2 = (i >> 2) & 1
        result = np.vstack((result, 
                            np.array((center_coordinate[0] + (1 if b_0 else -1)*half_side_length,
                                      center_coordinate[1] + (1 if b_1 else -1)*half_side_length,
                                      center_coordinate[2] + (1 if b_2 else -1)*half_side_length))))
    return result

if __name__ == "__main__":
    util_instance = UnitCoeffPolynomialUtils(5)
    print(util_instance.position.induced_coeff)

    # points = (np.array([[0, 0, 0],
    #                 [1, 2, 0],
    #                 [2, 0, 0],
    #                 [4, 5, 0],
    #                 [5, 2, 0]]))
    # time_span = 5.0

    points = np.array([ [ 0.        ,  0.        ,  0.        ],
                        [ 1.98436524,  0.00979175,  1.89768274],
                        [ 1.07734423,  2.24543732,  5.5656538 ],
                        [ 1.11486017,  2.0453859 ,  5.33408252],])
    time_span = 5.0*16
    time_span = 5.0*3

    waypoints = waypoint.Waypoint(points, time_span)
    config = trajectory_config.TrajectoryConfig(order_of_polynomial=7)
    trajectory = TrajectoryGenerator(config, waypoints)
    trajectory.get_trajectory()
    print(trajectory.cost)
    trajectory.plot_trajectory()
    plt.tight_layout()
    plt.show()




