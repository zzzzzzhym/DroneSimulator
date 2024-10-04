import numpy as np
import quaternion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import drone_trajectory as trajectory
import drone_dynamics as dynamics
import drone_controller as controller
import drone_utils as utils
import drone_parameters as params
import drone_plot_utils

class DroneSimulator:
    """
    Controller + Dynamics
    """

    def __init__(self) -> None:
        self.t = 0.0    # simulation epoch time
        self.dt = 0.01  # simulation step to log output
        self.dt_controller = 0.01   # controller cycle time
        self.dt_dynamics = 0.005    # dynamics model cycle time
        # controller steps per log step, must be an integer
        self.cl_ratio = round(self.dt/self.dt_controller)
        # dynamics steps per controller step, must be an integer
        self.dc_ratio = round(self.dt_controller/self.dt_dynamics)
        print("number of controller steps per simulation step: " + str(self.cl_ratio))
        print("number of dynamics model steps per controller step: " +
              str(self.dc_ratio))
        self.sim_trajectory = trajectory.RandomWaypoints(300, True)
        # self.sim_trajectory = trajectory.CircleYZ()
        # self.sim_trajectory = trajectory.Hover()
        self.sim_trajectory.set_init_state()
        self.sim_dynamics = dynamics.DroneDynamics(
            self.sim_trajectory.init_x,
            self.sim_trajectory.init_v,
            self.sim_trajectory.init_pose,
            self.sim_trajectory.init_omega,
            self.dt_dynamics)
        self.sim_controller = controller.DroneController()
        '''
        data to plot
        '''
        self.logger_np = self.initialize_data_logger("np") # contains all np array of result
        self.t_span = []
        self.ani = None

    def convert_buffer_to_array(self, buffer, logger):
        """Assume buffer and the final result dict has the same keys"""
        for key in buffer:
            logger[key] = np.array(buffer[key])

    @staticmethod
    def initialize_data_logger(which_init: str) -> dict:
        logger = {}
        logger["position"] = None
        logger["q"] = None
        logger["v"] = None
        logger["dv"] = None
        logger["e_x"] = None
        logger["e_v"] = None
        logger["e_a"] = None
        logger["e_j"] = None
        logger["e_r"] = None
        logger["e_omega"] = None
        logger["psi_r_rd"] = None
        logger["f_ctrl_input"] = None     # inertial frame
        logger["f_ctrl_input_dot"] = None     # inertial frame
        logger["f_d"] = None    # inertial frame
        logger["f_d_dot"] = None    # inertial frame
        logger["f_d_dot2"] = None    # inertial frame
        logger["torque_ctrl_input"] = None
        logger["x_d"] = None
        logger["v_d"] = None
        logger["x_d_dot2"] = None
        logger["x_d_dot3"] = None
        logger["f_motor"] = None
        logger["f_disturb"] = None  # inertial frame
        logger["f_disturb_est"] = None  # inertial frame
        logger["f_disturb_est_base"] = None  # inertial frame
        logger["torque_disturb"] = None  # inertial frame
        logger["torque_disturb_est"] = None  # inertial frame
        logger["torque_disturb_est_base"] = None  # inertial frame
        logger["omega"] = None  # inertial frame
        logger["omega_dot"] = None  # inertial frame
        logger["omega_desired"] = None  # inertial frame
        logger["pose"] = None
        logger["pose_dot"] = None
        logger["pose_desired"] = None
        logger["pose_desired_dot"] = None
        logger["pose_desired_dot2"] = None
        logger["rotor_spd"] = None
        if which_init == "list":
            for key in logger:
                logger[key] = []
        elif which_init == "np":
            for key in logger:
                logger[key] = None
        else:
            for key in logger:
                logger[key] = None     
        return logger

    def step_simulation(self, t: float):
        t_controller = t
        for i in range(self.cl_ratio):
            t_controller += i*self.dt_controller
            self.sim_trajectory.step_reference_state(self.t)
            self.sim_dynamics.f = self.sim_controller.f
            self.sim_dynamics.torque = self.sim_controller.torque
            t_dynamics = t_controller
            for j in range(self.dc_ratio):
                t_dynamics += j*self.dt_dynamics
                self.sim_dynamics.step_dynamics(t_dynamics)
            self.sim_controller.step_controller(
                self.sim_dynamics, self.sim_trajectory)
            self.t += self.dt_controller

    def run_simulation(self, t_end):
        buffer = self.initialize_data_logger("list")
        self.t_span = np.arange(0.0, t_end + self.dt, self.dt)
        for t in self.t_span:
            if t > 5:
                pass
            self.step_simulation(t)

            buffer["position"].append(self.sim_dynamics.position)
            buffer["q"].append(self.sim_dynamics.q)
            buffer["v"].append(self.sim_dynamics.v)
            buffer["dv"].append(self.sim_dynamics.v_dot)
            buffer["e_x"].append(self.sim_controller.e_x)
            buffer["e_v"].append(self.sim_controller.e_v)
            buffer["e_a"].append(self.sim_controller.e_a)
            buffer["e_j"].append(self.sim_controller.e_j)
            buffer["e_r"].append(self.sim_controller.e_r)
            buffer["e_omega"].append(self.sim_controller.e_omega)
            buffer["psi_r_rd"].append(self.sim_controller.psi_r_rd)
            buffer["f_ctrl_input"].append(-self.sim_dynamics.pose@self.sim_controller.f)
            buffer["f_ctrl_input_dot"].append(-self.sim_dynamics.pose@self.sim_controller.f_dot)
            buffer["f_d"].append(self.sim_controller.f_d)
            buffer["f_d_dot"].append(self.sim_controller.f_d_dot)
            buffer["f_d_dot2"].append(self.sim_controller.f_d_dot2)
            buffer["torque_ctrl_input"].append(self.sim_controller.torque)
            buffer["x_d"].append(self.sim_trajectory.x_d)
            buffer["v_d"].append(self.sim_trajectory.v_d)
            buffer["x_d_dot2"].append(self.sim_trajectory.x_d_dot2)
            buffer["x_d_dot3"].append(self.sim_trajectory.x_d_dot3)
            buffer["f_motor"].append(self.sim_controller.force_motor)
            buffer["f_disturb"].append(self.sim_dynamics.f_disturb)
            buffer["f_disturb_est"].append(self.sim_controller.f_disturb)
            buffer["f_disturb_est_base"].append(self.sim_controller.f_disturb_base)
            buffer["torque_disturb"].append(self.sim_dynamics.torque_disturb)
            buffer["torque_disturb_est"].append(self.sim_controller.torque_disturb)
            buffer["torque_disturb_est_base"].append(self.sim_controller.torque_disturb_base)
            buffer["omega"].append(self.sim_dynamics.pose@self.sim_dynamics.omega)
            buffer["omega_dot"].append(self.sim_dynamics.pose@self.sim_dynamics.omega_dot)
            buffer["omega_desired"].append(self.sim_controller.omega_desired)
            # suspect python pointer related bug occurs if remove *1
            buffer["pose"].append(self.sim_dynamics.pose*1)
            buffer["pose_dot"].append(utils.get_hat_map(
                self.sim_dynamics.pose@self.sim_dynamics.omega)@self.sim_dynamics.pose)
            buffer["pose_desired"].append(self.sim_controller.pose_desired)
            buffer["pose_desired_dot"].append(self.sim_controller.pose_desired_dot)
            buffer["pose_desired_dot2"].append(self.sim_controller.pose_desired_dot2)
            buffer["rotor_spd"].append(self.sim_dynamics.rotor_spd_avg)

        self.convert_buffer_to_array(buffer, self.logger_np)

    def make_plots(self, has_animation=False):
        # plot
        # self.plot_position_and_derivatives()
        # self.plot_omega_and_derivatives()
        # self.plot_pose_and_derivatives()
        ## self.plot_trajectory()
        # self.plot_quaternion()
        self.plot_force_and_torque()
        self.plot_position_tracking_error()
        self.plot_pose_tracking_error()
        self.plot_desired_force()
        self.plot_control_input_force()
        self.plot_pose_desired()
        self.plot_omega_desired()
        self.plot_rotor()
        self.plot_disturbance_force()
        self.plot_3d_trace()
        if has_animation:
            self.ani = self.animate_pose()

    def plot_position_and_derivatives(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_and_derivatives')
        axs[0, 0].plot(self.t_span, self.logger_np["position"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["position"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["position"][:, 2], marker='x')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, self.logger_np["v"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["v"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["v"][:, 2], marker='x')
        t_diff, position_diff = utils.get_signal_derivative(self.t_span, self.logger_np["position"], self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].set_ylabel('v_z')
        axs[0, 2].plot(self.t_span, self.logger_np["dv"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.logger_np["dv"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.logger_np["dv"][:, 2], marker='x')
        t_diff, v_diff = utils.get_signal_derivative(self.t_span, self.logger_np["v"], self.dt)
        axs[0, 2].plot(t_diff, v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, v_diff[:, 2], marker='.')
        a_trace = self.logger_np["f_ctrl_input"]/params.m
        axs[0, 2].plot(self.t_span, a_trace[:, 0])
        axs[1, 2].plot(self.t_span, a_trace[:, 1])
        axs[2, 2].plot(self.t_span, a_trace[:, 2])
        axs[0, 2].set_ylabel('a_x')
        axs[1, 2].set_ylabel('a_y')
        axs[2, 2].set_ylabel('a_z')        
        axs[0, 2].legend(['accel from dynamics', 'accel from dynamics', 'accel predicted from controller'])
        t_diff, a_diff = utils.get_signal_derivative(self.t_span, self.logger_np["dv"], self.dt)
        axs[0, 3].plot(t_diff, a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, a_diff[:, 2], marker='.')
        j_trace = self.logger_np["f_ctrl_input_dot"]/params.m
        axs[0, 3].plot(self.t_span, j_trace[:, 0], marker='x')
        axs[1, 3].plot(self.t_span, j_trace[:, 1], marker='x')
        axs[2, 3].plot(self.t_span, j_trace[:, 2], marker='x')
        axs[0, 3].legend(['jerk from dynamics', 'jerk predicted from controller'])
        axs[0, 2].set_ylabel('j_x')
        axs[1, 2].set_ylabel('j_y')
        axs[2, 2].set_ylabel('j_z')  

    def plot_quaternion(self):
        fig, axs = plt.subplots(5, 1, sharex=True)
        fig.suptitle('quaternion')
        axs[0].plot(self.t_span, self.logger_np["q"][:, 0], marker='x')
        axs[1].plot(self.t_span, self.logger_np["q"][:, 1], marker='x')
        axs[2].plot(self.t_span, self.logger_np["q"][:, 2], marker='x')
        axs[3].plot(self.t_span, self.logger_np["q"][:, 3], marker='x')
        axs[0].set_ylabel('q0')
        axs[1].set_ylabel('q1')
        axs[2].set_ylabel('q2')
        axs[3].set_ylabel('q3')
        axs[4].plot(self.t_span, self.logger_np["q"][:, 0]**2 +
                    self.logger_np["q"][:, 1]**2 +
                    self.logger_np["q"][:, 2]**2 +
                    self.logger_np["q"][:, 3]**2, marker='x')
        axs[4].set_ylabel('norm of q (should be 1)')

    def plot_omega_and_derivatives(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('omega_and_derivatives')
        axs[0, 0].plot(self.t_span, self.logger_np["omega"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["omega"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["omega"][:, 2], marker='x')
        axs[0, 0].set_ylabel("omega x")
        axs[1, 0].set_ylabel("omega y")
        axs[2, 0].set_ylabel("omega z")
        axs[0, 1].plot(self.t_span, self.logger_np["omega_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["omega_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["omega_dot"][:, 2], marker='x')
        t_diff, omega_diff = utils.get_signal_derivative(self.t_span, self.logger_np["omega"], self.dt)
        axs[0, 1].plot(t_diff, omega_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, omega_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, omega_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel("omega dot x")
        axs[1, 1].set_ylabel("omega dot y")
        axs[2, 1].set_ylabel("omega dot z")

    def plot_pose_and_derivatives(self):
        fig1, axs1 = plt.subplots(3, 6, sharex=True)
        fig1.suptitle('pose (left) and pose_dot (right)')
        axs1[0, 0].plot(self.t_span, self.logger_np["pose"][:, 0, 0], marker='x')
        axs1[1, 0].plot(self.t_span, self.logger_np["pose"][:, 1, 0], marker='x')
        axs1[2, 0].plot(self.t_span, self.logger_np["pose"][:, 2, 0], marker='x')
        axs1[0, 1].plot(self.t_span, self.logger_np["pose"][:, 0, 1], marker='x')
        axs1[1, 1].plot(self.t_span, self.logger_np["pose"][:, 1, 1], marker='x')
        axs1[2, 1].plot(self.t_span, self.logger_np["pose"][:, 2, 1], marker='x')
        axs1[0, 2].plot(self.t_span, self.logger_np["pose"][:, 0, 2], marker='x')
        axs1[1, 2].plot(self.t_span, self.logger_np["pose"][:, 1, 2], marker='x')
        axs1[2, 2].plot(self.t_span, self.logger_np["pose"][:, 2, 2], marker='x')

        axs1[0, 3].plot(self.t_span, self.logger_np["pose_dot"][:, 0, 0], marker='x')
        axs1[1, 3].plot(self.t_span, self.logger_np["pose_dot"][:, 1, 0], marker='x')
        axs1[2, 3].plot(self.t_span, self.logger_np["pose_dot"][:, 2, 0], marker='x')
        axs1[0, 4].plot(self.t_span, self.logger_np["pose_dot"][:, 0, 1], marker='x')
        axs1[1, 4].plot(self.t_span, self.logger_np["pose_dot"][:, 1, 1], marker='x')
        axs1[2, 4].plot(self.t_span, self.logger_np["pose_dot"][:, 2, 1], marker='x')
        axs1[0, 5].plot(self.t_span, self.logger_np["pose_dot"][:, 0, 2], marker='x')
        axs1[1, 5].plot(self.t_span, self.logger_np["pose_dot"][:, 1, 2], marker='x')
        axs1[2, 5].plot(self.t_span, self.logger_np["pose_dot"][:, 2, 2], marker='x')
        t_diff, pose_diff = utils.get_signal_derivative(self.t_span, self.logger_np["pose"], self.dt)
        axs1[0, 3].plot(t_diff, pose_diff[:, 0, 0], marker='.')
        axs1[1, 3].plot(t_diff, pose_diff[:, 1, 0], marker='.')
        axs1[2, 3].plot(t_diff, pose_diff[:, 2, 0], marker='.')
        axs1[0, 4].plot(t_diff, pose_diff[:, 0, 1], marker='.')
        axs1[1, 4].plot(t_diff, pose_diff[:, 1, 1], marker='.')
        axs1[2, 4].plot(t_diff, pose_diff[:, 2, 1], marker='.')
        axs1[0, 5].plot(t_diff, pose_diff[:, 0, 2], marker='.')
        axs1[1, 5].plot(t_diff, pose_diff[:, 1, 2], marker='.')
        axs1[2, 5].plot(t_diff, pose_diff[:, 2, 2], marker='.')

    def plot_trajectory(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('trajectory')
        axs[0, 0].plot(self.t_span, self.logger_np["x_d"][:, 0], marker='x')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].plot(self.t_span, self.logger_np["x_d"][:, 1], marker='x')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].plot(self.t_span, self.logger_np["x_d"][:, 2], marker='x')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, self.logger_np["v_d"][:, 0], marker='x')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].plot(self.t_span, self.logger_np["v_d"][:, 1], marker='x')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].plot(self.t_span, self.logger_np["v_d"][:, 2], marker='x')
        axs[2, 1].set_ylabel('v_z')
        t_diff, x_d_diff = utils.get_signal_derivative(self.t_span, self.logger_np["x_d"], self.dt)
        axs[0, 1].plot(t_diff, x_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, x_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, x_d_diff[:, 2], marker='.')

        axs[0, 2].plot(self.t_span, self.logger_np["x_d_dot2"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.logger_np["x_d_dot2"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.logger_np["x_d_dot2"][:, 2], marker='x')
        t_diff, v_d_diff = utils.get_signal_derivative(self.t_span, self.logger_np["v_d"], self.dt)
        axs[0, 2].plot(t_diff, v_d_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, v_d_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, v_d_diff[:, 2], marker='.')

        axs[0, 3].plot(self.t_span, self.logger_np["x_d_dot3"][:, 0], marker='x')
        axs[1, 3].plot(self.t_span, self.logger_np["x_d_dot3"][:, 1], marker='x')
        axs[2, 3].plot(self.t_span, self.logger_np["x_d_dot3"][:, 2], marker='x')
        t_diff, x_d_dot2_diff = utils.get_signal_derivative(self.t_span, self.logger_np["x_d_dot2"], self.dt)
        axs[0, 3].plot(t_diff, x_d_dot2_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, x_d_dot2_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, x_d_dot2_diff[:, 2], marker='.')

    def plot_force_and_torque(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('force_and_torque')
        axs[0, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 0], marker='x')
        axs[0, 0].plot(self.t_span, self.logger_np["f_d"][:, 0])
        axs[1, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 1], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["f_d"][:, 1])
        axs[2, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 2], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["f_d"][:, 2])
        axs[0, 0].set_ylabel('f_x')
        axs[1, 0].set_ylabel('f_y')
        axs[2, 0].set_ylabel('f_z')
        axs[0, 0].legend(['f_ctrl_input', "f_d"])
        axs[0, 1].plot(self.t_span, self.logger_np["torque_ctrl_input"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["torque_ctrl_input"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["torque_ctrl_input"][:, 2], marker='x')
        axs[0, 1].set_ylabel('M_x')
        axs[1, 1].set_ylabel('M_y')
        axs[2, 1].set_ylabel('M_z')

    def plot_position_tracking_error(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_tracking_error')
        axs[0, 0].set_title("position error x")
        axs[1, 0].set_title("position error y")
        axs[2, 0].set_title("position error z")
        axs[0, 0].plot(self.t_span, self.logger_np["e_x"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["e_x"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["e_x"][:, 2], marker='x')

        axs[0, 1].set_title("v error x")
        axs[1, 1].set_title("v error y")
        axs[2, 1].set_title("v error z")
        axs[0, 1].plot(self.t_span, self.logger_np["e_v"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["e_v"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["e_v"][:, 2], marker='x')
        t_diff, e_x_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["e_x"], self.dt)
        axs[0, 1].plot(t_diff, e_x_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, e_x_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, e_x_diff[:, 2], marker='.')
        t_diff, position_diff = utils.get_signal_derivative(self.t_span, self.logger_np["position"], self.dt)
        t_diff, x_d_diff = utils.get_signal_derivative(self.t_span, self.logger_np["x_d"], self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0] - x_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1] - x_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2] - x_d_diff[:, 2], marker='.')
        axs[0, 1].plot(self.t_span, self.logger_np["v"][:, 0] - self.logger_np["v_d"][:, 0])
        axs[1, 1].plot(self.t_span, self.logger_np["v"][:, 1] - self.logger_np["v_d"][:, 1])
        axs[2, 1].plot(self.t_span, self.logger_np["v"][:, 2] - self.logger_np["v_d"][:, 2])

        axs[0, 2].set_title("a error x")
        axs[1, 2].set_title("a error y")
        axs[2, 2].set_title("a error z")
        axs[0, 2].plot(self.t_span, self.logger_np["e_a"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.logger_np["e_a"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.logger_np["e_a"][:, 2], marker='x')
        t_diff, e_v_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["e_v"], self.dt)
        axs[0, 2].plot(t_diff, e_v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, e_v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, e_v_diff[:, 2], marker='.')

        axs[0, 3].set_title("j error x")
        axs[1, 3].set_title("j error y")
        axs[2, 3].set_title("j error z")
        axs[0, 3].plot(self.t_span, self.logger_np["e_j"][:, 0], marker='x')
        axs[1, 3].plot(self.t_span, self.logger_np["e_j"][:, 1], marker='x')
        axs[2, 3].plot(self.t_span, self.logger_np["e_j"][:, 2], marker='x')
        t_diff, e_a_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["e_a"], self.dt)
        axs[0, 3].plot(t_diff, e_a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, e_a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, e_a_diff[:, 2], marker='.')
        j_trace = self.logger_np["f_ctrl_input_dot"]/params.m
        axs[0, 3].plot(self.t_span, j_trace[:, 0] - self.logger_np["x_d_dot3"][:, 0])
        axs[1, 3].plot(self.t_span, j_trace[:, 1] - self.logger_np["x_d_dot3"][:, 1])
        axs[2, 3].plot(self.t_span, j_trace[:, 2] - self.logger_np["x_d_dot3"][:, 2])
        '''
        create one time step misalignment in jerk
        '''
        # axs[0, 3].plot(t_diff, j_trace[:-1, 0] - self.x_d_dot3_trace[1:, 0])
        # axs[1, 3].plot(t_diff, j_trace[:-1, 1] - self.x_d_dot3_trace[1:, 1])
        # axs[2, 3].plot(t_diff, j_trace[:-1, 2] - self.x_d_dot3_trace[1:, 2])

    def plot_pose_tracking_error(self):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('pose_tracking_error')
        axs[0, 0].plot(self.t_span, self.logger_np["e_r"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["e_r"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["e_r"][:, 2], marker='x')
        axs[0, 0].set_ylabel("e_r_x")
        axs[1, 0].set_ylabel("e_r_y")
        axs[2, 0].set_ylabel("e_r_z")        
        axs[0, 1].plot(self.t_span, self.logger_np["e_omega"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["e_omega"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["e_omega"][:, 2], marker='x')
        axs[0, 1].set_ylabel("e_omega_x")
        axs[1, 1].set_ylabel("e_omega_y")
        axs[2, 1].set_ylabel("e_omega_z")
        axs[0, 2].plot(self.t_span, self.logger_np["psi_r_rd"], marker='x')
        axs[0, 2].set_ylabel("psi_r_rd")

    def plot_desired_force(self):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('f_desired, f_desired_dot, f_desired_dot2')
        axs[0, 0].plot(self.t_span, self.logger_np["f_d"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["f_d"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["f_d"][:, 2], marker='x')
        axs[0, 0].set_ylabel("f_d_x")
        axs[1, 0].set_ylabel("f_d_y")
        axs[2, 0].set_ylabel("f_d_z")
        axs[0, 1].plot(self.t_span, self.logger_np["f_d_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["f_d_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["f_d_dot"][:, 2], marker='x')
        t_diff, f_d_diff = utils.get_signal_derivative(self.t_span, self.logger_np["f_d"], self.dt)
        axs[0, 1].plot(t_diff, f_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, f_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, f_d_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel("f_d_dot_x")
        axs[1, 1].set_ylabel("f_d_dot_y")
        axs[2, 1].set_ylabel("f_d_dot_z")
        axs[0, 2].plot(self.t_span, self.logger_np["f_d_dot2"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.logger_np["f_d_dot2"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.logger_np["f_d_dot2"][:, 2], marker='x')
        t_diff, f_d_dot_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["f_d_dot"], self.dt)
        axs[0, 2].plot(t_diff, f_d_dot_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, f_d_dot_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, f_d_dot_diff[:, 2], marker='.')
        axs[0, 2].set_ylabel("f_d_dot2_x")
        axs[1, 2].set_ylabel("f_d_dot2_y")
        axs[2, 2].set_ylabel("f_d_dot2_z")        

    def plot_control_input_force(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('force, force dot')
        axs[0, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.logger_np["f_ctrl_input"][:, 2], marker='x')
        axs[0, 0].set_ylabel("f_x")
        axs[1, 0].set_ylabel("f_y")
        axs[2, 0].set_ylabel("f_z")
        axs[0, 1].plot(self.t_span, self.logger_np["f_ctrl_input_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.logger_np["f_ctrl_input_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.logger_np["f_ctrl_input_dot"][:, 2], marker='x')
        t_diff, f_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["f_ctrl_input"], self.dt)
        axs[0, 1].plot(t_diff, f_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, f_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, f_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel("f_dot_x")
        axs[1, 1].set_ylabel("f_dot_y")
        axs[2, 1].set_ylabel("f_dot_z")        

    def plot_pose_desired(self):
        fig1, axs1 = plt.subplots(3, 9, sharex=True)
        fig1.suptitle('pose_desired, pose_desired_dot, pose_desired_dot2')
        axs1[0, 0].plot(self.t_span, self.logger_np["pose_desired"][:, 0, 0], marker='x')
        axs1[1, 0].plot(self.t_span, self.logger_np["pose_desired"][:, 1, 0], marker='x')
        axs1[2, 0].plot(self.t_span, self.logger_np["pose_desired"][:, 2, 0], marker='x')
        axs1[0, 1].plot(self.t_span, self.logger_np["pose_desired"][:, 0, 1], marker='x')
        axs1[1, 1].plot(self.t_span, self.logger_np["pose_desired"][:, 1, 1], marker='x')
        axs1[2, 1].plot(self.t_span, self.logger_np["pose_desired"][:, 2, 1], marker='x')
        axs1[0, 2].plot(self.t_span, self.logger_np["pose_desired"][:, 0, 2], marker='x')
        axs1[1, 2].plot(self.t_span, self.logger_np["pose_desired"][:, 1, 2], marker='x')
        axs1[2, 2].plot(self.t_span, self.logger_np["pose_desired"][:, 2, 2], marker='x')
        axs1[0, 3].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 0, 0], marker='x')
        axs1[1, 3].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 1, 0], marker='x')
        axs1[2, 3].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 2, 0], marker='x')
        axs1[0, 4].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 0, 1], marker='x')
        axs1[1, 4].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 1, 1], marker='x')
        axs1[2, 4].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 2, 1], marker='x')
        axs1[0, 5].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 0, 2], marker='x')
        axs1[1, 5].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 1, 2], marker='x')
        axs1[2, 5].plot(self.t_span, self.logger_np["pose_desired_dot"][:, 2, 2], marker='x')
        t_diff, pose_desired_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["pose_desired"], self.dt)
        axs1[0, 3].plot(t_diff, pose_desired_diff[:, 0, 0], marker='.')
        axs1[1, 3].plot(t_diff, pose_desired_diff[:, 1, 0], marker='.')
        axs1[2, 3].plot(t_diff, pose_desired_diff[:, 2, 0], marker='.')
        axs1[0, 4].plot(t_diff, pose_desired_diff[:, 0, 1], marker='.')
        axs1[1, 4].plot(t_diff, pose_desired_diff[:, 1, 1], marker='.')
        axs1[2, 4].plot(t_diff, pose_desired_diff[:, 2, 1], marker='.')
        axs1[0, 5].plot(t_diff, pose_desired_diff[:, 0, 2], marker='.')
        axs1[1, 5].plot(t_diff, pose_desired_diff[:, 1, 2], marker='.')
        axs1[2, 5].plot(t_diff, pose_desired_diff[:, 2, 2], marker='.')

        axs1[0, 6].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 0, 0], marker='x')
        axs1[1, 6].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 1, 0], marker='x')
        axs1[2, 6].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 2, 0], marker='x')
        axs1[0, 7].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 0, 1], marker='x')
        axs1[1, 7].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 1, 1], marker='x')
        axs1[2, 7].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 2, 1], marker='x')
        axs1[0, 8].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 0, 2], marker='x')
        axs1[1, 8].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 1, 2], marker='x')
        axs1[2, 8].plot(self.t_span, self.logger_np["pose_desired_dot2"][:, 2, 2], marker='x')
        t_diff, pose_desired_dot_diff = utils.get_signal_derivative(
            self.t_span, self.logger_np["pose_desired_dot"], self.dt)
        axs1[0, 6].plot(t_diff, pose_desired_dot_diff[:, 0, 0], marker='.')
        axs1[1, 6].plot(t_diff, pose_desired_dot_diff[:, 1, 0], marker='.')
        axs1[2, 6].plot(t_diff, pose_desired_dot_diff[:, 2, 0], marker='.')
        axs1[0, 7].plot(t_diff, pose_desired_dot_diff[:, 0, 1], marker='.')
        axs1[1, 7].plot(t_diff, pose_desired_dot_diff[:, 1, 1], marker='.')
        axs1[2, 7].plot(t_diff, pose_desired_dot_diff[:, 2, 1], marker='.')
        axs1[0, 8].plot(t_diff, pose_desired_dot_diff[:, 0, 2], marker='.')
        axs1[1, 8].plot(t_diff, pose_desired_dot_diff[:, 1, 2], marker='.')
        axs1[2, 8].plot(t_diff, pose_desired_dot_diff[:, 2, 2], marker='.')

    def plot_disturbance_force(self):
        fig, axs = plt.subplots(4, 2, sharex=True)
        fig.suptitle('disturbance')
        f_norm = np.sqrt(
            self.logger_np["f_disturb"][:, 0]**2 + self.logger_np["f_disturb"][:, 1]**2 + self.logger_np["f_disturb"][:, 2]**2)
        f_est_norm = np.sqrt(
            self.logger_np["f_disturb_est"][:, 0]**2 + self.logger_np["f_disturb_est"][:, 1]**2 + self.logger_np["f_disturb_est"][:, 2]**2)
        f_base_norm = np.sqrt(
            self.logger_np["f_disturb_est_base"][:, 0]**2 + self.logger_np["f_disturb_est_base"][:, 1]**2 + self.logger_np["f_disturb_est_base"][:, 2]**2)
        torque_norm = np.sqrt(
            self.logger_np["torque_disturb"][:, 0]**2 + self.logger_np["torque_disturb"][:, 1]**2 + self.logger_np["torque_disturb"][:, 2]**2)
        torque_est_norm = np.sqrt(
            self.logger_np["torque_disturb_est"][:, 0]**2 + self.logger_np["torque_disturb_est"][:, 1]**2 + self.logger_np["torque_disturb_est"][:, 2]**2)
        torque_base_norm = np.sqrt(
            self.logger_np["torque_disturb_est_base"][:, 0]**2 + self.logger_np["torque_disturb_est_base"][:, 1]**2 + self.logger_np["torque_disturb_est_base"][:, 2]**2)
        
        axs[0, 0].plot(self.t_span, self.logger_np["f_disturb"][:, 0], marker='.')
        axs[0, 0].plot(self.t_span, self.logger_np["f_disturb_est"][:, 0], marker='.')
        axs[0, 0].plot(self.t_span, self.logger_np["f_disturb_est_base"][:, 0], marker='.')
        axs[0, 0].set_ylabel("f_x")
        axs[0, 0].legend(['f_disturb', 'f_disturb_est', 'f_disturb_est_base'])
        
        axs[1, 0].plot(self.t_span, self.logger_np["f_disturb"][:, 1], marker='.')
        axs[1, 0].plot(self.t_span, self.logger_np["f_disturb_est"][:, 1], marker='.')
        axs[1, 0].plot(self.t_span, self.logger_np["f_disturb_est_base"][:, 1], marker='.')
        axs[1, 0].set_ylabel("f_y")
        axs[1, 0].legend(['f_disturb', 'f_disturb_est', 'f_disturb_est_base'])
        
        axs[2, 0].plot(self.t_span, self.logger_np["f_disturb"][:, 2], marker='.')
        axs[2, 0].plot(self.t_span, self.logger_np["f_disturb_est"][:, 2], marker='.')
        axs[2, 0].plot(self.t_span, self.logger_np["f_disturb_est_base"][:, 2], marker='.')
        axs[2, 0].set_ylabel("f_z")
        axs[2, 0].legend(['f_disturb', 'f_disturb_est', 'f_disturb_est_base'])
        
        axs[3, 0].plot(self.t_span, f_norm, marker='.')
        axs[3, 0].plot(self.t_span, f_est_norm, marker='.')
        axs[3, 0].plot(self.t_span, f_base_norm, marker='.')
        axs[3, 0].set_ylabel("f_norm")
        
        axs[0, 1].plot(self.t_span, self.logger_np["torque_disturb"][:, 0], marker='.')
        axs[0, 1].plot(self.t_span, self.logger_np["torque_disturb_est"][:, 0], marker='.')
        axs[0, 1].plot(self.t_span, self.logger_np["torque_disturb_est_base"][:, 0], marker='.')
        axs[0, 1].set_ylabel("torque_x")
        
        axs[1, 1].plot(self.t_span, self.logger_np["torque_disturb"][:, 1], marker='.')
        axs[1, 1].plot(self.t_span, self.logger_np["torque_disturb_est"][:, 1], marker='.')
        axs[1, 1].plot(self.t_span, self.logger_np["torque_disturb_est_base"][:, 1], marker='.')
        axs[1, 1].set_ylabel("torque_y")
        
        axs[2, 1].plot(self.t_span, self.logger_np["torque_disturb"][:, 2], marker='.')
        axs[2, 1].plot(self.t_span, self.logger_np["torque_disturb_est"][:, 2], marker='.')
        axs[2, 1].plot(self.t_span, self.logger_np["torque_disturb_est_base"][:, 2], marker='.')
        axs[2, 1].set_ylabel("torque_z")
        
        axs[3, 1].plot(self.t_span, torque_norm, marker='.')
        axs[3, 1].plot(self.t_span, torque_est_norm, marker='.')
        axs[3, 1].plot(self.t_span, torque_base_norm, marker='.')
        axs[3, 1].set_ylabel("torque_norm")

    def plot_rotor(self):
        fig, axs = plt.subplots(5, 1, sharex=True)
        fig.suptitle('rotor force, rotor speed')
        axs[0].plot(self.t_span, self.logger_np["f_motor"][:, 0], marker='x')
        axs[1].plot(self.t_span, self.logger_np["f_motor"][:, 1], marker='x')
        axs[2].plot(self.t_span, self.logger_np["f_motor"][:, 2], marker='x')
        axs[3].plot(self.t_span, self.logger_np["f_motor"][:, 3], marker='x')
        axs[0].set_ylabel("f_motor_0")
        axs[1].set_ylabel("f_motor_1")
        axs[2].set_ylabel("f_motor_2")
        axs[3].set_ylabel("f_motor_3")
        axs[4].plot(self.t_span, self.logger_np["rotor_spd"])
        axs[4].set_ylabel("rotor_spd [RPM]")

    def plot_omega_desired(self):
        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.suptitle('omega_desired vs omega')
        axs[0].plot(self.t_span, self.logger_np["omega_desired"][:, 0], marker='x')
        axs[0].plot(self.t_span, self.logger_np["omega"][:, 0], marker='.')
        axs[1].plot(self.t_span, self.logger_np["omega_desired"][:, 1], marker='x')
        axs[1].plot(self.t_span, self.logger_np["omega"][:, 1], marker='.')
        axs[2].plot(self.t_span, self.logger_np["omega_desired"][:, 2], marker='x')
        axs[2].plot(self.t_span, self.logger_np["omega"][:, 2], marker='.')
        axs[0].legend(['omega desired', 'omega'])

    def plot_3d_trace(self):
        fig9, axs9 = plt.subplots(1, 1, sharex=True)
        axs9 = fig9.add_subplot(111, projection='3d')
        axs9.plot3D(self.logger_np["x_d"][:, 0],
                    self.logger_np["x_d"][:, 1],
                    self.logger_np["x_d"][:, 2], '.', c='blue', label='Points')
        axs9.plot3D(self.logger_np["position"][:, 0],
                    self.logger_np["position"][:, 1],
                    self.logger_np["position"][:, 2], 'green')
        b1b2, b3 = drone_plot_utils.generate_drone_profile(
            self.sim_dynamics.position, self.sim_dynamics.pose)
        axs9.plot3D(b1b2[:, 0],
                    b1b2[:, 1],
                    b1b2[:, 2], 'red')
        axs9.plot3D(b3[:, 0],
                    b3[:, 1],
                    b3[:, 2], 'red')
        b1b2, b3 = drone_plot_utils.generate_drone_profile(
            self.sim_dynamics.position, self.sim_controller.pose_desired)
        axs9.plot3D(b1b2[:, 0],
                    b1b2[:, 1],
                    b1b2[:, 2], 'orange')
        axs9.plot3D(b3[:, 0],
                    b3[:, 1],
                    b3[:, 2], 'orange')
        b1 = np.vstack((self.sim_trajectory.x_d,
                        self.sim_trajectory.x_d + 0.5*self.sim_trajectory.b_1d))
        axs9.plot3D(b1[:, 0],
                    b1[:, 1],
                    b1[:, 2], 'purple')
        b1 = np.vstack((self.sim_dynamics.position,
                        self.sim_dynamics.position + 0.5*self.sim_trajectory.b_1d))
        axs9.plot3D(b1[:, 0],
                    b1[:, 1],
                    b1[:, 2], 'purple')
        axs9.set_title('3D Plot')
        axs9.set_xlabel('X')
        axs9.set_ylabel('Y')
        axs9.set_zlabel('Z')
        axs9.axis('equal')
        axs9.invert_zaxis()
        axs9.invert_yaxis()

    def animate_pose(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(self.logger_np["x_d"][:, 0],
                    self.logger_np["x_d"][:, 1],
                    self.logger_np["x_d"][:, 2], '.', c='blue', label='Points')
        ax.plot3D(self.logger_np["position"][:, 0],
                    self.logger_np["position"][:, 1],
                    self.logger_np["position"][:, 2], 'green')     
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')           
        ax.invert_zaxis()
        ax.invert_yaxis()   
        text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)   
        delta_frame = 100
        pos = self.logger_np["position"][::delta_frame, :]
        pose_desire = self.logger_np["pose_desired"][::delta_frame, :, :]
        pose = self.logger_np["pose"][::delta_frame, :, :]
        t = self.t_span[::delta_frame]
        ani = FuncAnimation(fig, lambda n: update_frame(ax, text, pos[n, :], pose_desire[n, :, :], pose[n, :, :], t[n]), frames=int(len(self.logger_np["pose_desired"])/delta_frame), interval=50, blit=True, repeat=False) 
        return ani

def update_frame(ax: plt.Axes, text, pos: np.ndarray, pose_ref: np.ndarray, pose_meas: np.ndarray, t: float):
    b1b2, b3 = drone_plot_utils.generate_drone_profile(pos, pose_ref)
    ax.plot(b1b2[:, 0],
            b1b2[:, 1],
            b1b2[:, 2], 'orange')
    ax.plot(b3[:, 0],
            b3[:, 1],
            b3[:, 2], 'orange')
    b1b2, b3 = drone_plot_utils.generate_drone_profile(pos, pose_meas)
    ax.plot(b1b2[:, 0],
            b1b2[:, 1],
            b1b2[:, 2], 'red')
    ax.plot(b3[:, 0],
            b3[:, 1],
            b3[:, 2], 'red')
    text.set_text(f'timestamp: {t}')
    return ax, text


if __name__ == "__main__":
    sim_test = DroneSimulator()
    sim_test.run_simulation(30)
    sim_test.make_plots(True)
    plt.show()
