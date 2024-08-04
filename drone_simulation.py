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
        self.sim_trajectory = trajectory.RandomWaypoints(300)
        # self.sim_trajectory = trajectory.SpiralAndSpin()
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
        self.position_trace = np.empty((0, 3))
        self.q_trace = np.empty((0, 4))
        self.v_trace = np.empty((0, 3))
        self.dv_trace = np.empty((0, 3))
        self.e_x_trace = np.empty((0, 3))
        self.e_v_trace = np.empty((0, 3))
        self.e_a_trace = np.empty((0, 3))
        self.e_j_trace = np.empty((0, 3))
        self.e_r_trace = np.empty((0, 3))
        self.e_omega_trace = np.empty((0, 3))
        self.psi_r_rd_trace = np.empty((0, 1))
        self.f_trace = np.empty((0, 3))     # inertial frame
        self.f_dot_trace = np.empty((0, 3))     # inertial frame
        self.f_d_trace = np.empty((0, 3))    # inertial frame
        self.f_d_dot_trace = np.empty((0, 3))    # inertial frame
        self.f_d_dot2_trace = np.empty((0, 3))    # inertial frame
        self.torque_trace = np.empty((0, 3))
        self.x_d_trace = np.empty((0, 3))
        self.v_d_trace = np.empty((0, 3))
        self.x_d_dot2_trace = np.empty((0, 3))
        self.x_d_dot3_trace = np.empty((0, 3))
        self.f_motor_trace = np.empty((0, 4))
        self.f_disturb_trace = np.empty((0, 3))  # inertial frame
        self.omega_trace = np.empty((0, 3))  # inertial frame
        self.omega_dot_trace = np.empty((0, 3))  # inertial frame
        self.omega_desired_trace = np.empty((0, 3))  # inertial frame
        self.pose_trace = []
        self.pose_dot_trace = []
        self.pose_desired_trace = []
        self.pose_desired_dot_trace = []
        self.pose_desired_dot2_trace = []
        self.t_span = []
        self.ani = None

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
        self.t_span = np.arange(0.0, t_end + self.dt, self.dt)
        for t in self.t_span:
            if t > 0.5:
                pass
            self.step_simulation(t)
            self.position_trace = np.vstack(
                (self.position_trace, self.sim_dynamics.position))
            self.q_trace = np.vstack(
                (self.q_trace, quaternion.as_float_array(self.sim_dynamics.q)))
            self.v_trace = np.vstack((self.v_trace, self.sim_dynamics.v))
            self.dv_trace = np.vstack((self.dv_trace, self.sim_dynamics.v_dot))
            self.e_x_trace = np.vstack(
                (self.e_x_trace, self.sim_controller.e_x))
            self.e_v_trace = np.vstack(
                (self.e_v_trace, self.sim_controller.e_v))
            self.e_a_trace = np.vstack(
                (self.e_a_trace, self.sim_controller.e_a))
            self.e_j_trace = np.vstack(
                (self.e_j_trace, self.sim_controller.e_j))
            self.e_r_trace = np.vstack(
                (self.e_r_trace, self.sim_controller.e_r))
            self.e_omega_trace = np.vstack(
                (self.e_omega_trace, self.sim_controller.e_omega))
            self.psi_r_rd_trace = np.vstack(
                (self.psi_r_rd_trace, self.sim_controller.psi_r_rd))
            self.f_trace = np.vstack(
                (self.f_trace, -self.sim_dynamics.pose@self.sim_controller.f))
            self.f_dot_trace = np.vstack(
                (self.f_dot_trace, -self.sim_dynamics.pose@self.sim_controller.f_dot))
            self.f_d_trace = np.vstack(
                (self.f_d_trace, self.sim_controller.f_d))
            self.f_d_dot_trace = np.vstack(
                (self.f_d_dot_trace, self.sim_controller.f_d_dot))
            self.f_d_dot2_trace = np.vstack(
                (self.f_d_dot2_trace, self.sim_controller.f_d_dot2))
            self.torque_trace = np.vstack(
                (self.torque_trace, self.sim_controller.torque))
            self.x_d_trace = np.vstack(
                (self.x_d_trace, self.sim_trajectory.x_d))
            self.v_d_trace = np.vstack(
                (self.v_d_trace, self.sim_trajectory.v_d))
            self.x_d_dot2_trace = np.vstack(
                (self.x_d_dot2_trace, self.sim_trajectory.x_d_dot2))
            self.x_d_dot3_trace = np.vstack(
                (self.x_d_dot3_trace, self.sim_trajectory.x_d_dot3))
            self.f_motor_trace = np.vstack(
                (self.f_motor_trace, self.sim_controller.force_motor))
            self.f_disturb_trace = np.vstack(
                (self.f_disturb_trace, self.sim_dynamics.f_disturb))
            self.omega_trace = np.vstack(
                (self.omega_trace, self.sim_dynamics.pose@self.sim_dynamics.omega))
            self.omega_dot_trace = np.vstack(
                (self.omega_dot_trace, self.sim_dynamics.pose@self.sim_dynamics.omega_dot))
            self.omega_desired_trace = np.vstack(
                (self.omega_desired_trace, self.sim_controller.omega_desired))
            # suspect python pointer related bug occurs if remove *1
            self.pose_trace.append(self.sim_dynamics.pose*1)
            self.pose_dot_trace.append(utils.get_hat_map(
                self.sim_dynamics.pose@self.sim_dynamics.omega)@self.sim_dynamics.pose)
            self.pose_desired_trace.append(self.sim_controller.pose_desired)
            self.pose_desired_dot_trace.append(
                self.sim_controller.pose_desired_dot)
            self.pose_desired_dot2_trace.append(
                self.sim_controller.pose_desired_dot2)
        self.pose_trace = np.array(self.pose_trace)
        self.pose_dot_trace = np.array(self.pose_dot_trace)
        self.pose_desired_trace = np.array(self.pose_desired_trace)
        self.pose_desired_dot_trace = np.array(self.pose_desired_dot_trace)
        self.pose_desired_dot2_trace = np.array(self.pose_desired_dot2_trace)


    def make_plot(self):
        # plot
        self.plot_position_and_derivatives()
        self.plot_omega_and_derivatives()
        self.plot_pose_and_derivatives()
        # self.plot_trajectory()
        self.plot_force_and_torque()
        self.plot_position_tracking_error()
        self.plot_pose_tracking_error()
        self.plot_desired_force()
        self.plot_force()
        self.plot_pose_desired()
        self.plot_omega_desired()
        self.plot_motor_force()
        self.plot_disturbance_force()
        self.plot_3d_trace()
        self.ani = self.animate_pose()

    def plot_position_and_derivatives(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_and_derivatives')
        axs[0, 0].plot(self.t_span, self.position_trace[:, 0], marker='x')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].plot(self.t_span, self.position_trace[:, 1], marker='x')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].plot(self.t_span, self.position_trace[:, 2], marker='x')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, self.v_trace[:, 0], marker='x')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].plot(self.t_span, self.v_trace[:, 1], marker='x')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].plot(self.t_span, self.v_trace[:, 2], marker='x')
        axs[2, 1].set_ylabel('v_z')
        t_diff, position_diff = utils.get_signal_derivative(
            self.t_span, self.position_trace, self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2], marker='.')

        axs[0, 2].plot(self.t_span, self.dv_trace[:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.dv_trace[:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.dv_trace[:, 2], marker='x')
        t_diff, v_diff = utils.get_signal_derivative(
            self.t_span, self.v_trace, self.dt)
        axs[0, 2].plot(t_diff, v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, v_diff[:, 2], marker='.')
        a_trace = self.f_trace/params.m
        axs[0, 2].plot(self.t_span, a_trace[:, 0])
        axs[1, 2].plot(self.t_span, a_trace[:, 1])
        axs[2, 2].plot(self.t_span, a_trace[:, 2])
        axs[0, 2].legend(
            ['accel from dynamics', 'accel from dynamics', 'accel from controller'])

        j_trace = self.f_dot_trace/params.m
        axs[0, 3].plot(self.t_span, j_trace[:, 0], marker='x')
        axs[1, 3].plot(self.t_span, j_trace[:, 1], marker='x')
        axs[2, 3].plot(self.t_span, j_trace[:, 2], marker='x')
        t_diff, a_diff = utils.get_signal_derivative(
            self.t_span, self.dv_trace, self.dt)
        axs[0, 3].plot(t_diff, a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, a_diff[:, 2], marker='.')
        axs[0, 3].legend(['jerk from controller', 'jerk from dynamics'])

    def plot_omega_and_derivatives(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('omega_and_derivatives')
        axs[0, 0].plot(self.t_span, self.omega_trace[:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.omega_trace[:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.omega_trace[:, 2], marker='x')

        axs[0, 1].plot(self.t_span, self.omega_dot_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.omega_dot_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.omega_dot_trace[:, 2], marker='x')
        t_diff, omega_diff = utils.get_signal_derivative(
            self.t_span, self.omega_trace, self.dt)
        axs[0, 1].plot(t_diff, omega_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, omega_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, omega_diff[:, 2], marker='.')

    def plot_pose_and_derivatives(self):
        fig1, axs1 = plt.subplots(3, 3, sharex=True)
        fig1.suptitle('pose_and_derivatives')
        axs1[0, 0].plot(self.t_span, self.pose_trace[:, 0, 0], marker='x')
        axs1[1, 0].plot(self.t_span, self.pose_trace[:, 1, 0], marker='x')
        axs1[2, 0].plot(self.t_span, self.pose_trace[:, 2, 0], marker='x')
        axs1[0, 1].plot(self.t_span, self.pose_trace[:, 0, 1], marker='x')
        axs1[1, 1].plot(self.t_span, self.pose_trace[:, 1, 1], marker='x')
        axs1[2, 1].plot(self.t_span, self.pose_trace[:, 2, 1], marker='x')
        axs1[0, 2].plot(self.t_span, self.pose_trace[:, 0, 2], marker='x')
        axs1[1, 2].plot(self.t_span, self.pose_trace[:, 1, 2], marker='x')
        axs1[2, 2].plot(self.t_span, self.pose_trace[:, 2, 2], marker='x')
        fig2, axs2 = plt.subplots(3, 3, sharex=True)
        fig2.suptitle('pose_dot')
        axs2[0, 0].plot(self.t_span, self.pose_dot_trace[:, 0, 0], marker='x')
        axs2[1, 0].plot(self.t_span, self.pose_dot_trace[:, 1, 0], marker='x')
        axs2[2, 0].plot(self.t_span, self.pose_dot_trace[:, 2, 0], marker='x')
        axs2[0, 1].plot(self.t_span, self.pose_dot_trace[:, 0, 1], marker='x')
        axs2[1, 1].plot(self.t_span, self.pose_dot_trace[:, 1, 1], marker='x')
        axs2[2, 1].plot(self.t_span, self.pose_dot_trace[:, 2, 1], marker='x')
        axs2[0, 2].plot(self.t_span, self.pose_dot_trace[:, 0, 2], marker='x')
        axs2[1, 2].plot(self.t_span, self.pose_dot_trace[:, 1, 2], marker='x')
        axs2[2, 2].plot(self.t_span, self.pose_dot_trace[:, 2, 2], marker='x')
        t_diff, pose_diff = utils.get_signal_derivative(
            self.t_span, self.pose_trace, self.dt)
        axs2[0, 0].plot(t_diff, pose_diff[:, 0, 0], marker='.')
        axs2[1, 0].plot(t_diff, pose_diff[:, 1, 0], marker='.')
        axs2[2, 0].plot(t_diff, pose_diff[:, 2, 0], marker='.')
        axs2[0, 1].plot(t_diff, pose_diff[:, 0, 1], marker='.')
        axs2[1, 1].plot(t_diff, pose_diff[:, 1, 1], marker='.')
        axs2[2, 1].plot(t_diff, pose_diff[:, 2, 1], marker='.')
        axs2[0, 2].plot(t_diff, pose_diff[:, 0, 2], marker='.')
        axs2[1, 2].plot(t_diff, pose_diff[:, 1, 2], marker='.')
        axs2[2, 2].plot(t_diff, pose_diff[:, 2, 2], marker='.')

    def plot_trajectory(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('trajectory')
        axs[0, 0].plot(self.t_span, self.x_d_trace[:, 0], marker='x')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].plot(self.t_span, self.x_d_trace[:, 1], marker='x')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].plot(self.t_span, self.x_d_trace[:, 2], marker='x')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, self.v_d_trace[:, 0], marker='x')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].plot(self.t_span, self.v_d_trace[:, 1], marker='x')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].plot(self.t_span, self.v_d_trace[:, 2], marker='x')
        axs[2, 1].set_ylabel('v_z')
        t_diff, x_d_diff = utils.get_signal_derivative(
            self.t_span, self.x_d_trace, self.dt)
        axs[0, 1].plot(t_diff, x_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, x_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, x_d_diff[:, 2], marker='.')

        axs[0, 2].plot(self.t_span, self.x_d_dot2_trace[:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.x_d_dot2_trace[:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.x_d_dot2_trace[:, 2], marker='x')
        t_diff, v_d_diff = utils.get_signal_derivative(
            self.t_span, self.v_d_trace, self.dt)
        axs[0, 2].plot(t_diff, v_d_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, v_d_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, v_d_diff[:, 2], marker='.')

        axs[0, 3].plot(self.t_span, self.x_d_dot3_trace[:, 0], marker='x')
        axs[1, 3].plot(self.t_span, self.x_d_dot3_trace[:, 1], marker='x')
        axs[2, 3].plot(self.t_span, self.x_d_dot3_trace[:, 2], marker='x')
        t_diff, x_d_dot2_diff = utils.get_signal_derivative(
            self.t_span, self.x_d_dot2_trace, self.dt)
        axs[0, 3].plot(t_diff, x_d_dot2_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, x_d_dot2_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, x_d_dot2_diff[:, 2], marker='.')

    def plot_force_and_torque(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('force_and_torque')
        axs[0, 0].plot(self.t_span, self.f_trace[:, 0], marker='x')
        axs[0, 0].plot(self.t_span, self.f_d_trace[:, 0])
        axs[1, 0].plot(self.t_span, self.f_trace[:, 1], marker='x')
        axs[1, 0].plot(self.t_span, self.f_d_trace[:, 1])
        axs[2, 0].plot(self.t_span, self.f_trace[:, 2], marker='x')
        axs[2, 0].plot(self.t_span, self.f_d_trace[:, 2])
        axs[0, 0].set_ylabel('f_x')
        axs[1, 0].set_ylabel('f_y')
        axs[2, 0].set_ylabel('f_z')
        axs[0, 1].plot(self.t_span, self.torque_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.torque_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.torque_trace[:, 2], marker='x')
        axs[0, 1].set_ylabel('M_x')
        axs[1, 1].set_ylabel('M_y')
        axs[2, 1].set_ylabel('M_z')

    def plot_position_tracking_error(self):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_tracking_error')
        axs[0, 0].set_title("position error x")
        axs[1, 0].set_title("position error y")
        axs[2, 0].set_title("position error z")
        axs[0, 0].plot(self.t_span, self.e_x_trace[:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.e_x_trace[:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.e_x_trace[:, 2], marker='x')

        axs[0, 1].set_title("v error x")
        axs[1, 1].set_title("v error y")
        axs[2, 1].set_title("v error z")
        axs[0, 1].plot(self.t_span, self.e_v_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.e_v_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.e_v_trace[:, 2], marker='x')
        t_diff, e_x_diff = utils.get_signal_derivative(
            self.t_span, self.e_x_trace, self.dt)
        axs[0, 1].plot(t_diff, e_x_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, e_x_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, e_x_diff[:, 2], marker='.')
        t_diff, position_diff = utils.get_signal_derivative(
            self.t_span, self.position_trace, self.dt)
        t_diff, x_d_diff = utils.get_signal_derivative(
            self.t_span, self.x_d_trace, self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0] -
                       x_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1] -
                       x_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2] -
                       x_d_diff[:, 2], marker='.')
        axs[0, 1].plot(self.t_span, self.v_trace[:, 0] - self.v_d_trace[:, 0])
        axs[1, 1].plot(self.t_span, self.v_trace[:, 1] - self.v_d_trace[:, 1])
        axs[2, 1].plot(self.t_span, self.v_trace[:, 2] - self.v_d_trace[:, 2])

        axs[0, 2].set_title("a error x")
        axs[1, 2].set_title("a error y")
        axs[2, 2].set_title("a error z")
        axs[0, 2].plot(self.t_span, self.e_a_trace[:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.e_a_trace[:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.e_a_trace[:, 2], marker='x')
        t_diff, e_v_diff = utils.get_signal_derivative(
            self.t_span, self.e_v_trace, self.dt)
        axs[0, 2].plot(t_diff, e_v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, e_v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, e_v_diff[:, 2], marker='.')

        axs[0, 3].set_title("j error x")
        axs[1, 3].set_title("j error y")
        axs[2, 3].set_title("j error z")
        axs[0, 3].plot(self.t_span, self.e_j_trace[:, 0], marker='x')
        axs[1, 3].plot(self.t_span, self.e_j_trace[:, 1], marker='x')
        axs[2, 3].plot(self.t_span, self.e_j_trace[:, 2], marker='x')
        t_diff, e_a_diff = utils.get_signal_derivative(
            self.t_span, self.e_a_trace, self.dt)
        axs[0, 3].plot(t_diff, e_a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, e_a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, e_a_diff[:, 2], marker='.')
        j_trace = self.f_dot_trace/params.m
        axs[0, 3].plot(self.t_span, j_trace[:, 0] - self.x_d_dot3_trace[:, 0])
        axs[1, 3].plot(self.t_span, j_trace[:, 1] - self.x_d_dot3_trace[:, 1])
        axs[2, 3].plot(self.t_span, j_trace[:, 2] - self.x_d_dot3_trace[:, 2])
        '''
        create one time step misalignment in jerk
        '''
        # axs[0, 3].plot(t_diff, j_trace[:-1, 0] - self.x_d_dot3_trace[1:, 0])
        # axs[1, 3].plot(t_diff, j_trace[:-1, 1] - self.x_d_dot3_trace[1:, 1])
        # axs[2, 3].plot(t_diff, j_trace[:-1, 2] - self.x_d_dot3_trace[1:, 2])

    def plot_pose_tracking_error(self):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('pose_tracking_error')
        axs[0, 0].plot(self.t_span, self.e_r_trace[:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.e_r_trace[:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.e_r_trace[:, 2], marker='x')
        axs[0, 1].plot(self.t_span, self.e_omega_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.e_omega_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.e_omega_trace[:, 2], marker='x')
        axs[0, 2].plot(self.t_span, self.psi_r_rd_trace, marker='x')

    def plot_desired_force(self):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('desired_force')
        axs[0, 0].plot(self.t_span, self.f_d_trace[:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.f_d_trace[:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.f_d_trace[:, 2], marker='x')
        axs[0, 1].plot(self.t_span, self.f_d_dot_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.f_d_dot_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.f_d_dot_trace[:, 2], marker='x')
        t_diff, f_d_diff = utils.get_signal_derivative(
            self.t_span, self.f_d_trace, self.dt)
        axs[0, 1].plot(t_diff, f_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, f_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, f_d_diff[:, 2], marker='.')
        axs[0, 2].plot(self.t_span, self.f_d_dot2_trace[:, 0], marker='x')
        axs[1, 2].plot(self.t_span, self.f_d_dot2_trace[:, 1], marker='x')
        axs[2, 2].plot(self.t_span, self.f_d_dot2_trace[:, 2], marker='x')
        t_diff, f_d_dot_diff = utils.get_signal_derivative(
            self.t_span, self.f_d_dot_trace, self.dt)
        axs[0, 2].plot(t_diff, f_d_dot_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, f_d_dot_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, f_d_dot_diff[:, 2], marker='.')

    def plot_force(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('force')
        axs[0, 0].plot(self.t_span, self.f_trace[:, 0], marker='x')
        axs[1, 0].plot(self.t_span, self.f_trace[:, 1], marker='x')
        axs[2, 0].plot(self.t_span, self.f_trace[:, 2], marker='x')
        axs[0, 1].plot(self.t_span, self.f_dot_trace[:, 0], marker='x')
        axs[1, 1].plot(self.t_span, self.f_dot_trace[:, 1], marker='x')
        axs[2, 1].plot(self.t_span, self.f_dot_trace[:, 2], marker='x')
        t_diff, f_dot_diff = utils.get_signal_derivative(
            self.t_span, self.f_trace, self.dt)
        axs[0, 1].plot(t_diff, f_dot_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, f_dot_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, f_dot_diff[:, 2], marker='.')

    def plot_pose_desired(self):
        fig1, axs1 = plt.subplots(3, 3, sharex=True)
        fig1.suptitle('pose_desired')
        axs1[0, 0].plot(
            self.t_span, self.pose_desired_trace[:, 0, 0], marker='x')
        axs1[1, 0].plot(
            self.t_span, self.pose_desired_trace[:, 1, 0], marker='x')
        axs1[2, 0].plot(
            self.t_span, self.pose_desired_trace[:, 2, 0], marker='x')
        axs1[0, 1].plot(
            self.t_span, self.pose_desired_trace[:, 0, 1], marker='x')
        axs1[1, 1].plot(
            self.t_span, self.pose_desired_trace[:, 1, 1], marker='x')
        axs1[2, 1].plot(
            self.t_span, self.pose_desired_trace[:, 2, 1], marker='x')
        axs1[0, 2].plot(
            self.t_span, self.pose_desired_trace[:, 0, 2], marker='x')
        axs1[1, 2].plot(
            self.t_span, self.pose_desired_trace[:, 1, 2], marker='x')
        axs1[2, 2].plot(
            self.t_span, self.pose_desired_trace[:, 2, 2], marker='x')
        fig2, axs2 = plt.subplots(3, 3, sharex=True)
        fig2.suptitle('pose_desired_dot')
        axs2[0, 0].plot(
            self.t_span, self.pose_desired_dot_trace[:, 0, 0], marker='x')
        axs2[1, 0].plot(
            self.t_span, self.pose_desired_dot_trace[:, 1, 0], marker='x')
        axs2[2, 0].plot(
            self.t_span, self.pose_desired_dot_trace[:, 2, 0], marker='x')
        axs2[0, 1].plot(
            self.t_span, self.pose_desired_dot_trace[:, 0, 1], marker='x')
        axs2[1, 1].plot(
            self.t_span, self.pose_desired_dot_trace[:, 1, 1], marker='x')
        axs2[2, 1].plot(
            self.t_span, self.pose_desired_dot_trace[:, 2, 1], marker='x')
        axs2[0, 2].plot(
            self.t_span, self.pose_desired_dot_trace[:, 0, 2], marker='x')
        axs2[1, 2].plot(
            self.t_span, self.pose_desired_dot_trace[:, 1, 2], marker='x')
        axs2[2, 2].plot(
            self.t_span, self.pose_desired_dot_trace[:, 2, 2], marker='x')
        t_diff, pose_desired_diff = utils.get_signal_derivative(
            self.t_span, self.pose_desired_trace, self.dt)
        axs2[0, 0].plot(t_diff, pose_desired_diff[:, 0, 0], marker='.')
        axs2[1, 0].plot(t_diff, pose_desired_diff[:, 1, 0], marker='.')
        axs2[2, 0].plot(t_diff, pose_desired_diff[:, 2, 0], marker='.')
        axs2[0, 1].plot(t_diff, pose_desired_diff[:, 0, 1], marker='.')
        axs2[1, 1].plot(t_diff, pose_desired_diff[:, 1, 1], marker='.')
        axs2[2, 1].plot(t_diff, pose_desired_diff[:, 2, 1], marker='.')
        axs2[0, 2].plot(t_diff, pose_desired_diff[:, 0, 2], marker='.')
        axs2[1, 2].plot(t_diff, pose_desired_diff[:, 1, 2], marker='.')
        axs2[2, 2].plot(t_diff, pose_desired_diff[:, 2, 2], marker='.')
        fig3, axs3 = plt.subplots(3, 3, sharex=True)
        fig3.suptitle('pose_desired_dot2')
        axs3[0, 0].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 0, 0], marker='x')
        axs3[1, 0].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 1, 0], marker='x')
        axs3[2, 0].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 2, 0], marker='x')
        axs3[0, 1].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 0, 1], marker='x')
        axs3[1, 1].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 1, 1], marker='x')
        axs3[2, 1].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 2, 1], marker='x')
        axs3[0, 2].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 0, 2], marker='x')
        axs3[1, 2].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 1, 2], marker='x')
        axs3[2, 2].plot(
            self.t_span, self.pose_desired_dot2_trace[:, 2, 2], marker='x')
        t_diff, pose_desired_dot_diff = utils.get_signal_derivative(
            self.t_span, self.pose_desired_dot_trace, self.dt)
        axs3[0, 0].plot(t_diff, pose_desired_dot_diff[:, 0, 0], marker='.')
        axs3[1, 0].plot(t_diff, pose_desired_dot_diff[:, 1, 0], marker='.')
        axs3[2, 0].plot(t_diff, pose_desired_dot_diff[:, 2, 0], marker='.')
        axs3[0, 1].plot(t_diff, pose_desired_dot_diff[:, 0, 1], marker='.')
        axs3[1, 1].plot(t_diff, pose_desired_dot_diff[:, 1, 1], marker='.')
        axs3[2, 1].plot(t_diff, pose_desired_dot_diff[:, 2, 1], marker='.')
        axs3[0, 2].plot(t_diff, pose_desired_dot_diff[:, 0, 2], marker='.')
        axs3[1, 2].plot(t_diff, pose_desired_dot_diff[:, 1, 2], marker='.')
        axs3[2, 2].plot(t_diff, pose_desired_dot_diff[:, 2, 2], marker='.')

    def plot_disturbance_force(self):
        fig, axs = plt.subplots(4, 1, sharex=True)
        fig.suptitle('disturbance force')
        v_norm = np.sqrt(
            self.v_trace[:, 0]**2 + self.v_trace[:, 1]**2 + self.v_trace[:, 2]**2)
        f_norm = np.sqrt(
            self.f_disturb_trace[:, 0]**2 + self.f_disturb_trace[:, 1]**2 + self.f_disturb_trace[:, 2]**2)
        axs[0].plot(self.t_span, self.f_disturb_trace[:, 0], marker='x')
        axs[0].plot(self.t_span, f_norm*self.v_trace[:, 0]/v_norm, marker='.')
        axs[1].plot(self.t_span, self.f_disturb_trace[:, 1], marker='x')
        axs[1].plot(self.t_span, f_norm*self.v_trace[:, 1]/v_norm, marker='.')
        axs[2].plot(self.t_span, self.f_disturb_trace[:, 2], marker='x')
        axs[2].plot(self.t_span, f_norm*self.v_trace[:, 2]/v_norm, marker='.')
        axs[3].plot(self.t_span, f_norm, marker='x')
        axs[3].plot(self.t_span, 0.5*params.c_d *
                    params.area_frontal*v_norm**2, marker='.')

    def plot_motor_force(self):
        fig, axs = plt.subplots(4, 1, sharex=True)
        fig.suptitle('motor force')
        axs[0].plot(self.t_span, self.f_motor_trace[:, 0], marker='x')
        axs[1].plot(self.t_span, self.f_motor_trace[:, 1], marker='x')
        axs[2].plot(self.t_span, self.f_motor_trace[:, 2], marker='x')
        axs[3].plot(self.t_span, self.f_motor_trace[:, 3], marker='x')

    def plot_omega_desired(self):
        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.suptitle('omega_desired')
        axs[0].plot(self.t_span, self.omega_desired_trace[:, 0], marker='x')
        axs[0].plot(self.t_span, self.omega_trace[:, 0], marker='.')
        axs[1].plot(self.t_span, self.omega_desired_trace[:, 1], marker='x')
        axs[1].plot(self.t_span, self.omega_trace[:, 1], marker='.')
        axs[2].plot(self.t_span, self.omega_desired_trace[:, 2], marker='x')
        axs[2].plot(self.t_span, self.omega_trace[:, 2], marker='.')
        axs[0].legend(['omega desired', 'omega'])

    def plot_3d_trace(self):
        fig9, axs9 = plt.subplots(1, 1, sharex=True)
        axs9 = fig9.add_subplot(111, projection='3d')
        axs9.plot3D(self.x_d_trace[:, 0],
                    self.x_d_trace[:, 1],
                    self.x_d_trace[:, 2], '.', c='blue', label='Points')
        axs9.plot3D(self.position_trace[:, 0],
                    self.position_trace[:, 1],
                    self.position_trace[:, 2], 'green')
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
        ax.plot3D(self.x_d_trace[:, 0],
                    self.x_d_trace[:, 1],
                    self.x_d_trace[:, 2], '.', c='blue', label='Points')
        ax.plot3D(self.position_trace[:, 0],
                    self.position_trace[:, 1],
                    self.position_trace[:, 2], 'green')     
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')           
        ax.invert_zaxis()
        ax.invert_yaxis()   
        text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)   
        k = 10
        pos = self.position_trace[::k, :]
        pose_desire = self.pose_desired_trace[::k, :, :]
        pose = self.pose_trace[::k, :, :]
        t = self.t_span[::k]
        ani = FuncAnimation(fig, lambda n: update_frame(ax, text, pos[n, :], pose_desire[n, :, :], pose[n, :, :], t[n]), frames=int(len(self.pose_desired_trace)/k), interval=50, blit=True, repeat=False) 
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
    sim_test.run_simulation(10)
    sim_test.make_plot()
    plt.show()
