import numpy as np
import drone_trajectory as trajectory
import drone_dynamics as dynamics
import drone_controller as controller
import matplotlib.pyplot as plt


class DroneSimulator:
    """
    Controller + Dynamics
    """

    def __init__(self) -> None:
        self.init_x = np.array([0.0, 0.0, 0.0])
        self.init_v = np.array([0.0, 0.0, 0.0])
        self.init_omega = np.array([0.0, 0.0, 0.0])
        self.init_pose = np.identity(3)
        self.t = 0.0
        self.dt = 0.01
        self.sim_trajectory = trajectory.TrajectoryReference()
        self.sim_dynamics = dynamics.DroneDynamics(self.init_x, self.init_v, self.init_pose, self.init_omega, self.dt)
        self.sim_controller = controller.DroneController()

    def step_simulation(self):
        self.sim_trajectory.step_reference_state(self.t)
        self.sim_dynamics.f = self.sim_controller.f
        self.sim_dynamics.torque = self.sim_controller.torque
        self.sim_dynamics.step_dynamics()
        self.sim_controller.step_controller(self.sim_dynamics, self.sim_trajectory)
        self.t += self.dt

    def run_simulation(self, t_end):
        t_span = np.arange(0.0, t_end + self.dt, self.dt)
        fig1, axs1 = plt.subplots(3,1, sharex=True)
        fig2, axs2 = plt.subplots(3,2, sharex=True)
        fig3, axs3 = plt.subplots(3,2, sharex=True)
        position_trace = np.empty((0,3))
        e_x_trace = np.empty((0,3))
        f_trace = np.empty((0,3))
        torque_trace = np.empty((0,3))
        for _ in t_span:
            self.step_simulation()
            position_trace = np.vstack((position_trace, self.sim_dynamics.position))
            e_x_trace = np.vstack((e_x_trace, self.sim_controller.e_x))
            f_trace = np.vstack((f_trace, self.sim_controller.f))
            torque_trace = np.vstack((torque_trace, self.sim_controller.torque))
        axs1[0].plot(t_span, position_trace[:,0], marker='x')
        axs1[1].plot(t_span, position_trace[:,1], marker='x')
        axs1[2].plot(t_span, position_trace[:,2], marker='x')
        axs2[0,0].plot(t_span, f_trace[:,0], marker='x')
        axs2[1,0].plot(t_span, f_trace[:,1], marker='x')
        axs2[2,0].plot(t_span, f_trace[:,2], marker='x')
        axs2[0,1].plot(t_span, torque_trace[:,0], marker='x')
        axs2[1,1].plot(t_span, torque_trace[:,1], marker='x')
        axs2[2,1].plot(t_span, torque_trace[:,2], marker='x')

        axs3[0,1].plot(t_span, e_x_trace[:,0], marker='x')
        axs3[1,1].plot(t_span, e_x_trace[:,1], marker='x')
        axs3[2,1].plot(t_span, e_x_trace[:,2], marker='x')

        plt.show()

if __name__ == "__main__":
    sim_test = DroneSimulator()
    sim_test.run_simulation(10.0) 
