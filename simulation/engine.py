import numpy as np

import drone.trajectory
import drone.dynamics
import drone.dynamics_state
import drone.controller
import drone.propeller
import drone.disturbance_model
import drone.utils as utils
import drone.parameters
import sim_logger
import scenario_factory

class Engine:
    """Core looping mechanism of simulation. Coordinates controller, trajectory and dynamics model.
    """

    def __init__(self) -> None:
        self.t = 0.0    # simulation epoch time
        self.dt_log = 0.01  # simulation step to log output
        self.dt_controller = 0.01   # controller cycle time
        self.dt_dynamics = 0.005    # dynamics model cycle time
        # controller steps per log step, must be an integer
        self.cl_ratio = round(self.dt_log/self.dt_controller)
        # dynamics steps per controller step, must be an integer
        self.dc_ratio = round(self.dt_controller/self.dt_dynamics)
        print("number of controller steps per simulation step: " + str(self.cl_ratio))
        print("number of dynamics model steps per controller step: " +
              str(self.dc_ratio))
        
        self.scenario = scenario_factory.Factory.make_scenario(
            drone.trajectory.RandomWaypoints(300, True),
            # drone.trajectory.CircleYZ(),
            # drone.trajectory.Hover(),
            drone.parameters.PennStateARILab550(),
            drone.propeller.apc_8x6,
            drone.disturbance_model.WindEffectNearWall(),
            self.dt_dynamics)
        '''
        data to plot
        '''
        self.t_span = []
        self.ani = None

    def step_simulation(self, t: float):
        t_controller = t
        for i in range(self.cl_ratio):
            t_controller += i*self.dt_controller
            self.scenario.trajectory.step_reference_state(self.t)
            self.scenario.dynamics.f = self.scenario.controller.f
            self.scenario.dynamics.torque = self.scenario.controller.torque
            t_dynamics = t_controller
            for j in range(self.dc_ratio):
                t_dynamics += j*self.dt_dynamics
                self.scenario.dynamics.step_dynamics(t_dynamics)
            self.scenario.controller.step_controller(
                self.scenario.dynamics, self.scenario.trajectory)
            self.t += self.dt_controller

    def run_simulation(self, logger: sim_logger.Logger, t_end):
        self.t_span = np.arange(0.0, t_end + self.dt_log, self.dt_log)
        for t in self.t_span:
            if t > 5:
                pass
            self.step_simulation(t)

            logger.buffer["position"].append(self.scenario.dynamics.state.position)
            logger.buffer["q"].append(self.scenario.dynamics.state.q)
            logger.buffer["v"].append(self.scenario.dynamics.state.v)
            logger.buffer["dv"].append(self.scenario.dynamics.v_dot)
            logger.buffer["e_x"].append(self.scenario.controller.e_x)
            logger.buffer["e_v"].append(self.scenario.controller.e_v)
            logger.buffer["e_a"].append(self.scenario.controller.e_a)
            logger.buffer["e_j"].append(self.scenario.controller.e_j)
            logger.buffer["e_r"].append(self.scenario.controller.e_r)
            logger.buffer["e_omega"].append(self.scenario.controller.e_omega)
            logger.buffer["psi_r_rd"].append(self.scenario.controller.psi_r_rd)
            logger.buffer["f_ctrl_input"].append(-self.scenario.dynamics.state.pose@self.scenario.controller.f)
            logger.buffer["f_ctrl_input_dot"].append(-self.scenario.dynamics.state.pose@self.scenario.controller.f_dot)
            logger.buffer["f_d"].append(self.scenario.controller.f_d)
            logger.buffer["f_d_dot"].append(self.scenario.controller.f_d_dot)
            logger.buffer["f_d_dot2"].append(self.scenario.controller.f_d_dot2)
            logger.buffer["torque_ctrl_input"].append(self.scenario.controller.torque)
            logger.buffer["x_d"].append(self.scenario.trajectory.x_d)
            logger.buffer["v_d"].append(self.scenario.trajectory.v_d)
            logger.buffer["x_d_dot2"].append(self.scenario.trajectory.x_d_dot2)
            logger.buffer["x_d_dot3"].append(self.scenario.trajectory.x_d_dot3)
            logger.buffer["f_motor"].append(self.scenario.controller.force_motor)
            logger.buffer["f_disturb"].append(self.scenario.dynamics.f_disturb)
            logger.buffer["f_disturb_est"].append(self.scenario.controller.f_disturb)
            logger.buffer["f_disturb_est_base"].append(self.scenario.controller.f_disturb_base)
            logger.buffer["torque_disturb"].append(self.scenario.dynamics.torque_disturb)
            logger.buffer["torque_disturb_est"].append(self.scenario.controller.torque_disturb)
            logger.buffer["torque_disturb_est_base"].append(self.scenario.controller.torque_disturb_base)
            logger.buffer["omega"].append(self.scenario.dynamics.state.pose@self.scenario.dynamics.state.omega)
            logger.buffer["omega_dot"].append(self.scenario.dynamics.state.pose@self.scenario.dynamics.omega_dot)
            logger.buffer["omega_desired"].append(self.scenario.controller.omega_desired)
            # suspect python pointer related bug occurs if remove *1
            logger.buffer["pose"].append(self.scenario.dynamics.state.pose*1)
            logger.buffer["pose_dot"].append(utils.get_hat_map(
                self.scenario.dynamics.state.pose@self.scenario.dynamics.state.omega)@self.scenario.dynamics.state.pose)
            logger.buffer["pose_desired"].append(self.scenario.controller.pose_desired)
            logger.buffer["pose_desired_dot"].append(self.scenario.controller.pose_desired_dot)
            logger.buffer["pose_desired_dot2"].append(self.scenario.controller.pose_desired_dot2)
            logger.buffer["rotor_spd"].append(self.scenario.dynamics.rotor_spd_avg)
            logger.buffer["b_1d"].append(self.scenario.trajectory.b_1d)


