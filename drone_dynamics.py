import numpy as np
from scipy.integrate import solve_ivp
import quaternion
import quaternion_updater
import drone_parameters as params
import drone_utils as utils
import drone_disturbance_model as disturbance
import drone_propeller
import pandas as pd


class DroneDynamics:
    def __init__(self, position: np.ndarray, v: np.ndarray,
                 pose: np.ndarray, omega: np.ndarray, dt: float = 0.01) -> None:
        """
        pose is a 3x3 rotation matrix from body to inertial frame
        omega is in body fix frame
        inertial frame has front-right-down as x-y-z direction
        """
        self.dt = dt
        '''
        states
        '''
        self.position = position    # in inertial frame
        self.pose = pose    # rotation matrix in inertial frame
        self.v = v  # in inertial frame
        self.omega = omega  # omega in body fix frame
        self.q = utils.convert_rotation_matrix_to_quaternion(self.pose)
        # self.q_rotation = quaternion_updater.QuaternionOfRotation(utils.convert_rotation_matrix_to_quaternion(self.pose))
        '''
        derivatives
        '''
        self.v_dot = np.array([0.0, 0.0, 0.0])
        self.omega_dot = np.array([0.0, 0.0, 0.0])  # in body fix frame
        self.q_dot = np.quaternion(1.0, 0.0, 0.0, 0.0)
        '''
        input
        '''
        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force (positive in body frame -z direction); directly assigned from controller
        self.torque = np.array([0.0, 0.0, 0.0]) # in body fix frame
        '''
        disturbance
        '''
        self.disturbance = disturbance.WallEffect()
        self.f_disturb = self.disturbance.f_implicit + self.disturbance.f_explicit
        self.torque_disturb = self.disturbance.t_implicit + self.disturbance.t_explicit

        '''
        rotor positions
        rotors are labeled as _px, _nx, _py, _ny to denote the location of rotors on positive/negative x/y axis 
        in the dataframe px nx py ny are rows, position velocity are columns
        '''
        self.rotor = pd.DataFrame({
            "position": [np.array([params.d, 0, 0]), np.array([-params.d, 0, 0]), np.array([0, params.d, 0]), np.array([0, -params.d, 0])],
            "velocity": [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
        }, index=["px", "nx", "py", "ny"])
        self.rotor_spd_avg = 0.0    # rpm


    def pack_state_vector(self) -> np.ndarray:
        y = np.array([self.position[0], # 0
                      self.position[1], # 1
                      self.position[2], # 2
                      self.v[0],        # 3
                      self.v[1],        # 4
                      self.v[2],        # 5
                      self.q[0],        # 6
                      self.q[1],        # 7
                      self.q[2],        # 8
                      self.q[3],        # 9
                      self.omega[0],    # 10
                      self.omega[1],    # 11
                      self.omega[2],    # 12
                      self.disturbance.f_implicit[0],    # 13
                      self.disturbance.f_implicit[1],    # 14
                      self.disturbance.f_implicit[2],    # 15
                      self.disturbance.t_implicit[0],    # 16
                      self.disturbance.t_implicit[1],    # 17
                      self.disturbance.t_implicit[2]])   # 18
                      
        return y

    def unpack_state_vector(self, y):
        self.position = y[0:3]
        self.v = y[3:6]
        self.q = np.array([y[6], y[7], y[8], y[9]])
        self.pose = utils.convert_quaternion_to_rotation_matrix(self.q)
        self.omega = y[10:13]
        self.f_disturb_implicit = y[13:16]

    def unpack_state_derivatives(self, y_dot):
        self.v_dot = y_dot[3:6]
        self.q_dot = np.quaternion(y_dot[6], y_dot[7], y_dot[8], y_dot[9])
        self.omega_dot = y_dot[10:13]
        self.f_disturb_dot = y_dot[13:16]

    def step_state_vector(self, y_0: np.ndarray, t: float) -> np.ndarray:
        t_start = t
        t_end = t + self.dt
        # t_start = 0
        # t_end = 0 + self.dt
        result = solve_ivp(self.get_derivatives_from_eom, 
                           (t_start, t_end), 
                           y_0, 
                           method = 'RK45', 
                           t_eval=[t_end])
        if not result.success:
            raise ValueError(result.message)
        return result.y.reshape(-1)    # convert an nx1 matrix to 1xn vector

    def step_disturbance_force(self, t: float, state: np.ndarray):
        self.disturbance.update_explicit_force(t, state, drone_propeller.convert_rpm_to_rps(self.rotor_spd_avg))
        self.disturbance.update_explicit_torque(t, state, drone_propeller.convert_rpm_to_rps(self.rotor_spd_avg))
        self.f_disturb = self.disturbance.f_implicit + self.disturbance.f_explicit
        self.torque_disturb = self.disturbance.t_implicit + self.disturbance.t_explicit

    def step_dynamics(self, t: float) -> None:
        y_0 = self.pack_state_vector()
        self.step_disturbance_force(t, y_0)
        y_t = self.step_state_vector(y_0, t)
        self.unpack_state_vector(y_t)
        y_t_dot = self.get_derivatives_from_eom(t, y_t)   # argument t does affect output
        self.unpack_state_derivatives(y_t_dot)
        self.update_rotor_states()

    def get_derivatives_from_eom(self, t: float, y: np.ndarray) -> np.ndarray:
        '''
        equations of motion:
        y = np.array([position[0],  # 0
                      position[1],  # 1
                      position[2],  # 2
                      v[0],         # 3
                      v[1],         # 4
                      v[2],         # 5
                      q[0],         # 6
                      q[1],         # 7
                      q[2],         # 8
                      q[3],         # 9
                      omega[0],     # 10
                      omega[1],     # 11
                      omega[2]])    # 12
        '''
        position = y[0:3]
        v = y[3:6]
        q_instance = quaternion_updater.QuaternionOfRotation(np.quaternion(y[6], y[7], y[8], y[9]))
        omega = y[10:13]
        omega_in_inertial_frame = self.pose@omega
        pose = utils.convert_quaternion_to_rotation_matrix(np.array([y[6], y[7], y[8], y[9]]))
        self.disturbance.f_implicit = y[13:16]
        self.disturbance.update_explicit_force(t, y, drone_propeller.convert_rpm_to_rps(self.rotor_spd_avg))
        self.disturbance.update_explicit_torque(t, y, drone_propeller.convert_rpm_to_rps(self.rotor_spd_avg))

        position_dot = v
        v_dot = params.g*np.array([0.0, 0.0, 1.0]) + \
                pose@(-self.f + self.disturbance.f_implicit + self.disturbance.f_explicit)/params.m
        q_instance.step_derivative(omega_in_inertial_frame)     # quaternion derivative takes omega in intertal frame
        omega_dot = params.inertia_inv@(self.torque + self.disturbance.t_explicit + self.disturbance.t_implicit -
                                        utils.get_hat_map(omega)@params.inertia@omega)
        f_disturb_implicit_dot = self.disturbance.get_implicit_force_derivatives()
        t_disturb_implicit_dot = self.disturbance.get_implicit_torque_derivatives()
        y_dot = np.array([  position_dot[0],  # 0
                            position_dot[1],  # 1
                            position_dot[2],  # 2
                            v_dot[0],         # 3
                            v_dot[1],         # 4
                            v_dot[2],         # 5
                            q_instance.q_dot.w, # 6
                            q_instance.q_dot.x, # 7
                            q_instance.q_dot.y, # 8
                            q_instance.q_dot.z, # 9
                            omega_dot[0],   # 10
                            omega_dot[1],   # 11
                            omega_dot[2],   # 12
                            f_disturb_implicit_dot[0],   # 13
                            f_disturb_implicit_dot[1],   # 14
                            f_disturb_implicit_dot[2],   # 15
                            t_disturb_implicit_dot[0],   # 16
                            t_disturb_implicit_dot[1],   # 17
                            t_disturb_implicit_dot[2]])  # 18
        return y_dot
    
    def update_rotor_states(self) -> None:
        """refresh rotor position and velocity
        p_inertial_frame = p_frame + R*p_body_frame
        v_inertial_frame = v_frame + cross(omega_inertial_frame, R)*p_body + R*v_body
        """
        new = []
        for p in params.rotor_position:
            new.append(self.position + p@self.pose.T)
        self.rotor.loc[:, 'position'] = new

        new = []
        omega_r = np.cross(self.omega, self.pose.T).T # note the np.cross operation
        for p in params.rotor_position:
            new.append(self.v + omega_r@p)
        self.rotor.loc[:, 'velocity'] = new
        self.rotor_spd_avg = drone_propeller.prop_12x5.get_rotation_speed(self.f[2]/params.num_of_rotors)
        

if __name__ == "__main__":
    x = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 2.0, 3.0])
    pose1 = np.identity(3)
    omega1 = np.array([np.pi/6, 0.0, 0.0])
    dt1 = 1.0
    kinematics = DroneDynamics(x, v1, pose1, omega1, dt1)
    kinematics.step_dynamics(0.0)
    pose_answer = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                       [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
    position_answer = np.array([v1[0]*dt1, v1[1]*dt1, v1[2]*dt1 + 0.5*params.g*dt1*dt1])
    print(kinematics.pose - pose_answer)
    print(kinematics.position - position_answer)
    dt1 = 0.01
    kinematics = DroneDynamics(x, v1, pose1, omega1, dt1)
    for i in range(100):
        kinematics.step_dynamics(dt1*i)
    print(kinematics.pose - pose_answer)
    print(kinematics.position - position_answer)

    kinematics.torque = np.array([1.0, 0.0, 0.0])
    kinematics.step_dynamics(dt1*100)
    print(kinematics.omega_dot - params.inertia_inv@kinematics.torque)

    kinematics = DroneDynamics(x, v1, pose1, omega1, dt1)
    kinematics.update_rotor_states()
    print(kinematics.rotor)
    kinematics = DroneDynamics(x + np.array([1.0, 2.0, 3.0]), v1, pose1, omega1, dt1)
    kinematics.update_rotor_states()
    print(kinematics.rotor)

