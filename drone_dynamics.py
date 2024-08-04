import numpy as np
from scipy.integrate import solve_ivp
import quaternion
import quaternion_updater
import drone_parameters as params
import drone_utils as utils
import drone_disturbance_force as disturbance


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
        self.q = np.quaternion(1.0, 0.0, 0.0, 0.0)
        '''
        derivatives
        '''
        self.v_dot = np.array([0.0, 0.0, 0.0])
        self.omega_dot = np.array([0.0, 0.0, 0.0])  # in body fix frame
        self.q_dot = np.quaternion(1.0, 0.0, 0.0, 0.0)
        '''
        input
        '''
        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force opposite to self.pose[:,2]; directly assigned from outside
        self.torque = np.array([0.0, 0.0, 0.0]) # in body fix frame
        '''
        disturbance
        '''
        self.air_drag = disturbance.AirDrag()
        self.f_disturb = self.air_drag.f_implicit + self.air_drag.f_explicit

    def pack_state_vector(self) -> np.ndarray:
        q = np.array([1.0, 0.0, 0.0, 0.0]) # quaternion 
        y = np.array([self.position[0], # 0
                      self.position[1], # 1
                      self.position[2], # 2
                      self.v[0],        # 3
                      self.v[1],        # 4
                      self.v[2],        # 5
                      q[0],             # 6
                      q[1],             # 7
                      q[2],             # 8
                      q[3],             # 9
                      self.omega[0],    # 10
                      self.omega[1],    # 11
                      self.omega[2],    # 12
                      self.air_drag.f_implicit[0],    # 13
                      self.air_drag.f_implicit[1],    # 14
                      self.air_drag.f_implicit[2]])   # 15
                      
        return y

    def unpack_state_vector(self, y):
        self.position = y[0:3]
        self.v = y[3:6]
        self.q = np.quaternion(y[6], y[7], y[8], y[9])
        q_instance = quaternion_updater.QuaternionOfRotation(self.q)
        self.pose = q_instance.step_rotation_matrix(self.pose)
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
        self.air_drag.update_explicit_force(t, state)
        self.f_disturb = self.air_drag.f_implicit + self.air_drag.f_explicit

    def step_dynamics(self, t: float) -> None:
        y_0 = self.pack_state_vector()
        self.step_disturbance_force(t, y_0)
        y_t = self.step_state_vector(y_0, t)
        self.unpack_state_vector(y_t)
        y_t_dot = self.get_derivatives_from_eom(t, y_t)   # argument t does affect output
        self.unpack_state_derivatives(y_t_dot)

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
        pose = q_instance.step_rotation_matrix(self.pose)
        self.air_drag.f_implicit = y[13:16]
        self.air_drag.update_explicit_force(t, y)

        position_dot = v
        v_dot = params.g*np.array([0.0, 0.0, 1.0]) - \
                pose@self.f/params.m + self.air_drag.f_implicit + self.air_drag.f_explicit
        q_instance.step_derivative(omega_in_inertial_frame)     # quaternion derivative takes omega in intertal frame
        omega_dot = params.inertia_inv@(self.torque -
                                        utils.get_hat_map(omega)@params.inertia@omega)
        f_disturb_implicit_dot = self.air_drag.get_implicit_force_derivatives()
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
                            f_disturb_implicit_dot[2]])  # 15
        return y_dot
    

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
