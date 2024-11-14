import unittest
import numpy as np

import drone_parameters 
import drone_dynamics
import drone_disturbance_model
import drone_utils as utils

class TestDroneDynamics(unittest.TestCase):
    def setUp(self):
        self.drone = drone_parameters.TrackingOnSE3()
        self.position = np.array([0.0, 0.0, 0.0])
        self.v = np.array([1.0, 2.0, 3.0])
        self.pose = np.eye(3)
        self.omega = np.array([np.pi/6, 0.0, 0.0])
        self.dt = 0.01  # Time step
        self.dynamics = drone_dynamics.DroneDynamics(
            self.drone, self.position, self.v, self.pose, self.omega, self.dt
        )
        self.dynamics.disturbance = drone_disturbance_model.Free()

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.dynamics.state.position, self.position))
        self.assertTrue(np.array_equal(self.dynamics.state.v, self.v))
        self.assertTrue(np.array_equal(self.dynamics.state.pose, self.pose))
        self.assertTrue(np.array_equal(self.dynamics.state.omega, self.omega))
        self.assertEqual(self.dynamics.dt, self.dt)

    def test_pack_state_vector(self):
        state_vector = self.dynamics.pack_state_vector()
        # Expected state vector based on the initial conditions
        expected_vector = np.array([
            self.position[0],  # x-coordinate of the position
            self.position[1],  # y-coordinate of the position
            self.position[2],  # z-coordinate of the position
            self.v[0],         # x-component of the velocity
            self.v[1],         # y-component of the velocity
            self.v[2],         # z-component of the velocity
            1.0,               # w-component of the quaternion (real part)
            0.0,               # x-component of the quaternion (imaginary part)
            0.0,               # y-component of the quaternion (imaginary part)
            0.0,               # z-component of the quaternion (imaginary part)
            self.omega[0],     # x-component of the angular velocity
            self.omega[1],     # y-component of the angular velocity
            self.omega[2],     # z-component of the angular velocity
            self.dynamics.disturbance.f_implicit[0],  # x-component of the disturbance force
            self.dynamics.disturbance.f_implicit[1],  # y-component of the disturbance force
            self.dynamics.disturbance.f_implicit[2],  # z-component of the disturbance force
            self.dynamics.disturbance.t_implicit[0],  # x-component of the disturbance torque
            self.dynamics.disturbance.t_implicit[1],  # y-component of the disturbance torque
            self.dynamics.disturbance.t_implicit[2]   # z-component of the disturbance torque
        ])
        np.testing.assert_almost_equal(state_vector, expected_vector, decimal=5)

    def test_unpack_state_vector(self):
        state_vector = self.dynamics.pack_state_vector()
        self.dynamics.unpack_state_vector(state_vector)
        self.assertTrue(np.array_equal(self.dynamics.state.position, self.position))
        self.assertTrue(np.array_equal(self.dynamics.state.v, self.v))
        self.assertTrue(np.array_equal(self.dynamics.state.omega, self.omega))

    def test_step_dynamics_0(self):
        """one step integration for 1sec"""
        dt = 1.0
        self.dynamics.dt = 1.0
        self.dynamics.step_dynamics(0.0)
        pose_answer = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                                [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
        position_answer = np.array([self.v[0]*dt, self.v[1]*dt, self.v[2]*dt + 0.5*drone_parameters.Environment.g*dt*dt])
        np.testing.assert_array_almost_equal(self.dynamics.state.pose, pose_answer)
        np.testing.assert_array_almost_equal(self.dynamics.state.position, position_answer)

    def test_step_dynamics_1(self):
        """fine integration for 1sec"""
        self.dynamics.dt = 0.01
        for i in range(100):
            self.dynamics.step_dynamics(self.dt*i)
        pose_answer = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
                                [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]])
        dt = 1.0
        position_answer = np.array([self.v[0]*dt, self.v[1]*dt, self.v[2]*dt + 0.5*drone_parameters.Environment.g*dt*dt])
        np.testing.assert_array_almost_equal(self.dynamics.state.pose, pose_answer, decimal=5)
        np.testing.assert_array_almost_equal(self.dynamics.state.position, position_answer, decimal=5)

    def test_step_dynamics_2(self):
        """test torque"""
        self.dynamics.torque = np.array([1.0, 0.0, 0.0])
        self.dynamics.step_dynamics(self.dt*100)
        expected_omega_dot = self.drone.inertia_inv@self.dynamics.torque
        print(self.dynamics.omega_dot - expected_omega_dot)

    def test_update_rotor_states(self):
        self.dynamics.update_rotor_states()
        print(self.dynamics.rotors.rotors[0].relative_position_body_frame)

    # def test_update_rotor_states(self):
    #     self.drone.update_rotor_states()
    #     expected_positions = [self.position + p@self.drone.state.pose.T for p in params.rotor_position]
    #     expected_velocities = [self.v + np.cross(self.omega, self.drone.state.pose.T).T@p for p in params.rotor_position]
    #     for i, rotor in enumerate(self.drone.rotors.index):
    #         np.testing.assert_almost_equal(self.drone.rotors.loc[rotor, 'position'], expected_positions[i], decimal=5)
    #         np.testing.assert_almost_equal(self.drone.rotors.loc[rotor, 'velocity'], expected_velocities[i], decimal=5)

if __name__ == '__main__':
    unittest.main()
