import unittest
import numpy as np
from drone_dynamics import DroneDynamics
import drone_parameters as params
import drone_utils as utils

class TestDroneDynamics(unittest.TestCase):

    def setUp(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.v = np.array([1.0, 2.0, 3.0])
        self.pose = np.identity(3)
        self.omega = np.array([np.pi/6, 0.0, 0.0])
        self.dt = 0.01
        self.drone = DroneDynamics(self.position, self.v, self.pose, self.omega, self.dt)

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.drone.position, self.position))
        self.assertTrue(np.array_equal(self.drone.v, self.v))
        self.assertTrue(np.array_equal(self.drone.pose, self.pose))
        self.assertTrue(np.array_equal(self.drone.omega, self.omega))
        self.assertEqual(self.drone.dt, self.dt)

    def test_pack_state_vector(self):
        state_vector = self.drone.pack_state_vector()
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
            self.drone.disturbance.f_implicit[0],  # x-component of the disturbance force
            self.drone.disturbance.f_implicit[1],  # y-component of the disturbance force
            self.drone.disturbance.f_implicit[2],  # z-component of the disturbance force
            self.drone.disturbance.t_implicit[0],  # x-component of the disturbance torque
            self.drone.disturbance.t_implicit[1],  # y-component of the disturbance torque
            self.drone.disturbance.t_implicit[2]   # z-component of the disturbance torque
        ])
        np.testing.assert_almost_equal(state_vector, expected_vector, decimal=5)

    def test_unpack_state_vector(self):
        state_vector = self.drone.pack_state_vector()
        self.drone.unpack_state_vector(state_vector)
        self.assertTrue(np.array_equal(self.drone.position, self.position))
        self.assertTrue(np.array_equal(self.drone.v, self.v))
        self.assertTrue(np.array_equal(self.drone.omega, self.omega))

    def test_step_dynamics(self):
        dt1 = 0.01
        for i in range(100):
            self.drone.step_dynamics(dt1*i)
        t_span = 1.0
        pose_answer = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi/6), -np.sin(np.pi/6)],
            [0.0, np.sin(np.pi/6), np.cos(np.pi/6)]
        ])
        q_answer = utils.convert_rotation_matrix_to_quaternion(pose_answer)
        position_answer = np.array([self.v[0]*t_span, self.v[1]*t_span, self.v[2]*t_span + 0.5*params.g*t_span*t_span])
        np.testing.assert_almost_equal(self.drone.pose, pose_answer, decimal=5)
        np.testing.assert_almost_equal(self.drone.q, q_answer, decimal=5)
        np.testing.assert_almost_equal(self.drone.position, position_answer, decimal=5)

    def test_update_rotor_states(self):
        self.drone.update_rotor_states()
        expected_positions = [self.position + p@self.drone.pose.T for p in params.rotor_position]
        expected_velocities = [self.v + np.cross(self.omega, self.drone.pose.T).T@p for p in params.rotor_position]
        for i, rotor in enumerate(self.drone.rotor.index):
            np.testing.assert_almost_equal(self.drone.rotor.loc[rotor, 'position'], expected_positions[i], decimal=5)
            np.testing.assert_almost_equal(self.drone.rotor.loc[rotor, 'velocity'], expected_velocities[i], decimal=5)

if __name__ == '__main__':
    unittest.main()