import unittest
import numpy as np

import parameters as params
import disturbance_model as disturbance
import dynamics_state as state
import rotor
import propeller

class TestWindEffectNearWall(unittest.TestCase):
    def test_update_explicit_wrench(self):
        drone_state = state.State()
        rotor_instance = rotor.RotorSet(params.PennStateARILab550(), propeller.apc_8x6)
        instance = disturbance.WindEffectNearWall()
        instance.u_free = np.array([0.0, 0.0, 0.0])
        t = 0.0
        force_control = np.array([0.0, 0.0, 5.0])
        torque_control = np.array([0.0, 0.0, 0.0])
        thrust = params.PennStateARILab550().m_wrench_to_thrust@np.hstack((force_control[2], torque_control))
        drone_state.omega = np.array([0.0, 0.0, 0.0])
        rotor_instance.step_rotor_states(drone_state, thrust)
        instance.update_explicit_wrench(t, drone_state, rotor_instance, force_control, torque_control)
        np.testing.assert_array_almost_equal(instance.f_explicit, np.zeros(3), decimal=1)
        np.testing.assert_array_almost_equal(instance.t_explicit, np.zeros(3))


if __name__ == '__main__':
    unittest.main()