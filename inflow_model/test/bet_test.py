import unittest
import numpy as np

import bet
from blade_params import Blade, apc_8x6

class TestBet(unittest.TestCase):
    def setUp(self):
        self.bet_instance = bet.BladeElementTheory(apc_8x6())

    def test_get_v_flow_disk_frame(self):
        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free=np.array([10, 0, 0]), v_i=0.0, v_forward=np.array([0, 0, 0]), r_disk=np.eye(3))
        np.testing.assert_array_equal(v_flow_disk_frame, np.array([10, 0, 0]))
        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free=np.array([0, 10, 0]), v_i=0.0, v_forward=np.array([0, 0, 0]), r_disk=np.eye(3))
        np.testing.assert_array_equal(v_flow_disk_frame, np.array([0, 10, 0]))

        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free=np.array([10, 0, 0]), v_i=10.0, v_forward=np.array([0, 0, 0]), r_disk=np.eye(3))
        np.testing.assert_array_equal(v_flow_disk_frame, np.array([10, 0, -10]))

        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free=np.array([0, 0, 0]), v_i=10.0, v_forward=np.array([10, 0, 0]), r_disk=np.eye(3))
        np.testing.assert_array_equal(v_flow_disk_frame, np.array([-10, 0, -10]))

        # disk rotates w.r.t y axis to make z point to x direction in the inertial frame
        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free=np.array([0, 0, 0]), v_i=10.0, v_forward=np.array([10, 0, 0]), r_disk=np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
        np.testing.assert_array_equal(v_flow_disk_frame, np.array([0, 0, 0]))

    def test_get_relative_v_to_blade_section(self):
        u_t, u_p, alpha_flow = self.bet_instance.get_relative_v_to_blade_section(v_flow_disk_frame=np.array([10, 0, 0]), psi_blade_angle=0.0, y=0.0, omega_blade=0.0)
        self.assertEqual(u_t, 0)
        self.assertEqual(u_p, 0)
        self.assertEqual(alpha_flow, 0)

        u_t, u_p, alpha_flow = self.bet_instance.get_relative_v_to_blade_section(v_flow_disk_frame=np.array([10, 0, 0]), psi_blade_angle=0.0, y=0.0, omega_blade=1.0)
        self.assertEqual(u_t, 0)
        self.assertEqual(u_p, 0)
        self.assertEqual(alpha_flow, 0)

        u_t, u_p, alpha_flow = self.bet_instance.get_relative_v_to_blade_section(v_flow_disk_frame=np.array([10, 0, 0]), psi_blade_angle=0.0, y=1.0, omega_blade=0.0)
        self.assertEqual(u_t, 0)
        self.assertEqual(u_p, 0)
        self.assertEqual(alpha_flow, 0)

        u_t, u_p, alpha_flow = self.bet_instance.get_relative_v_to_blade_section(v_flow_disk_frame=np.array([10, 0, 0]), psi_blade_angle=0.0, y=1.0, omega_blade=1.0)
        self.assertEqual(u_t, 1.0)
        self.assertEqual(u_p, 0.0)
        self.assertEqual(alpha_flow, 0.0)

        u_t, u_p, alpha_flow = self.bet_instance.get_relative_v_to_blade_section(v_flow_disk_frame=np.array([10, 0, 0]), psi_blade_angle=np.pi/3, y=0.0, omega_blade=0.0)
        self.assertEqual(u_t, 10*np.sin(np.pi/3))
        self.assertEqual(u_p, 0.0)
        self.assertEqual(alpha_flow, 0.0)

    def test_get_attack_angle_to_blade(self):
        attack_angle = self.bet_instance.get_attack_angle_to_blade(0.0, 0.0)
        self.assertEqual(attack_angle, np.radians(45))
        attack_angle = self.bet_instance.get_attack_angle_to_blade(0.1, 0.0)
        self.assertEqual(attack_angle, np.radians(45) + 0.1)
        attack_angle = self.bet_instance.get_attack_angle_to_blade(0.1, self.bet_instance.blade.y_max)
        self.assertEqual(attack_angle, np.radians(17) + 0.1)

    def test_get_blade_element_force_in_airfoil_frame(self):
        lift, drag = self.bet_instance.get_blade_element_force_in_airfoil_frame(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(lift, 0)
        self.assertEqual(drag, 0)

        lift, drag = self.bet_instance.get_blade_element_force_in_airfoil_frame(0.0, 0.0, 0.0, self.bet_instance.blade.y_max)
        self.assertEqual(lift, 0)
        self.assertEqual(drag, 0)

        lift, drag = self.bet_instance.get_blade_element_force_in_airfoil_frame(0.0, 0.0, 0.1, 0.0)
        self.assertEqual(lift, 0)
        self.assertEqual(drag, 0)

        lift, drag = self.bet_instance.get_blade_element_force_in_airfoil_frame(0.0, 0.0, 0.1, self.bet_instance.blade.y_max)
        self.assertEqual(lift, 0)
        self.assertEqual(drag, 0)

    def convert_force_from_airfoil_to_blade_frame(self):
        f_z, f_y = self.bet_instance.convert_force_from_airfoil_to_blade_frame(lift=1.0, drag=0.0, alpha_flow=0.0)
        self.assertEqual(f_z, 1.0)
        self.assertEqual(f_y, 0)

        f_z, f_y = self.bet_instance.convert_force_from_airfoil_to_blade_frame(lift=0.0, drag=1.0, alpha_flow=0.0)
        self.assertEqual(f_z, 0)
        self.assertEqual(f_y, 1.0)

        f_z, f_y = self.bet_instance.convert_force_from_airfoil_to_blade_frame(lift=1.0, drag=0.0, alpha_flow=np.pi/3)
        self.assertAlmostEqual(f_z, 0.5)
        self.assertAlmostEqual(f_y, np.sqrt(3)/2)

        f_z, f_y = self.bet_instance.convert_force_from_airfoil_to_blade_frame(lift=0.0, drag=1.0, alpha_flow=np.pi/3)
        self.assertAlmostEqual(f_z, np.sqrt(3)/2)
        self.assertAlmostEqual(f_y, -0.5)

    def convert_force_from_blade_to_disk_frame(self):
        f_x, f_y, f_z = self.bet_instance.convert_force_from_blade_to_disk_frame(f_z_blade=1.0, f_y_blade=0.0, psi_blade_angle=0.0)
        self.assertEqual(f_x, 0)
        self.assertEqual(f_y, 0)
        self.assertEqual(f_z, 1.0)

        f_x, f_y, f_z = self.bet_instance.convert_force_from_blade_to_disk_frame(f_z_blade=0.0, f_y_blade=1.0, psi_blade_angle=0.0)
        self.assertEqual(f_x, 0)
        self.assertEqual(f_y, 1.0)
        self.assertEqual(f_z, 0)

        f_x, f_y, f_z = self.bet_instance.convert_force_from_blade_to_disk_frame(f_z_blade=1.0, f_y_blade=0.0, psi_blade_angle=np.pi/3)
        self.assertAlmostEqual(f_x, 0.0)
        self.assertAlmostEqual(f_y, 0.0)
        self.assertAlmostEqual(f_z, 1.0)

        f_x, f_y, f_z = self.bet_instance.convert_force_from_blade_to_disk_frame(f_z_blade=0.0, f_y_blade=1.0, psi_blade_angle=np.pi/3)
        self.assertAlmostEqual(f_x, -np.sqrt(3)/2)
        self.assertAlmostEqual(f_y, 0.5)
        self.assertAlmostEqual(f_z, 0)

    def test_integrate_element_force(self):
        # test ccw and cw symmetry
        u_free = np.array([10, 5, 0])
        v_i = 10
        v_forward = np.array([0, 0, 0])
        r_disk = np.eye(3)
        omega_blade = 2000*2*np.pi/60
        is_ccw_blade = True
        f_ccw = self.bet_instance.integrate_element_force(u_free, v_i, v_forward, r_disk, omega_blade, is_ccw_blade=is_ccw_blade)
        u_free[1] = -u_free[1]
        omega_blade = -omega_blade
        is_ccw_blade = False
        f_cw = self.bet_instance.integrate_element_force(u_free, v_i, v_forward, r_disk, omega_blade, is_ccw_blade=is_ccw_blade)
        self.assertAlmostEqual(f_ccw[0], f_cw[0])
        self.assertAlmostEqual(f_ccw[1], -f_cw[1])
        self.assertAlmostEqual(f_ccw[2], f_cw[2])

    def test_get_force_in_blade_frame(self):
        y = self.bet_instance.blade.y_max*0.5
        u_free_ccw = np.array([10, 5, 0])
        v_i = 0
        v_forward_ccw = np.array([0, 0, 0])
        r_disk = np.eye(3)
        omega_blade_ccw = 2000*2*np.pi/60
        psi_blade_angle_ccw = np.pi*0.3

        is_ccw_blade = True
        v_flow_disk_frame_ccw = self.bet_instance.get_v_flow_disk_frame(u_free_ccw, v_i, v_forward_ccw, r_disk)
        f_z_blade_ccw, f_y_blade_ccw = self.bet_instance.get_force_in_blade_frame(v_flow_disk_frame_ccw, psi_blade_angle_ccw, y, omega_blade_ccw, is_ccw_blade)

        u_free_cw = u_free_ccw.copy()
        u_free_cw[1] = -u_free_ccw[1]
        v_forward_cw = v_forward_ccw.copy()
        v_forward_cw[1] = -v_forward_ccw[1]
        omega_blade_cw = -omega_blade_ccw
        psi_blade_angle_cw = -psi_blade_angle_ccw
        is_ccw_blade = False
        v_flow_disk_frame_cw = self.bet_instance.get_v_flow_disk_frame(u_free_cw, v_i, v_forward_cw, r_disk)
        f_z_blade_cw, f_y_blade_cw = self.bet_instance.get_force_in_blade_frame(v_flow_disk_frame_cw, psi_blade_angle_cw, y, omega_blade_cw, is_ccw_blade)
        self.assertAlmostEqual(f_z_blade_ccw, f_z_blade_cw)
        self.assertAlmostEqual(f_y_blade_ccw, -f_y_blade_cw)

    def test_integrate_element_force_over_one_revolution(self):
        # test ccw and cw symmetry
        y = self.bet_instance.blade.y_max*0.5
        u_free = np.array([10, 5, 0])
        v_i = 10
        v_forward = np.array([0, 0, 0])
        r_disk = np.eye(3)
        omega_blade = 2000*2*np.pi/60
        is_ccw_blade = True
        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        df_ccw = self.bet_instance.integrate_element_force_over_one_revolution(y, v_flow_disk_frame, omega_blade, is_ccw_blade)

        u_free[1] = -u_free[1]
        omega_blade = -omega_blade
        is_ccw_blade = False
        v_flow_disk_frame = self.bet_instance.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        df_cw = self.bet_instance.integrate_element_force_over_one_revolution(y, v_flow_disk_frame, omega_blade, is_ccw_blade)

        self.assertAlmostEqual(df_ccw[0], df_cw[0])
        self.assertAlmostEqual(df_ccw[1], -df_cw[1])
        self.assertAlmostEqual(df_ccw[2], df_cw[2])

    def test_solve_v_i(self):
        u_free = np.array([10, 5, 0])
        v_forward = np.array([0, 0, 0])
        r_disk = np.eye(3)
        omega_blade = 2000*2*np.pi/60
        is_ccw_blade = True
        y = 0.1
        v_i = self.bet_instance.solve_v_i(y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade)
        should_be_almost_0 = self.bet_instance.get_thrust_difference(np.array([v_i]), y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade)
        self.assertAlmostEqual(should_be_almost_0, 0.0) 

if __name__ == '__main__':
    unittest.main()