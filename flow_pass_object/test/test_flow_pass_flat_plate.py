import unittest
import numpy as np

from flow_pass_flat_plate import FlowPassFlatPlate

class TestFlowPassFlatPlateInterface(unittest.TestCase):
    def test_get_attack_angle_from_wall_facing(self):
        wall_norm = np.array([0.0, -1.0, 0.0])   # facing x-axis
        wall_origin = np.array([0.0, 0.0, 0.0])
        wall_length = 8.0
        interface_instance = FlowPassFlatPlate.Interface(wall_norm, wall_origin, wall_length)
        u_free = np.array([10.0, 0.0, 0.0])
        transformation_matrix_inertial_to_model, transformation_matrix_model_to_inertial = interface_instance.get_rotation_matrix_between_inertial_and_model_frame(u_free)
        self.assertTrue(np.allclose(transformation_matrix_inertial_to_model, np.eye(3)))
        self.assertTrue(np.allclose(transformation_matrix_model_to_inertial, np.eye(3)))       
        u, alpha = interface_instance.get_model_input(u_free)
        self.assertTrue(np.allclose(u, 10.0))
        self.assertTrue(np.allclose(alpha, 0.0))
        u_free = np.array([0.0, 10.0, 0.0])
        transformation_matrix_inertial_to_model, transformation_matrix_model_to_inertial = interface_instance.get_rotation_matrix_between_inertial_and_model_frame(u_free)
        print(transformation_matrix_inertial_to_model)
        print(transformation_matrix_model_to_inertial)
        expected_transformation_matrix_inertial_to_model = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(transformation_matrix_inertial_to_model, expected_transformation_matrix_inertial_to_model))
        self.assertTrue(np.allclose(transformation_matrix_model_to_inertial, expected_transformation_matrix_inertial_to_model.T))
        u, alpha = interface_instance.get_model_input(u_free)
        self.assertTrue(np.allclose(u, 10.0))
        self.assertTrue(np.allclose(alpha, 0.5*np.pi))

if __name__ == '__main__':
    unittest.main()