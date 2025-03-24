import numpy as np

import utils


class State:
    """This class manage the state of the drone. The convention is in FRD frame.
    """
    def __init__(self, position: np.ndarray=np.array([0.0, 0.0, 0.0]), 
                 v: np.ndarray=np.array([0.0, 0.0, 0.0]),
                 pose: np.ndarray=np.eye(3), 
                 omega: np.ndarray=np.array([0.0, 0.0, 0.0])) -> None:
        self.position = position    # in inertial frame
        self.pose = pose    # rotation matrix in inertial frame
        self.v = v          # in inertial frame
        self.omega = omega  # omega in body fix frame
        self.q = utils.convert_rotation_matrix_to_quaternion(self.pose)

    def convert_to_body_frame(self, input):
        return self.pose.T@input

    def convert_to_inertial_frame(self, input):
        return self.pose@input

    def get_position_in_body_frame(self):
        return self.convert_to_body_frame(self.position)
    
    def get_position_in_inertial_frame(self):
        return self.position
    
    def get_velocity_in_body_frame(self):
        return self.convert_to_body_frame(self.v)
    
    def get_velocity_in_inertial_frame(self):
        return self.v
    
    def get_omega_in_body_frame(self):
        return self.omega
    
    def get_omega_in_inertial_frame(self):
        return self.convert_to_inertial_frame(self.omega)
    
    def get_pose_in_inertial_frame(self):
        return self.pose
    
    def get_pose_in_body_frame(self):
        return np.eye(3) 