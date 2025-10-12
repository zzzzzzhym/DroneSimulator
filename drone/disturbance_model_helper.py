import numpy as np

class VerticalWall:
    """Wall parameters
    """
    def __init__(self, wall_origin=np.array([-0.55, 0, 0]), wall_norm=np.array([1, 0, 0]), wall_length=4.0):
        self.wall_origin = wall_origin  # a point on the wall [m]
        self.wall_norm = wall_norm / np.linalg.norm(wall_norm)  # normal vector of the wall, should be unit vector
        self.wall_length = wall_length  # length of the wall in y and z direction [m]
        self.rigid_contact_stiffness = 10000.0  # [N/m], large value to approximate rigid contact and handle wall penetration
        self.rigid_contact_damping = 10000.0  # [N/(m/s)], damping when penetration happens