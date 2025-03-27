import numpy as np

def generate_drone_profile(position: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """generates a drone figure, square shape is the drone x-y plane, the line is the z direciton

    Args:
        position (np.ndarray): location to draw
        pose (np.ndarray): 3x3 matrix represent [e_x, e_y, e_z] 

    Returns:
        tuple[np.ndarray, np.ndarray]: b1b2 is the plane, b3 is the z direciton
    """
    b1b2 = np.vstack((position,
                      position + 0.5*pose[:, 0].T,
                      position + 0.5*pose[:, 1].T,
                      position - 0.5*pose[:, 0].T,
                      position - 0.5*pose[:, 1].T,
                      position + 0.5*pose[:, 0].T))
    b3 = np.vstack((position,
                    position + 0.5*pose[:, 2].T))
    return b1b2, b3