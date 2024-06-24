import numpy as np
import matplotlib.pyplot as plt     # test only
from mpl_toolkits import mplot3d    # test only
import pickle
from trajectory_generation import trajectory_generator_qp 

class TrajectoryReference:
    def __init__(self) -> None:
        self.init_x = np.array([0.0, 0.0, 0.0])
        self.init_v = np.array([0.0, 0.0, 0.0])
        self.init_omega = np.array([0.0, 0.0, 0.0])
        self.init_pose = np.identity(3)        
        self.x_d = np.array([0.0, 0.0, 0.0])    # x_d
        self.v_d = np.array([0.0, 0.0, 0.0])    # x_d_dot
        self.x_d_dot2 = np.array([0.0, 0.0, 0.0])    # x_d_dot2
        self.x_d_dot3 = np.array([0.0, 0.0, 0.0])    # x_d_dot3
        self.x_d_dot4 = np.array([0.0, 0.0, 0.0])    # x_d_dot4
        self.b_1d = np.array([1.0, 0.0, 0.0])
        self.b_1d_dot = np.array([0.0, 0.0, 0.0])
        self.b_1d_dot2 = np.array([0.0, 0.0, 0.0])

    def set_init_state(self) -> None:
        pass

class RandomWaypoints(TrajectoryReference):
    def __init__(self):
        super().__init__()  # Initialize parent class parameters
        self.trajectory: trajectory_generator_qp.TrajectoryGenerator = self.load_trajectory('trajectory.pkl')
        for dim in self.trajectory.profiles:
            for section in dim:
                section.extend_to_derivative_order(4)

    def load_trajectory(self, pkl_path) -> trajectory_generator_qp.TrajectoryGenerator:
        with open(pkl_path, 'rb') as file:
        # Deserialize each object from the file
            return pickle.load(file)

    def step_reference_state(self, t) -> None:
        t_clamped = min(self.trajectory.waypoints.waypoint_time_stamp[-1], t)
        i = self.find_section(t_clamped)
        self.x_d = np.array([self.trajectory.profiles[0][i].sample_polynomial(0, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[1][i].sample_polynomial(0, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[2][i].sample_polynomial(0, t_clamped - self.trajectory.time_shift[i])])
        self.v_d = np.array([self.trajectory.profiles[0][i].sample_polynomial(1, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[1][i].sample_polynomial(1, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[2][i].sample_polynomial(1, t_clamped - self.trajectory.time_shift[i])])
        self.x_d_dot2 = np.array([self.trajectory.profiles[0][i].sample_polynomial(2, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[1][i].sample_polynomial(2, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[2][i].sample_polynomial(2, t_clamped - self.trajectory.time_shift[i])])
        self.x_d_dot3 = np.array([self.trajectory.profiles[0][i].sample_polynomial(3, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[1][i].sample_polynomial(3, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[2][i].sample_polynomial(3, t_clamped - self.trajectory.time_shift[i])])
        self.x_d_dot4 = np.array([self.trajectory.profiles[0][i].sample_polynomial(4, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[1][i].sample_polynomial(4, t_clamped - self.trajectory.time_shift[i]),
                             self.trajectory.profiles[2][i].sample_polynomial(4, t_clamped - self.trajectory.time_shift[i])])
        

    def find_section(self, t) -> int:
        i = np.searchsorted(self.trajectory.waypoints.waypoint_time_stamp, t) - 1
        i = min(self.trajectory.waypoints.number_of_sections - 1, i)
        i = max(0, i)
        return i

class SpiralAndSpin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*t,
                             0.4*np.sin(np.pi*t),
                             0.6*np.cos(np.pi*t)])
        self.v_d = np.array([0.4,
                             0.4*np.pi*np.cos(np.pi*t),
                             -0.6*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([0.0,
                             -0.4*(np.pi**2)*np.sin(np.pi*t),
                             -0.6*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([0.0,
                             -0.4*(np.pi**3)*np.cos(np.pi*t),
                             0.6*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.0,
                             0.4*(np.pi**4)*np.sin(np.pi*t),
                             0.6*(np.pi**4)*np.cos(np.pi*t)])
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])
        
class SpiralZNegativeAndSpin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*t,
                             0.4*np.sin(np.pi*t),
                             -0.6*np.cos(np.pi*t)])
        self.v_d = np.array([0.4,
                             0.4*np.pi*np.cos(np.pi*t),
                             0.6*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([0.0,
                             -0.4*(np.pi**2)*np.sin(np.pi*t),
                             0.6*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([0.0,
                             -0.4*(np.pi**3)*np.cos(np.pi*t),
                             -0.6*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.0,
                             0.4*(np.pi**4)*np.sin(np.pi*t),
                             -0.6*(np.pi**4)*np.cos(np.pi*t)])
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])

class Spiral(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*t,
                             0.4*np.sin(np.pi*t),
                             0.6*np.cos(np.pi*t)])
        self.v_d = np.array([0.4,
                             0.4*np.pi*np.cos(np.pi*t),
                             -0.6*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([0.0,
                             -0.4*(np.pi**2)*np.sin(np.pi*t),
                             -0.6*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([0.0,
                             -0.4*(np.pi**3)*np.cos(np.pi*t),
                             0.6*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.0,
                             0.4*(np.pi**4)*np.sin(np.pi*t),
                             0.6*(np.pi**4)*np.cos(np.pi*t)])

class StraightXAndSpin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*t,
                             0.0,
                             0.0])
        self.v_d = np.array([0.4,
                             0.0,
                             0.0])   
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])  
          
class StraightX(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*t,
                             0.0,
                             0.0])
        self.v_d = np.array([0.4,
                             0.0,
                             0.0])       
        
class CircleYZ(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.0,
                             0.4*np.sin(np.pi*t),
                             0.4*np.cos(np.pi*t)-0.4])
        self.v_d = np.array([0.0,
                             0.4*np.pi*np.cos(np.pi*t),
                             -0.4*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([0.0,
                             -0.4*(np.pi**2)*np.sin(np.pi*t),
                             -0.4*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([0.0,
                             -0.4*(np.pi**3)*np.cos(np.pi*t),
                             0.4*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.0,
                             0.4*(np.pi**4)*np.sin(np.pi*t),
                             0.4*(np.pi**4)*np.cos(np.pi*t)])
        
class CircleYZAndSpin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.0,
                             0.4*np.sin(np.pi*t),
                             0.4*np.cos(np.pi*t)-0.4])
        self.v_d = np.array([0.0,
                             0.4*np.pi*np.cos(np.pi*t),
                             -0.4*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([0.0,
                             -0.4*(np.pi**2)*np.sin(np.pi*t),
                             -0.4*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([0.0,
                             -0.4*(np.pi**3)*np.cos(np.pi*t),
                             0.4*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.0,
                             0.4*(np.pi**4)*np.sin(np.pi*t),
                             0.4*(np.pi**4)*np.cos(np.pi*t)])
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])        

class CircleXZ(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*np.sin(np.pi*t),
                             0.0,
                             0.4*np.cos(np.pi*t)-0.4])
        self.v_d = np.array([0.4*np.pi*np.cos(np.pi*t),
                             0.0,
                             -0.4*np.pi*np.sin(np.pi*t)])
        self.x_d_dot2 = np.array([-0.4*(np.pi**2)*np.sin(np.pi*t),
                                  0.0,
                                  -0.4*(np.pi**2)*np.cos(np.pi*t)])
        self.x_d_dot3 = np.array([-0.4*(np.pi**3)*np.cos(np.pi*t),
                                  0.0,
                                  0.4*(np.pi**3)*np.sin(np.pi*t)])
        self.x_d_dot4 = np.array([0.4*(np.pi**4)*np.sin(np.pi*t),
                                  0.0,
                                  0.4*(np.pi**4)*np.cos(np.pi*t)])
        
class CircleXY(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*np.sin(np.pi*t),
                             0.4*np.cos(np.pi*t)-0.4,
                             0.0])
        self.v_d = np.array([0.4*np.pi*np.cos(np.pi*t),
                             -0.4*np.pi*np.sin(np.pi*t),
                             0.0])
        self.x_d_dot2 = np.array([-0.4*(np.pi**2)*np.sin(np.pi*t),
                                  -0.4*(np.pi**2)*np.cos(np.pi*t),
                                  0.0])
        self.x_d_dot3 = np.array([-0.4*(np.pi**3)*np.cos(np.pi*t),
                                  0.4*(np.pi**3)*np.sin(np.pi*t),
                                  0.0])
        self.x_d_dot4 = np.array([0.4*(np.pi**4)*np.sin(np.pi*t),
                                  0.4*(np.pi**4)*np.cos(np.pi*t),
                                  0.0])
        
class CircleXYAndSpin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.x_d = np.array([0.4*np.sin(np.pi*t),
                             0.4*np.cos(np.pi*t)-0.4,
                             0.0])
        self.v_d = np.array([0.4*np.pi*np.cos(np.pi*t),
                             -0.4*np.pi*np.sin(np.pi*t),
                             0.0])
        self.x_d_dot2 = np.array([-0.4*(np.pi**2)*np.sin(np.pi*t),
                                  -0.4*(np.pi**2)*np.cos(np.pi*t),
                                  0.0])
        self.x_d_dot3 = np.array([-0.4*(np.pi**3)*np.cos(np.pi*t),
                                  0.4*(np.pi**3)*np.sin(np.pi*t),
                                  0.0])
        self.x_d_dot4 = np.array([0.4*(np.pi**4)*np.sin(np.pi*t),
                                  0.4*(np.pi**4)*np.cos(np.pi*t),
                                  0.0])      
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])              

class Spin(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        self.b_1d = np.array([np.cos(np.pi*t),
                              np.sin(np.pi*t),
                              0.0])
        self.b_1d_dot = np.array([-np.pi*np.sin(np.pi*t),
                                  np.pi*np.cos(np.pi*t),
                                  0.0])
        self.b_1d_dot2 = np.array([-(np.pi**2)*np.cos(np.pi*t),
                                  -(np.pi**2)*np.sin(np.pi*t),
                                  0.0])
        
class Hover(TrajectoryReference):
    def step_reference_state(self, t) -> None:
        return None
    def set_init_state(self) -> None:
        self.init_x = np.array([0.0, 0.0, 0.0])
        self.init_v = np.array([0.0, 0.0, 0.0])
        self.init_omega = np.array([0.0, 0.0, 0.0])
        self.init_pose = np.array([[1.0, 0.0, 0.0],
                                   [0.0, -0.9995, -0.0314],
                                   [0.0, 0.0314, -0.9995]])


if __name__ == "__main__":
    t_test = np.arange(0, 10.1, 0.1)
    ref_test = RandomWaypoints()
    # ref_test = SpiralAndSpin()
    x = []
    y = []
    z = []
    dx = []
    dy = []
    dz = []
    ddx = []
    ddy = []
    ddz = []
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    for tt in t_test:
        ref_test.step_reference_state(tt)
        x.append(ref_test.x_d[0])
        y.append(ref_test.x_d[1])
        z.append(ref_test.x_d[2])
        dx.append(ref_test.v_d[0])
        dy.append(ref_test.v_d[1])
        dz.append(ref_test.v_d[2])
        ddx.append(ref_test.x_d_dot2[0])
        ddy.append(ref_test.x_d_dot2[1])
        ddz.append(ref_test.x_d_dot2[2])
    ax1.plot(x, y, z)
    ax2.plot(dx, dy, dz)
    ax3.plot(ddx, ddy, ddz)
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')
    plt.show()
