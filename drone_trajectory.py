import numpy as np
import matplotlib.pyplot as plt     # test only
from mpl_toolkits import mplot3d    # test only


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
    ref_test = Circle()
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
    ax = fig.add_subplot(111, projection='3d')
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
    ax.plot(x, y, z)
    ax.plot(dx, dy, dz)
    ax.plot(ddx, ddy, ddz)
    ax.axis('equal')
    plt.show()
