import numpy as np
import matplotlib.pyplot as plt     # test only

from drone_utils import KinematicVars
from trajectory_generation import flight_map
import drone_parameters as params
import drone_plot_utils

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

    def step_reference_state(self) -> None:
        raise NotImplementedError("Method needed for trajectory")

class RandomWaypoints(TrajectoryReference):
    def __init__(self, num_of_segments):
        super().__init__()  # Initialize parent class parameters
        file_name_prefix = "5_random_wp_map_"
        self.trajectory: flight_map.FlightMap = flight_map.construct_map_with_subtrajs(is_random=True, num_of_subtrajs=num_of_segments, subtraj_id=[0,1,2,3,4])
        
    def step_reference_state(self, t) -> None:
        self.x_d, self.v_d, self.x_d_dot2, self.x_d_dot3, self.x_d_dot4 = self.trajectory.read_data_by_time(t)   

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
    def set_init_state(self, is_upside_down=True) -> None:
        self.init_x = np.array([0.0, 0.0, 0.0])
        self.init_v = np.array([0.0, 0.0, 0.0])
        self.init_omega = np.array([0.0, 0.0, 0.0])
        if is_upside_down:
            self.init_pose = np.array([[1.0, 0.0, 0.0],
                                    [0.0, -0.9995, -0.0314],
                                    [0.0, 0.0314, -0.9995]])
        else:
            self.init_pose = np.identity(3)


def plot_desired_pose(ax: plt.Axes, traj: TrajectoryReference, t_start, t_stop, dt = 0.1):
    g = np.array([0, 0, params.g])  # z points down
    t_span = np.arange(t_start, t_stop, 0.1)
    for t in t_span:
        traj.step_reference_state(t)
        b_1d = traj.b_1d
        b_3d = -(traj.x_d_dot2 - g)    # a_input + g = x_d_dot2; a_input is opposite to b_3d
        norm = np.sqrt(b_3d@b_3d)
        if norm < 0.0001:
            b_3d = -0.0001*g
            norm = 0.0001*params.g
        b_3d = b_3d/norm
        b_2d = np.cross(b_3d, b_1d)
        b1b2, b3 = drone_plot_utils.generate_drone_profile(traj.x_d, np.vstack([b_1d,b_2d,b_3d]).T)
        ax.plot(b1b2[:, 0],
                b1b2[:, 1],
                b1b2[:, 2], 'orange')
        ax.plot(b3[:, 0],
                b3[:, 1],
                b3[:, 2], 'orange')

if __name__ == "__main__":
    t_test = np.arange(0, 20.1, 0.1)
    ref_test = RandomWaypoints(5)
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
    dddx = []
    dddy = []
    dddz = []
    ddddx = []
    ddddy = []
    ddddz = []
    fig0 = plt.figure()
    ax0_1 = fig0.add_subplot(131, projection='3d')
    ax0_2 = fig0.add_subplot(132, projection='3d')
    ax0_3 = fig0.add_subplot(133, projection='3d')
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
        dddx.append(ref_test.x_d_dot3[0])
        dddy.append(ref_test.x_d_dot3[1])
        dddz.append(ref_test.x_d_dot3[2])
        ddddx.append(ref_test.x_d_dot4[0])
        ddddy.append(ref_test.x_d_dot4[1])
        ddddz.append(ref_test.x_d_dot4[2])
    ax0_1.plot(x, y, z)
    ax0_2.plot(dx, dy, dz)
    ax0_3.plot(ddx, ddy, ddz)
    ax0_1.axis('equal')
    ax0_2.axis('equal')
    ax0_3.axis('equal')
    
    fig1, ax1 = plt.subplots(5, 3)
    ax1[0,0].plot(t_test, x)
    ax1[0,1].plot(t_test, y)
    ax1[0,2].plot(t_test, z)
    ax1[1,0].plot(t_test, dx)
    ax1[1,1].plot(t_test, dy)
    ax1[1,2].plot(t_test, dz)
    ax1[2,0].plot(t_test, ddx)
    ax1[2,1].plot(t_test, ddy)
    ax1[2,2].plot(t_test, ddz)
    ax1[3,0].plot(t_test, dddx)
    ax1[3,1].plot(t_test, dddy)
    ax1[3,2].plot(t_test, dddz)
    ax1[4,0].plot(t_test, ddddx)
    ax1[4,1].plot(t_test, ddddy)
    ax1[4,2].plot(t_test, ddddz)

    ax1[0,0].set_ylabel("x")
    ax1[0,1].set_ylabel("y")
    ax1[0,2].set_ylabel("z")
    ax1[1,0].set_ylabel("dx")
    ax1[1,1].set_ylabel("dy")
    ax1[1,2].set_ylabel("dz")
    ax1[2,0].set_ylabel("ddx")
    ax1[2,1].set_ylabel("ddy")
    ax1[2,2].set_ylabel("ddz")
    ax1[3,0].set_ylabel("dddx")
    ax1[3,1].set_ylabel("dddy")
    ax1[3,2].set_ylabel("dddz")
    ax1[4,0].set_ylabel("ddddx")
    ax1[4,1].set_ylabel("ddddy")
    ax1[4,2].set_ylabel("ddddz")    

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d') 
    plot_desired_pose(ax2, ref_test, 0, 20)   
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.axis('equal')           
    ax2.invert_zaxis()
    ax2.invert_yaxis()
    plt.show()
