import numpy as np
import warnings

import drone_parameters as params
import drone_dynamics_state
import drone_propeller
import drone_rotor
import drone_utils as utils
import inflow_model.propeller_lookup_table as propeller_lookup_table
import flow_pass_object.flow_pass_flat_plate as flow_pass_flat_plate

class DisturbanceForce:
    """Generate a force in inertial frame
    API of implicit/explicit disturbance (derivatives) 

        Args:
            t (float): time for disturbance force explicitly dependent on time
            state (np.ndarray): np.array([  position[0],  # 0
                                            position[1],  # 1
                                            position[2],  # 2
                                            v[0],         # 3
                                            v[1],         # 4
                                            v[2],         # 5
                                            q[0],         # 6
                                            q[1],         # 7
                                            q[2],         # 8
                                            q[3],         # 9
                                            omega[0],     # 10
                                            omega[1],     # 11
                                            omega[2]])    # 12
        Returns:
            to be specified                                          
    
    """
    def __init__(self) -> None:
        """f: force t: torque, in body frame
        """
        self.f_implicit = np.zeros(3)
        self.f_explicit = np.zeros(3)
        self.t_implicit = np.zeros(3)
        self.t_explicit = np.zeros(3)

    def update_explicit_wrench(self, t: float=0.0, state: drone_dynamics_state.State=None) -> None:
        """API of explicit disturbance force

        Args:
            see class level API
        Returns:
            tuple of np.ndarray: (3,) array to represent (f_x, f_y, f_z) and (3,) array to represent (t_x, t_y, t_z)
        """
        raise NotImplementedError("This function need to be implemented by subclasses")
    
    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        """API of implicit disturbance derivatives

        Args:
            see class level API
        Returns:
            tuple of np.ndarray: (3,) array to represent (f_x, f_y, f_z) and (3,) array to represent (t_x, t_y, t_z)
        """
        raise NotImplementedError("This function need to be implemented by subclasses")

def const_force(weight):
    # typical payload of a drone ranges from 0.2kg to 1kg
    # of course, to properly simulate payload, we should change mass instead
    return np.array([0, 0, params.Environment.g*weight])

class Free(DisturbanceForce):
    def __init__(self) -> None:
        super().__init__()

    def update_explicit_wrench(self, t: float=0.0, state: drone_dynamics_state.State=None, dummy=0) -> None:
        self.f_explicit = np.zeros(3)
        self.t_explicit = np.zeros(3)

    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3), np.zeros(3)

class AirDrag(DisturbanceForce):
    def __init__(self) -> None:
        super().__init__()

    def update_explicit_wrench(self, t: float=0.0, state: drone_dynamics_state.State=None) -> None:
        self.f_explicit = self.get_air_drag(state.v)
        self.t_explicit = np.zeros(3)

    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3), np.zeros(3)
    
    @staticmethod
    def get_air_drag(v: np.ndarray) -> np.ndarray:
        """air drag force in inertial frame
        f = 0.5*c_d*area*v_norm^2*(v/v_norm)

        Args:
            v (np.ndarray): speed np.array([vx, vy, vz])

        Returns:
            np.ndarray: (3,) array to represent (f_x, f_y, f_z)
        """

        f = 0.5*params.c_d*params.area_frontal*np.sqrt(v[0]**2+v[1]**2+v[2]**2)*v
        return f

class WallEffect(DisturbanceForce):
    """Ref:
    Ground, Ceiling and Wall Effect Evaluation of Small Quadcopters in Pressure-controlled Environments
    """
    def __init__(self) -> None:
        """Wall location"""
        self.wall_origin = np.array([-params.rotor_radius*2, 0, 0])
        self.wall_norm = np.array([1, 0, 0])  # norm vector of the wall; 
        """Wall effect params"""
        self.max_force = 0.02
        self.max_force_dr = 4.0     # distance - radius ratio
        self.max_torque = 0.02
        self.max_torque_dr = 4.0
        """Propeller model"""
        self.propeller = drone_propeller.prop_kde4215xf465_6s_15_5x5_3_dual
        super().__init__()
        print(f"Wall location {self.wall_origin}")

    def get_c_q(self, distance):
        dr = distance/params.rotor_radius
        if dr < 1:
            warnings.warn("drone-wall interference detected")
            dr = 1
        if dr > self.max_torque_dr:
            norm_torque = 0
        else:
            k = -self.max_torque / self.max_torque_dr
            norm_torque = k*dr + self.max_torque
        return norm_torque
        
    def get_c_f(self, distance):
        dr = distance/params.rotor_radius
        if dr < 1:
            warnings.warn("drone-wall interference detected")
            dr = 1        
        dr = distance/params.rotor_radius
        if dr > self.max_force_dr:
            norm_force = 0
        else:
            k = -self.max_force / self.max_force_dr
            norm_force = k*dr + self.max_force
        return norm_force

    def get_distance_to_wall(self, location: np.ndarray):
        d = (location - self.wall_origin)@self.wall_norm
        return d

    def update_explicit_wrench(self, t: float=0.0, state: drone_dynamics_state.State=None, rotor_spd: float=0.0) -> None:
        """F_wall = 0.5*C_F*rho_air*rotor_spd^2*d*4
        rotor_spd: rps
        """
        d = self.get_distance_to_wall(state.position)
        f = self.get_c_f(d)*0.5*params.Environment.rho_air*rotor_spd**2*self.propeller.diameter**4
        self.f_explicit = -f*self.wall_norm
        t = self.get_c_q(d)*0.5*params.Environment.rho_air*rotor_spd**2*self.propeller.diameter**5
        self.t_explicit = t*np.array([0, 1, 0])
    
    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3), np.zeros(3)
    

class WindEffectNearWall(DisturbanceForce):
    """This model integrates flow pass a vertical wall and the inflow model of propeller. It simulates wind velocity field around a wall and its impact on drone rotors.
    """
    def __init__(self, wall_origin=np.array([-0.5, 0, 0]), wall_norm=np.array([1, 0, 0]), wall_length=4.0) -> None:
        super().__init__()
        self.propeller_force_table = propeller_lookup_table.PropellerLookupTable.Reader("apc_8x6")
        self.wind_field_model = flow_pass_flat_plate.FlowPassFlatPlate.Interface(wall_norm, wall_origin, wall_length)
        self.u_free = np.array([-5.0, 0.0, 0.0])    # in FLU inertial frame

    def update_explicit_wrench(self, t: float, state: drone_dynamics_state.State, rotor_set: drone_rotor.RotorSet, force_control, torque_control) -> None:
        """WARNING: To be tested

        Args:
            rotors (_type_): _description_
        """
        forces = []
        torques = []
        for rotor in rotor_set.rotors:
            wind_velocity = self.wind_field_model.get_solution(self.u_free, rotor.position_inertial_frame)
            force = self.propeller_force_table.get_rotor_forces(wind_velocity, rotor.velocity_inertial_frame, rotor.pose, rotor.rotation_speed, rotor.is_ccw_blade)
            forces.append(force)
            torques.append(np.cross(rotor.relative_position_inertial_frame, force))
        self.f_explicit = sum(forces)
        self.t_explicit = sum(torques)
        self.f_explicit = utils.FrdFluConverter.flip(self.f_explicit)
        self.t_explicit = utils.FrdFluConverter.flip(self.t_explicit)
        self.f_explicit = state.pose.T@self.f_explicit  # convert to body frame
        self.t_explicit = state.pose.T@self.t_explicit  # convert to body frame
        self.f_explicit = self.f_explicit - (-force_control)   # only the difference is considered as disturbance. postive force_control in negative z axis
        self.t_explicit = self.t_explicit - torque_control  # only the difference is considered as disturbance

        # debug
        # if np.linalg.norm(self.f_explicit) > 10 and t > 0.5:
        #     print(f"Wind effect: {self.f_explicit}")
        self.t_explicit[2] = 0.0    # the inflow model did not model torque in z axis
        

    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3), np.zeros(3)

if __name__ == "__main__":
    wall = WallEffect()
    wall.update_explicit_wrench(0.0, drone_dynamics_state.State(), rotor_spd=2000.0/60)
    print(wall.f_explicit)
    print(wall.t_explicit)
