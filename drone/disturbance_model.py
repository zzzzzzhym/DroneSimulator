import numpy as np
import warnings

import parameters as params
import dynamics_state
import propeller
import rotor
import utils
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

    def update_explicit_wrench(self, t: float=0.0, state: dynamics_state.State=None) -> None:
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
        return np.zeros(3), np.zeros(3)    

def const_force(weight):
    # typical payload of a drone ranges from 0.2kg to 1kg
    # of course, to properly simulate payload, we should change mass instead
    return np.array([0, 0, params.Environment.g*weight])

class Free(DisturbanceForce):
    def __init__(self) -> None:
        super().__init__()

    def update_explicit_wrench(self, *args, **kwargs) -> None:
        self.f_explicit = np.zeros(3)
        self.t_explicit = np.zeros(3)

    def get_implicit_wrench_derivatives(self, *args, **kwargs) -> np.ndarray:
        return np.zeros(3), np.zeros(3)

class AirDrag(DisturbanceForce):
    def __init__(self) -> None:
        super().__init__()

    def update_explicit_wrench(self, t: float=0.0, v: np.ndarray=0.0) -> None:
        self.f_explicit = self.get_air_drag(v)
        self.t_explicit = np.zeros(3)

    def get_implicit_wrench_derivatives(self, t: float=0.0, state: np.ndarray=np.zeros(13)) -> np.ndarray:
        return np.zeros(3), np.zeros(3)
    
    @staticmethod
    def get_air_drag(v_wind: np.ndarray) -> np.ndarray:
        """air drag force in inertial frame
        f = 0.5*c_d*area*v_norm^2*(v/v_norm)

        Args:
            v_wind (np.ndarray): speed np.array([vx, vy, vz])

        Returns:
            np.ndarray: (3,) array to represent (f_x, f_y, f_z)
        """

        f = 0.5*params.c_d*params.area_frontal*np.sqrt(v_wind[0]**2+v_wind[1]**2+v_wind[2]**2)*v_wind
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
        self.propeller = propeller.prop_kde4215xf465_6s_15_5x5_3_dual
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

    def update_explicit_wrench(self, t: float=0.0, state: dynamics_state.State=None, rotor_spd: float=0.0) -> None:
        """F_wall = 0.5*C_F*rho_air*rotor_spd^2*d*4
        rotor_spd: rps
        """
        d = self.get_distance_to_wall(state.position)
        f = self.get_c_f(d)*0.5*params.Environment.rho_air*rotor_spd**2*self.propeller.diameter**4
        self.f_explicit = -f*self.wall_norm
        t = self.get_c_q(d)*0.5*params.Environment.rho_air*rotor_spd**2*self.propeller.diameter**5
        self.t_explicit = t*np.array([0, 1, 0])
    
    

class WindEffectNearWall(DisturbanceForce):
    """This model integrates flow pass a vertical wall and the inflow model of propeller. It simulates wind velocity field around a wall and its impact on drone rotors.
    """
    def __init__(self, wall_origin=np.array([-0.5, 0, 0]), wall_norm=np.array([1, 0, 0]), wall_length=4.0, u_free=np.array([-5.0, 0.0, 0.0])) -> None:
        super().__init__()
        self.propeller_force_table = propeller_lookup_table.PropellerLookupTable.Reader("apc_8x6_with_trail")
        self.wind_field_model = flow_pass_flat_plate.FlowPassFlatPlate.Interface(wall_norm, np.array([0.0, 0.0, 1.0]), wall_origin, wall_length)
        self.u_free_const = u_free    # in FLU inertial frame
        self.v_i_average = np.zeros(3)  # average downwash in FLU inertial frame
        self.u_free = self.u_free_const.copy()
        self.delayed_rotor_set_speed = None
        self.f_propeller = np.zeros(3)  # force on propeller in FLU inertial frame
        self.t_propeller = np.zeros(3)  # force on propeller in FLU inertial frame
        self.f_body = np.zeros(3)  # force on drone body in FLU inertial frame
        
    def generate_sinusoidal_wind(self, t: float) -> None:
        self.u_free[0] = self.u_free_const[0] + 5.0*np.sin(t) # Neural Fly test condition

    def update_explicit_wrench(self, t: float, state: dynamics_state.State, rotor_set: rotor.RotorSet, force_control, torque_control) -> None:
        """WARNING: To be tested

        Args:
            rotors (_type_): _description_
        """
        # self.generate_sinusoidal_wind(t)
        forces = []
        torques = []
        induced_flows = []
        self.step_delayed_rotation_speed(rotor_set)
        for rotor, delayed_rotor_speed in zip(rotor_set.rotors, self.delayed_rotor_set_speed):
            u_horizontal = self.u_free*np.array([1, 1, 0])  # horizontal wind velocity
            wind_velocity = self.wind_field_model.get_solution(u_horizontal, rotor.position_inertial_frame)  # in FLU inertial frame
            wind_velocity[2] = self.u_free[2]  # set the vertical wind velocity to be the same as the free stream velocity
            force, v_i = self.propeller_force_table.get_rotor_forces(wind_velocity, rotor.velocity_inertial_frame, rotor.pose, delayed_rotor_speed, rotor.is_ccw_blade)
            rotor.f_rotor_inertial_frame = force  # for debugging purpose, to check the force generated by air dynamics
            forces.append(force)
            torques.append(np.cross(rotor.relative_position_inertial_frame, force))
            induced_flows.append(v_i)
            rotor.local_wind_velocity = wind_velocity  # update the local wind velocity in rotor frame
        self.f_propeller = sum(forces)
        self.t_propeller = sum(torques)
        self.f_propeller = utils.FrdFluConverter.flip_vector(self.f_propeller)
        self.t_propeller = utils.FrdFluConverter.flip_vector(self.t_propeller)
        self.f_propeller = state.pose.T@self.f_propeller  # convert to body frame
        self.t_propeller = state.pose.T@self.t_propeller  # convert to body frame
        self.f_propeller = self.f_propeller - (-force_control)   # only the difference is considered as disturbance. postive force_control in negative z axis
        self.t_propeller = self.t_propeller - torque_control  # only the difference is considered as disturbance
        self.t_propeller[2] = 0.0    # the inflow model did not model torque in z axis
        
        self.v_i_average = sum(induced_flows) / len(induced_flows)
        self.f_body = self.get_disturbance_on_drone_body(state)  # add the air drag force on the drone body to the propeller force
        # self.blend_white_noise()  # add white noise to the force and torque

        self.f_explicit = self.f_propeller + self.f_body  # add the force on drone body to the propeller force
        self.t_explicit = self.t_propeller


    def blend_white_noise(self):
        """Blend the white noise to the force and torque
        """
        f_noise_std = np.abs(self.f_explicit)*0.2
        tq_noise_std = np.array([0.001, 0.001, 0.001])
        f_noise = np.random.normal(0, f_noise_std)
        tq_noise = np.random.normal(0, tq_noise_std)  
        self.f_explicit += f_noise
        self.t_explicit += tq_noise

    def get_disturbance_on_drone_body(self, state: dynamics_state.State) -> np.ndarray:
        v_local_wind = self.wind_field_model.get_solution(self.u_free, utils.FrdFluConverter.flip_vector(state.position)) 
        alpha = 0.3  # a magic number accounting the distance between drone body and the rotor
        v_total_wind = utils.FrdFluConverter.flip_vector(self.v_i_average*alpha + v_local_wind)
        f = AirDrag.get_air_drag(v_total_wind - state.v)
        return f
    
    def step_delayed_rotation_speed(self, rotor_set: rotor.RotorSet) -> None:
        """This function low pass filter the rotation speed
        """
        alpha = 0.0 # amount of delay
        if self.delayed_rotor_set_speed is None:
            self.delayed_rotor_set_speed = []
            for rotor in rotor_set.rotors:
                self.delayed_rotor_set_speed.append(rotor.rotation_speed)
        else:
            for i, rotor in enumerate(rotor_set.rotors):
                self.delayed_rotor_set_speed[i] = (1.0 - alpha) * rotor.rotation_speed + alpha * self.delayed_rotor_set_speed[i]


class AggregatedDisturbanceForce(DisturbanceForce):
    """This class is used to aggregate multiple disturbance forces. It is not a disturbance force itself.
    """
    def __init__(self, wind_effect_disturbance: WindEffectNearWall, air_drag: AirDrag) -> None:
        super().__init__()
        self.wind_effect_disturbance = wind_effect_disturbance

    def update_explicit_wrench(self, t: float, state: dynamics_state.State, rotor_set: rotor.RotorSet, force_control, torque_control) -> None:
        self.wind_effect_disturbance.update_explicit_wrench(t, state, rotor_set, force_control, torque_control)
        
        self.f_explicit = self.wind_effect_disturbance.f_explicit
        self.t_explicit = self.wind_effect_disturbance.t_explicit


if __name__ == "__main__":
    wall = WallEffect()
    wall.update_explicit_wrench(0.0, dynamics_state.State(), rotor_spd=2000.0/60)
    print(wall.f_explicit)
    print(wall.t_explicit)
