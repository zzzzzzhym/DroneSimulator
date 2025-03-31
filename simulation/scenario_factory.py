import drone.trajectory
import drone.dynamics
import drone.dynamics_state
import drone.controller
import drone.propeller
import drone.disturbance_model
import drone.parameters
import scenario

class Factory:
    @staticmethod
    def make_scenario(trajectory: drone.trajectory.TrajectoryReference, 
                      drone_params: drone.parameters.Drone, 
                      propeller_params: drone.propeller.Propeller, 
                      disturbance_model: drone.disturbance_model.DisturbanceForce,
                      dt_dynamics: float) -> scenario.Scenario:
        product = scenario.Scenario()
        trajectory.set_init_state()
        init_state = drone.dynamics_state.State(
            trajectory.init_x,
            trajectory.init_v,
            trajectory.init_pose,
            trajectory.init_omega)
        
        product.dynamics = Factory.make_dynamics_assembly(
            drone_params, 
            propeller_params, 
            disturbance_model, 
            init_state, 
            dt_dynamics)
        product.controller = Factory.make_controller_assembly(drone_params)
        product.trajectory = trajectory
        return product

    @staticmethod
    def make_dynamics_assembly(drone_params: drone.parameters.Drone, 
                               propeller_params: drone.propeller.Propeller, 
                               disturbance_model: drone.disturbance_model.DisturbanceForce, 
                               init_state: drone.dynamics_state.State, 
                               dt_dynamics: float) -> drone.dynamics.DroneDynamics:
        assembly = drone.dynamics.DroneDynamics(
            drone_params,
            propeller_params,
            disturbance_model,
            init_state,
            dt_dynamics)
        return assembly
    
    @staticmethod
    def make_controller_assembly(drone_params: drone.parameters.Drone) -> drone.controller.DroneController:
        assembly = drone.controller.DroneController(drone_params)
        return assembly
    