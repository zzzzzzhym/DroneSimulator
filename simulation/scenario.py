import drone.trajectory
import drone.dynamics
import drone.controller
import drone.sensor

class Scenario:
    def __init__(self):
        self.dynamics: drone.dynamics.DroneDynamics = None    # assemble drone type, disturbance model, dynamics model
        self.controller: drone.controller.DroneController = None  # assemble drone type, controller and disturbance estimator
        self.trajectory: drone.trajectory.TrajectoryReference = None  
        self.sensor: drone.sensor.Sensor = None
