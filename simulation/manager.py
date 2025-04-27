import scenario_factory
import engine
import sim_logger
import plotter
import drone.trajectory
import drone.propeller
import drone.disturbance_model
import drone.parameters

class Manager:
    """this module calls scenario maker, run engine, log and plot result"""
    def __init__(self) -> None:
        self.logger = sim_logger.Logger()
        self.scenario = None
        self.engine = None

    def set_up(self, 
               trajectory=drone.trajectory.RandomWaypoints(300, True, True),    # when test training effect, we may want to use random subtrajectory set
            #    trajectory=drone.trajectory.RandomWaypoints(300, True), 
            #    trajectory= drone.trajectory.CircleYZ(),
            #    trajectory= drone.trajectory.Hover(),
               drone_params=drone.parameters.PennStateARILab550(), 
               propeller_params=drone.propeller.apc_8x6, 
               disturbance_model=drone.disturbance_model.WindEffectNearWall(),
               dt_dynamics=0.005) -> None:
        self.scenario = scenario_factory.Factory.make_scenario(
            trajectory, 
            drone_params, 
            propeller_params, 
            disturbance_model,
            dt_dynamics)
        self.engine = engine.Engine(self.scenario)

    def run(self, t_end: float) -> None:
        self.engine.run_simulation(self.logger, t_end)
        self.logger.convert_buffer_to_output()

    def save_result(self, file_name: str) -> None:
        self.logger.log_sim_result(file_name)

    def plot(self) -> None:
        result = plotter.Plotter(self.engine.t_span, self.engine.dt_log)
        result.make_plots(self.logger.output)
        