import matplotlib.pyplot as plt

import engine
import sim_logger
import plotter


class Manager:
    """this module calls scenario maker, run engine, log and plot result"""
    def __init__(self) -> None:
        self.logger = sim_logger.Logger()
        self.engine = engine.Engine()

    def run(self, t_end: float) -> None:
        self.engine.run_simulation(self.logger, t_end)
        self.logger.convert_buffer_to_output()

    def save_result(self, file_name: str) -> None:
        self.logger.log_sim_result(file_name)

    def plot(self) -> None:
        result = plotter.Plotter(self.engine.t_span, self.engine.dt_log)
        result.make_plots(self.logger.output)
        