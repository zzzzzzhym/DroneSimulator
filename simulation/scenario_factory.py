class Factory:
    def __init__(self, scenario):
        self.scenario = scenario

    def create(self):
        pass
        
class Scenario:
    def __init__(self):
        self.environment_assembly = None    # assemble drone type, disturbance model, dynamics model
        self.controller_assembly = None  # assemble drone type, controller and disturbance estimator
        self.trajecotry = None  

    def run(self):
        raise NotImplementedError

