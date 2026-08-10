"""Composition root: factories that wire a fitting variant into an engine.

Each EngineFactory subclass owns the concrete model / objective / decision-variable
choices for one fitting variant and assembles them into a FittingEngine via build().
This is the only module that imports the concrete model and objective classes;
FittingSession depends on these factories, not on the concrete types.

The quadratic path (QuadraticRotorFactory) has a different engine shape — closed-form,
no solvers or seed generator — so it does not share the EngineFactory skeleton; it
exposes the same build() -> engine contract.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import common_utils.file_manager
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.objective import FittingObjective
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective
from learning.bemt_traditional_fit.body_drag_objective import BodyDragObjective
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.engines.quadratic_fitting_engine import QuadraticFittingEngine
from learning.bemt_traditional_fit.engines.staged_single_rotor_engine import StagedSingleRotorEngine
from learning.bemt_traditional_fit.fitting_config import FittingConfig
from learning.bemt_traditional_fit.engines.engine import Engine, assemble_engine
from learning.bemt_traditional_fit import decision_var

_CONFIG_DIR = Path(common_utils.file_manager.find_path_to_folder(["learning", "bemt_traditional_fit"]))


class EngineFactory(ABC):
    """Builds the engine for one iterative-BEMT fitting variant.

    Subclasses supply the model, decision variable and objective (make_* hooks) plus an
    optional configure() step; the build() skeleton assembles them with the config-derived
    solvers and a seed generator. Each subclass's __init__ takes exactly the inputs that
    variant needs.
    """
    config_file: str = None  # subclasses set this

    def __init__(self, config: FittingConfig = None):
        self.config = config or FittingConfig.from_yaml(_CONFIG_DIR / self.config_file)

    @abstractmethod
    def make_model(self):
        """Construct the (unconfigured) model for this variant."""

    @abstractmethod
    def make_decision_var(self):
        """Construct the decision variable (search space) for this variant."""

    @abstractmethod
    def make_objective(self, model):
        """Construct the loss objective bound to the given model."""

    def configure(self, model, decision_var):
        """Optional post-construction model setup (default: none)."""

    def build(self) -> Engine:
        """Build and return the integrated engine (the pipeline) for this variant."""
        model = self.make_model()
        decision_var = self.make_decision_var()
        self.configure(model, decision_var)
        objective = self.make_objective(model)
        return assemble_engine(model, objective, decision_var, self.config)


class FullVehicleFactory(EngineFactory):
    """Full four-rotor vehicle fit: cl_1, cl_2, cd, alpha_0, k_body_drag."""
    config_file = "config_full_vehicle.yaml"

    def __init__(self, blade, params, config: FittingConfig = None):
        super().__init__(config)
        self.blade = blade
        self.params = params

    def make_model(self):
        return BemtModel(self.blade, self.params, model_config=self.config.model)

    def make_decision_var(self):
        return decision_var.full_vehicle()

    def make_objective(self, model):
        return FittingObjective(model)


class SingleRotorFactory(EngineFactory):
    """Single-rotor test-stand fit (BET, sensed wind): offset-augmented aero coefficients.

    Base class for the single-rotor family; subclasses only override what differs.
    """
    config_file = "config_single_rotor.yaml"

    def __init__(self, blade, is_ccw_rotor0: bool, config: FittingConfig = None):
        super().__init__(config)
        self.blade = blade
        self.is_ccw_rotor0 = is_ccw_rotor0

    def make_model(self):
        return SingleRotorBemtModel(self.blade, self.is_ccw_rotor0, model_config=self.config.model)

    def make_decision_var(self):
        return decision_var.OffsetAugmentedAeroCoeffDecisionVar()

    def make_objective(self, model):
        return SingleRotorObjective(model)


class SingleRotorBemtFactory(SingleRotorFactory):
    """Single-rotor fit using root-found v_i (BEMT) instead of sensed-wind BET."""

    def configure(self, model, decision_var):
        model.use_bemt = True


class SingleRotorBandCorrectionFactory(SingleRotorFactory):
    """Single-rotor fit plus a bounded piecewise-linear CL/CD correction band."""

    def make_decision_var(self):
        return decision_var.BandCorrectedDecisionVar()

    def configure(self, model, decision_var):
        model.configure_for_band_correction_fit(decision_var.cl_knots, decision_var.cd_knots)


class BodyDragFactory(EngineFactory):
    """Body-drag-only fit (k_body_drag) with frozen aero coefficients and a lookup table."""
    config_file = "config_body_drag.yaml"

    def __init__(self, blade, params, lookup_table, fixed_aero_params, config: FittingConfig = None):
        super().__init__(config)
        self.blade = blade
        self.params = params
        self.lookup_table = lookup_table
        self.fixed_aero_params = fixed_aero_params

    def make_model(self):
        return BemtModel(self.blade, self.params, model_config=self.config.model)

    def make_decision_var(self):
        return decision_var.body_drag()

    def make_objective(self, model):
        return BodyDragObjective(model, self.lookup_table)

    def configure(self, model, decision_var):
        model.blade.aero_coeff.update_fit_params(*self.fixed_aero_params)
        model.configure_for_body_drag_fit()


class QuadraticRotorFactory:
    """Quadratic per-condition fit (closed-form least squares).

    Different engine shape from the EngineFactory family (no solvers / decision var /
    seed generator); it keeps the same build() -> engine contract.
    """

    def __init__(self, nominal_u_free_x=None, nominal_pitch=None):
        self.nominal_u_free_x = nominal_u_free_x
        self.nominal_pitch = nominal_pitch

    def build(self):
        model = QuadraticRotorModel()
        return QuadraticFittingEngine(model, self.nominal_u_free_x, self.nominal_pitch)


class StagedSingleRotorFactory:
    """Builds a StagedSingleRotorEngine (the three-stage single-rotor pipeline)."""

    def __init__(self, blade, is_ccw_rotor0: bool, config: FittingConfig = None):
        self._base = SingleRotorFactory(blade, is_ccw_rotor0, config)

    def build(self) -> StagedSingleRotorEngine:
        return StagedSingleRotorEngine(self._base.make_model(), self._base.config)
