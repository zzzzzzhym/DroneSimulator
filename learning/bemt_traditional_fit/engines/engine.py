"""Engine abstraction + shared assembly helpers.

An Engine runs a fitting pipeline end to end: `fit(datasets, init_guess=None)` returns the
fitted result and exposes the fitted `.model`. Concrete engines live one-per-file
(fitting_engine, quadratic_fitting_engine, staged_single_rotor_engine); the factories in
engine_factory build them. assemble_engine / build_seed_gen are the low-level builders both
the factories and the composite (staged) engine share, kept here (neutral) so neither has
to import the other.
"""
from abc import ABC, abstractmethod

from learning.bemt_traditional_fit.fitting_config import FittingConfig
from learning.bemt_traditional_fit.seed_generator import MultiSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class Engine(ABC):
    """Runs a fitting pipeline.

    Implementations set `self.model` (the fitted model, used by post-processing) and run
    the pipeline in fit(). Subclasses may accept extra engine-specific run options beyond
    init_guess (e.g. is_multiseed / is_fine_tune).
    """
    model: object

    @abstractmethod
    def fit(self, datasets, init_guess=None):
        """Fit on datasets, optionally warm-started from a physical init_guess."""


def build_seed_gen(config: FittingConfig) -> MultiSeedGenerator:
    """Multi-start LHS seed generator from the config's seed section."""
    sc = config.seed
    return MultiSeedGenerator(n_lhs=sc.n_lhs, n_keep=sc.n_keep, random_seed=sc.random_seed)


def assemble_engine(model, objective, decision_var, config: FittingConfig):
    """Assemble a FittingEngine (model + objective + decision var + seed gen + solvers)."""
    # Local import: FittingEngine subclasses Engine, so importing it at module top would
    # create an engine <-> fitting_engine cycle.
    from learning.bemt_traditional_fit.engines.fitting_engine import FittingEngine
    bounds = decision_var.bounds
    sv = config.solver
    coarse_solver = Solver(bounds, maxiter=sv.coarse_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    fine_solver = Solver(bounds, maxiter=sv.fine_maxiter, options={'ftol': sv.ftol, 'xtol': sv.xtol, 'disp': True})
    single_solver = Solver(bounds, maxiter=sv.single_maxiter, options={'disp': True, 'fatol': 1e-1})
    single_fine_solver = Solver(bounds, maxiter=sv.single_fine_maxiter, options={'disp': True, 'fatol': 1e-1})
    return FittingEngine(model, objective, decision_var, build_seed_gen(config),
                         coarse_solver, fine_solver, single_solver, single_fine_solver)
