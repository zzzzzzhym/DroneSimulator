"""Engine for QuadraticRotorModel — fits one quadratic profile per wind condition."""

from pathlib import Path

import numpy as np

import data_factory
from learning.bemt_traditional_fit.engines.engine import Engine
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.quadratic_solver import QuadraticSolver
from learning.bemt_traditional_fit.per_condition_result import (
    PerConditionResult, extract_wind_condition,
)


class QuadraticFittingEngine(Engine):
    """Fits an independent quadratic profile for every wind condition (dataset).

    Owns the per-condition loop; fit(datasets) returns a list of PerConditionResult.
    """

    def __init__(self, model: QuadraticRotorModel,
                 nominal_u_free_x=None, nominal_pitch=None):
        self.model = model
        self._solver = QuadraticSolver()
        # Optional nominal sweep grid; when set, each condition's label is snapped
        # onto it so the lookup table ends up on a clean regular grid.
        self._nominal_u_free_x = nominal_u_free_x
        self._nominal_pitch = nominal_pitch

    def fit(self, datasets: list[data_factory.FittingDataset],
            init_guess=None) -> list[PerConditionResult]:
        """Fit one quadratic per dataset; return a list of PerConditionResult.

        init_guess is accepted (the manager always passes it) but ignored — the
        closed-form least-squares solve has no warm start. Other run options are NOT
        accepted: passing e.g. is_multiseed here is meaningless and should fail loudly.
        """
        results = []
        for idx, dataset in enumerate(datasets):
            u_free_x, pitch = extract_wind_condition(
                dataset, self._nominal_u_free_x, self._nominal_pitch)
            name = Path(dataset.path_to_data_file).name
            print(f"Fitting condition {idx + 1}/{len(datasets)} [{name}]:  "
                  f"u_free_x={u_free_x:.3f} m/s  pitch={np.degrees(pitch):.1f} deg")
            params = self._solver.run(self.model, dataset)
            results.append(PerConditionResult(params=params, u_free_x=u_free_x,
                                              pitch=pitch, dataset=dataset))
        return results
