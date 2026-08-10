"""Three-stage single-rotor fitting pipeline, as an Engine."""
import numpy as np

from learning.bemt_traditional_fit import decision_var
from learning.bemt_traditional_fit.engines.engine import Engine, assemble_engine
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective


class StagedSingleRotorEngine(Engine):
    """Runs the three-stage single-rotor pipeline, reusing ONE model across stages:

      Stage 1   — BET (sensed-wind v_i); global multi-start over the six base coeffs.
      Stage 1.5 — BEMT (root-found v_i); coarse single-start refine of the same six.
      Stage 2   — BEMT; freeze the base coeffs, fit only the CL/CD correction offsets.

    A composite engine: each stage runs a FittingEngine built via assemble_engine. It is
    handed the (single-rotor) model and config directly, so it stays independent of
    engine_factory.
    """

    def __init__(self, model, config):
        self.model = model
        self._config = config

    def _objective(self):
        return SingleRotorObjective(self.model)

    def fit(self, datasets, init_guess=None):
        model = self.model
        config = self._config
        base_dv = decision_var.OffsetAugmentedAeroCoeffDecisionVar()
        n_base = len(base_dv.search_names)

        # Stage 1: BET (measured v_i) — global multi-start over the six base coefficients.
        print("\n===== Stage 1: BET global fit (base aero coefficients) =====")
        model.use_bemt = False
        base_engine = assemble_engine(model, self._objective(), base_dv, config)
        physical_base = base_engine.fit(datasets, init_guess=init_guess, is_multiseed=True)

        # Stage 1.5: BEMT (root-found v_i) — refine the same six coefficients.
        print("\n===== Stage 1.5: BEMT refine (base coefficients) =====")
        model.use_bemt = True
        physical_base = base_engine.fit_single(datasets, physical_base, is_fine_tune=False)

        # Stage 2: BEMT — freeze base coefficients, fit only the correction offsets.
        print("\n===== Stage 2: BEMT fine-tune (CL/CD correction offsets only) =====")
        band_dv = decision_var.BandCorrectedDecisionVar()
        model.configure_for_band_correction_fit(band_dv.cl_knots, band_dv.cd_knots)
        band_engine = assemble_engine(model, self._objective(), band_dv, config)
        n_offsets = len(band_dv.search_names) - n_base
        seed_physical = np.concatenate([physical_base, np.zeros(n_offsets)])
        offset_indices = range(n_base, len(band_dv.search_names))
        return band_engine.fit_single(datasets, seed_physical, is_fine_tune=True,
                                      free_indices=offset_indices)
