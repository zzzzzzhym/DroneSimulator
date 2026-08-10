import unittest
from unittest.mock import patch
import numpy as np

from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit import decision_var
from learning.bemt_traditional_fit import engine_factory as ef
from learning.bemt_traditional_fit.engines.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.seed_generator import MultiSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class _FakeModel:
    """Minimal model for engine-level subspace tests (no BET)."""

    def __init__(self):
        self.applied = None

    def adjust_resolution(self, is_fine_tune):
        pass

    def apply_params(self, x):
        self.applied = np.asarray(x, dtype=float)


class _QuadraticObjective:
    """Loss = sum((physical - target)^2); minimized at target."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def get_loss(self, physical, datasets):
        d = np.asarray(physical, dtype=float) - self.target
        return float(d @ d)


def _simple_engine(objective, bounds):
    names = tuple(f"p{i}" for i in range(len(bounds)))
    search_space = decision_var.DecisionVar(
        search_names=names, physical_names=names, bounds=bounds,
    )
    s = lambda: Solver(bounds, options={'disp': False, 'fatol': 1e-6})
    seed_gen = MultiSeedGenerator(n_lhs=1, n_keep=1, random_seed=0)
    return FittingEngine(_FakeModel(), objective, search_space, seed_gen, s(), s(), s(), s())


class TestFitSingleSubspace(unittest.TestCase):
    """fit_single restricted to a subset of the decision vector via free_indices."""

    def test_frozen_indices_unchanged_free_index_optimized(self):
        bounds = [(0.0, 10.0)] * 3
        engine = _simple_engine(_QuadraticObjective([2.0, 5.0, 8.0]), bounds)
        result = engine.fit_single([], [0.0, 0.0, 0.0], is_fine_tune=True, free_indices=[1])
        # Only index 1 was free: 0 and 2 stay at the seed, 1 moves toward 5.
        self.assertAlmostEqual(result[0], 0.0, places=6)
        self.assertAlmostEqual(result[2], 0.0, places=6)
        self.assertAlmostEqual(result[1], 5.0, places=1)

    def test_multiple_free_indices_optimized(self):
        bounds = [(0.0, 10.0)] * 3
        engine = _simple_engine(_QuadraticObjective([2.0, 5.0, 8.0]), bounds)
        result = engine.fit_single([], np.zeros(3), is_fine_tune=True, free_indices=[0, 2])
        self.assertAlmostEqual(result[1], 0.0, places=6)   # frozen
        self.assertAlmostEqual(result[0], 2.0, places=1)
        self.assertAlmostEqual(result[2], 8.0, places=1)

    def test_no_free_indices_optimizes_everything(self):
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        result = engine.fit_single([], np.zeros(2), is_fine_tune=True)
        self.assertAlmostEqual(result[0], 3.0, places=1)
        self.assertAlmostEqual(result[1], 4.0, places=1)

    def test_result_applied_to_model(self):
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        result = engine.fit_single([], np.zeros(2), is_fine_tune=True)
        np.testing.assert_array_equal(engine.model.applied, result)

    def test_missing_init_guess_raises(self):
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), [(0.0, 10.0)] * 2)
        with self.assertRaises(ValueError):
            engine.fit_single([], None)

    def test_budget_comes_from_config_solver(self):
        # The reduced solver inherits the (config-built) template solver's maxiter.
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        engine.single_fine_solver = Solver(bounds, maxiter=7, options={'disp': False})
        captured = {}
        real_run = Solver.run

        def spy(self, *a, **k):
            captured['maxiter'] = self.options['maxiter']
            return real_run(self, *a, **k)

        with patch.object(Solver, 'run', spy):
            engine.fit_single([], np.zeros(2), is_fine_tune=True, free_indices=[0])
        self.assertEqual(captured['maxiter'], 7)


class TestMultiseedInitGuess(unittest.TestCase):
    """fit(..., is_multiseed=True) seeds the owned seed_gen from init_guess."""

    def test_init_guess_sets_physical_seed(self):
        bounds = [(0.0, 10.0)] * 2
        engine = _simple_engine(_QuadraticObjective([3.0, 4.0]), bounds)
        with patch.object(engine, '_screen_seeds', side_effect=StopIteration):
            try:
                engine.fit([], init_guess=[3.0, 4.0], is_multiseed=True)
            except StopIteration:
                pass
        np.testing.assert_allclose(engine.seed_gen.physical_seed, [3.0, 4.0])


class TestStagedEngine(unittest.TestCase):
    """The three-stage pipeline is now an engine (StagedSingleRotorEngine)."""

    def setUp(self):
        self.blade = APC_8x6()

    def test_build_owns_model_no_engine_yet(self):
        engine = ef.StagedSingleRotorFactory(self.blade, is_ccw_rotor0=False).build()
        self.assertIsInstance(engine, ef.StagedSingleRotorEngine)

    def test_three_stage_sequence(self):
        staged = ef.StagedSingleRotorFactory(self.blade, is_ccw_rotor0=False).build()
        physical_base = np.array([5.0, 2.0, 1.0, 0.4, 0.0, 0.0])
        single_calls = []

        def fake_fit(self, datasets, init_guess=None, is_multiseed=True, is_fine_tune=False):
            # stage 1 (multiseed, BET)
            single_calls.append({'kind': 'multiseed', 'use_bemt': staged.model.use_bemt})
            return physical_base.copy()

        def fake_fit_single(self, datasets, init_guess, is_fine_tune=False, free_indices=None):
            single_calls.append({
                'kind': 'single',
                'seed_len': len(np.asarray(init_guess)),
                'free_indices': list(free_indices) if free_indices is not None else None,
                'is_fine_tune': is_fine_tune,
                'use_bemt': staged.model.use_bemt,
            })
            return np.zeros(len(np.asarray(init_guess)))

        with patch.object(FittingEngine, 'fit', fake_fit), \
             patch.object(FittingEngine, 'fit_single', fake_fit_single), \
             patch.object(staged.model, 'configure_for_band_correction_fit') as mock_cfg:
            result = staged.fit([], init_guess=None)

        mock_cfg.assert_called_once()
        # Stage 1: multiseed, BET (use_bemt False)
        self.assertEqual(single_calls[0]['kind'], 'multiseed')
        self.assertFalse(single_calls[0]['use_bemt'])
        # Stage 1.5: single, all 6 base free, coarse, BEMT on
        self.assertEqual(single_calls[1]['kind'], 'single')
        self.assertIsNone(single_calls[1]['free_indices'])
        self.assertEqual(single_calls[1]['seed_len'], 6)
        self.assertFalse(single_calls[1]['is_fine_tune'])
        self.assertTrue(single_calls[1]['use_bemt'])
        # Stage 2: single, only the 14 offsets free, 20-vector seed, fine, BEMT on
        self.assertEqual(single_calls[2]['free_indices'], list(range(6, 20)))
        self.assertEqual(single_calls[2]['seed_len'], 20)
        self.assertTrue(single_calls[2]['is_fine_tune'])
        self.assertTrue(single_calls[2]['use_bemt'])

        self.assertEqual(len(result), 20)

    def test_manager_runs_staged_engine(self):
        from learning.bemt_traditional_fit.session import FittingSession
        mgr = FittingSession(ef.StagedSingleRotorFactory(self.blade, is_ccw_rotor0=False))
        with patch.object(mgr.engine, 'fit', return_value=np.zeros(20)) as mock_fit:
            out = mgr.run([], init_guess=None)
        mock_fit.assert_called_once_with([], None)
        self.assertEqual(len(out), 20)


if __name__ == "__main__":
    unittest.main()
