import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit import engine_factory as ef
from learning.bemt_traditional_fit.engines.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.session import FittingSession


class _FakeFactory:
    """Stand-in order whose build() returns a supplied engine."""

    def __init__(self, engine):
        self._engine = engine
        self.built = 0

    def build(self):
        self.built += 1
        return self._engine


class TestSessionIsThin(unittest.TestCase):
    """The manager takes an order (factory), gets the engine, and only forwards run()."""

    def test_constructor_gets_engine_from_factory(self):
        engine = MagicMock()
        factory = _FakeFactory(engine)
        manager = FittingSession(factory)
        self.assertEqual(factory.built, 1)
        self.assertIs(manager.engine, engine)

    def test_manager_does_not_own_model_or_datasets(self):
        # The passive 5-field holder is gone; only the engine is referenced.
        manager = FittingSession(_FakeFactory(MagicMock()))
        self.assertFalse(hasattr(manager, "datasets"))
        self.assertFalse(hasattr(manager, "model"))
        self.assertFalse(hasattr(manager, "seed_gen"))
        self.assertFalse(hasattr(manager, "init_guess"))

    def test_run_forwards_datasets_and_init_guess_to_engine(self):
        engine = MagicMock()
        engine.fit.return_value = "result"
        manager = FittingSession(_FakeFactory(engine))
        datasets = [object()]
        out = manager.run(datasets, init_guess=[1.0, 2.0], is_multiseed=False)
        engine.fit.assert_called_once_with(datasets, [1.0, 2.0], is_multiseed=False)
        self.assertEqual(out, "result")

    def test_run_defaults_init_guess_none(self):
        engine = MagicMock()
        manager = FittingSession(_FakeFactory(engine))
        datasets = [object()]
        manager.run(datasets)
        engine.fit.assert_called_once_with(datasets, None)

    def test_plot_uses_engine_model(self):
        engine = MagicMock()
        manager = FittingSession(_FakeFactory(engine))
        fake_dataset = MagicMock()
        fake_figs = (MagicMock(), MagicMock())
        with patch('learning.bemt_traditional_fit.fit_plotter.FitPlotter.plot_the_fit',
                   return_value=fake_figs) as mock_plot:
            result = manager.plot([fake_dataset], dataset_idx=0, sample_step=2)
        mock_plot.assert_called_once_with(
            engine.model, fake_dataset,
            lookup_table=None, is_using_lookup_table=False, sample_step=2,
        )
        self.assertIs(result, fake_figs)

    def test_plot_without_lookup_table_skips_post_analysis(self):
        manager = FittingSession(_FakeFactory(MagicMock()))
        with patch('learning.bemt_traditional_fit.fit_plotter.FitPlotter.plot_the_fit',
                   return_value="fit"), \
             patch('learning.bemt_traditional_fit.post_analysis.PostAnalysis.plot_loss') as mock_loss:
            result = manager.plot([MagicMock()])
        mock_loss.assert_not_called()
        self.assertEqual(result, "fit")

    def test_plot_with_lookup_table_adds_post_analysis(self):
        engine = MagicMock()
        engine.model.is_ccw_rotor0 = False
        manager = FittingSession(_FakeFactory(engine))
        lut = MagicMock()
        datasets = [MagicMock(), MagicMock()]
        loss_ret = ("loss_fig", {"lookup_table": [1.0, 2.0]})
        with patch('learning.bemt_traditional_fit.fit_plotter.FitPlotter.plot_the_fit',
                   return_value="fit") as mock_fit, \
             patch('learning.bemt_traditional_fit.post_analysis.PostAnalysis.plot_loss',
                   return_value=loss_ret) as mock_loss:
            fit_figs, loss_fig, losses = manager.plot(
                datasets, lookup_table=lut, sample_step=3, use_sensed_wind=True)
        self.assertEqual(fit_figs, "fit")
        self.assertEqual(loss_fig, "loss_fig")
        self.assertEqual(losses, {"lookup_table": [1.0, 2.0]})
        mock_loss.assert_called_once_with(
            datasets, lut, is_ccw_rotor0=False, use_sensed_wind=True, sample_step=3)


class TestSessionWithRealFactories(unittest.TestCase):
    """End-to-end construction through the real factories (no fitting run)."""

    def test_single_rotor_order_builds_fitting_engine(self):
        manager = FittingSession(ef.SingleRotorFactory(APC_8x6(), is_ccw_rotor0=False))
        self.assertIsInstance(manager.engine, FittingEngine)

    def test_full_vehicle_order_builds_fitting_engine(self):
        manager = FittingSession(
            ef.FullVehicleFactory(APC_8x6(), parameters.PennStateARILab550()))
        self.assertIsInstance(manager.engine, FittingEngine)

    def test_run_single_delegates_through_to_engine_fit(self):
        manager = FittingSession(ef.SingleRotorFactory(APC_8x6(), is_ccw_rotor0=False))
        with patch.object(manager.engine, 'fit', return_value=np.zeros(6)) as mock_fit:
            manager.run([], init_guess=None, is_multiseed=True)
        mock_fit.assert_called_once_with([], None, is_multiseed=True)


if __name__ == "__main__":
    unittest.main()
