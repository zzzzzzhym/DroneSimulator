import unittest
from unittest.mock import MagicMock

from drone import parameters
from inflow_model.blade_params import APC_8x6
from inflow_model.airfoil.aero_coeff import BandCorrectedAeroCoefficients
from learning.bemt_traditional_fit import engine_factory as ef
from learning.bemt_traditional_fit import decision_var
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.single_rotor_model import SingleRotorBemtModel
from learning.bemt_traditional_fit.objective import FittingObjective
from learning.bemt_traditional_fit.single_rotor_objective import SingleRotorObjective
from learning.bemt_traditional_fit.body_drag_objective import BodyDragObjective
from learning.bemt_traditional_fit.quadratic_rotor_model import QuadraticRotorModel
from learning.bemt_traditional_fit.engines.quadratic_fitting_engine import QuadraticFittingEngine
from learning.bemt_traditional_fit.engines.fitting_engine import FittingEngine
from learning.bemt_traditional_fit.seed_generator import MultiSeedGenerator


class TestFullVehicleFactory(unittest.TestCase):

    def setUp(self):
        self.factory = ef.FullVehicleFactory(APC_8x6(), parameters.PennStateARILab550())

    def test_build_returns_engine_owning_model_and_seedgen(self):
        engine = self.factory.build()
        self.assertIsInstance(engine, FittingEngine)
        self.assertIsInstance(engine.model, BemtModel)
        self.assertIsInstance(engine.seed_gen, MultiSeedGenerator)

    def test_engine_wired_to_model_and_objective(self):
        engine = self.factory.build()
        self.assertIsInstance(engine.model, BemtModel)
        self.assertIsInstance(engine.objective, FittingObjective)

    def test_decision_var_is_full_vehicle(self):
        engine = self.factory.build()
        self.assertEqual(engine.decision_var.search_names, decision_var.full_vehicle().search_names)


class TestSingleRotorFactories(unittest.TestCase):

    def setUp(self):
        self.blade = APC_8x6()

    def test_single_rotor_base(self):
        engine = ef.SingleRotorFactory(self.blade, is_ccw_rotor0=False).build()
        self.assertIsInstance(engine.model, SingleRotorBemtModel)
        self.assertFalse(engine.model.use_bemt)
        self.assertIsInstance(engine.objective, SingleRotorObjective)
        self.assertEqual(len(engine.decision_var.search_names), 6)

    def test_bemt_factory_sets_use_bemt(self):
        engine = ef.SingleRotorBemtFactory(self.blade, is_ccw_rotor0=False).build()
        self.assertTrue(engine.model.use_bemt)

    def test_bemt_factory_inherits_model_and_objective(self):
        # Inheritance from SingleRotorFactory: same model type + objective, only use_bemt differs.
        engine = ef.SingleRotorBemtFactory(self.blade, is_ccw_rotor0=True).build()
        self.assertIsInstance(engine.model, SingleRotorBemtModel)
        self.assertIsInstance(engine.objective, SingleRotorObjective)
        self.assertEqual(len(engine.decision_var.search_names), 6)

    def test_band_correction_decision_var_and_model(self):
        engine = ef.SingleRotorBandCorrectionFactory(self.blade, is_ccw_rotor0=False).build()
        self.assertEqual(len(engine.decision_var.search_names), 20)
        self.assertIsInstance(engine.model.blade.aero_coeff, BandCorrectedAeroCoefficients)

    def test_is_ccw_rotor0_forwarded(self):
        engine = ef.SingleRotorFactory(self.blade, is_ccw_rotor0=True).build()
        self.assertTrue(engine.model.is_ccw_rotor0)


class TestBodyDragFactory(unittest.TestCase):

    def test_body_drag_wiring_and_configure(self):
        lookup_table = MagicMock()
        factory = ef.BodyDragFactory(
            APC_8x6(), parameters.PennStateARILab550(),
            lookup_table=lookup_table, fixed_aero_params=(5.0, 2.0, 1.5, 0.4),
        )
        engine = factory.build()
        self.assertIsInstance(engine.objective, BodyDragObjective)
        self.assertIs(engine.objective.lookup_table, lookup_table)
        # body-drag fit is 1-D (only k_body_drag)
        self.assertEqual(len(engine.decision_var.search_names), 1)
        # configure() froze the aero coefficients at fixed_aero_params
        self.assertAlmostEqual(engine.model.blade.aero_coeff.cl_1, 5.0)
        self.assertAlmostEqual(engine.model.blade.aero_coeff.cd, 1.5)


class TestQuadraticRotorFactory(unittest.TestCase):

    def test_build_returns_quadratic_engine(self):
        engine = ef.QuadraticRotorFactory().build()
        self.assertIsInstance(engine, QuadraticFittingEngine)
        self.assertIsInstance(engine.model, QuadraticRotorModel)

    def test_nominal_grids_forwarded(self):
        engine = ef.QuadraticRotorFactory(nominal_u_free_x=[3.0], nominal_pitch=[0.0]).build()
        self.assertEqual(engine._nominal_u_free_x, [3.0])
        self.assertEqual(engine._nominal_pitch, [0.0])


class TestStagedSingleRotorFactory(unittest.TestCase):

    def test_build_returns_staged_engine_owning_model(self):
        engine = ef.StagedSingleRotorFactory(APC_8x6(), is_ccw_rotor0=False).build()
        self.assertIsInstance(engine, ef.StagedSingleRotorEngine)
        self.assertIsInstance(engine.model, SingleRotorBemtModel)


class TestConfigOverride(unittest.TestCase):

    def test_explicit_config_used(self):
        from learning.bemt_traditional_fit.fitting_config import FittingConfig
        from pathlib import Path
        cfg = FittingConfig.from_yaml(
            Path(ef.__file__).parent / "config_single_rotor.yaml")
        cfg.solver.single_maxiter = 7
        factory = ef.SingleRotorFactory(APC_8x6(), is_ccw_rotor0=False, config=cfg)
        engine = factory.build()
        self.assertEqual(engine.single_solver.options['maxiter'], 7)


if __name__ == "__main__":
    unittest.main()
