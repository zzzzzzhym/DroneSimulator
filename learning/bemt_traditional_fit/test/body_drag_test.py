import unittest
from unittest.mock import MagicMock
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel
from learning.bemt_traditional_fit.body_drag_objective import BodyDragObjective
from learning.bemt_traditional_fit.fitting_config import ModelConfig

# Body-drag engine *wiring* (factory) is covered in engine_factory_test.TestBodyDragFactory;
# this file focuses on the BodyDragObjective loss computation.


def _make_model_config():
    return ModelConfig(
        coarse_n_elements=2, coarse_n_rotation_segments=6, coarse_sample_distance=100,
        fine_n_elements=20, fine_n_rotation_segments=18, fine_sample_distance=20,
    )


def _make_model():
    return BemtModel(APC_8x6(), parameters.PennStateARILab550(), _make_model_config())


def _make_mock_dataset(n=10):
    """Minimal FittingDataset-like mock with n samples."""
    ds = MagicMock()
    ds.u_free_0 = np.zeros((n, 3))
    ds.u_free_1 = np.zeros((n, 3))
    ds.u_free_2 = np.zeros((n, 3))
    ds.u_free_3 = np.zeros((n, 3))
    return ds


class TestBodyDragObjectiveGetLoss(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.model.configure_for_body_drag_fit()
        self.lookup_table = MagicMock()
        self.objective = BodyDragObjective(self.model, self.lookup_table)

    def test_sets_k_body_drag_from_x(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        self.objective.get_loss([3.7], [dataset])
        self.assertAlmostEqual(self.model.k_body_drag, 3.7)

    def test_calls_get_residual_force_with_lookup_table(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        self.objective.get_loss([1.0], [dataset])
        for call in self.model.get_residual_force.call_args_list:
            _, kwargs = call
            self.assertIs(kwargs.get('lookup_table') or call[0][2], self.lookup_table)
            self.assertTrue(kwargs.get('is_using_lookup_table', False) or call[0][3])

    def test_loss_is_nonnegative(self):
        self.model.get_residual_force = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        dataset = _make_mock_dataset(n=10)
        loss = self.objective.get_loss([1.0], [dataset])
        self.assertGreaterEqual(loss, 0.0)

    def test_zero_residual_gives_zero_loss(self):
        self.model.get_residual_force = MagicMock(return_value=np.zeros(3))
        dataset = _make_mock_dataset(n=10)
        loss = self.objective.get_loss([0.5], [dataset])
        self.assertAlmostEqual(loss, 0.0)

    def test_loss_averages_over_datasets(self):
        residual_a = np.array([0.0, 0.0, 2.0])
        residual_b = np.array([0.0, 0.0, 4.0])
        ds_a = _make_mock_dataset(n=self.model.sample_distance)
        ds_b = _make_mock_dataset(n=self.model.sample_distance)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return residual_a if call_count[0] <= 1 else residual_b

        self.model.get_residual_force = MagicMock(side_effect=side_effect)
        loss = self.objective.get_loss([1.0], [ds_a, ds_b])
        # loss_a = 4.0, loss_b = 16.0, averaged = 10.0
        self.assertAlmostEqual(loss, 10.0)


if __name__ == "__main__":
    unittest.main()
