import unittest

import numpy as np
import pandas as pd

import data_factory
from learning.bemt_traditional_fit.post_analysis import PostAnalysis


def _make_mock_dataset(n: int, measured_force=(0.0, 0.0, 5.0),
                       with_sensed_wind: bool = True) -> data_factory.FittingDataset:
    """Minimal single-condition dataset with identity disk pose (disk frame == inertial)."""
    measured_force = np.asarray(measured_force, dtype=float)
    d = {}
    for r in range(4):
        d[f"rotor_{r}_local_wind_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
        d[f"rotor_{r}_velocity"] = [np.zeros(3).tolist() for _ in range(n)]
        d[f"rotor_{r}_rotation_spd"] = [300.0 if r == 0 else 0.0] * n
        d[f"rotor_{r}_f_rotor_inertial_frame"] = [measured_force.tolist() for _ in range(n)]
        if with_sensed_wind:
            d[f"rotor_{r}_sensed_wind_velocity"] = [np.array([0.0, 0.0, -2.0]).tolist() for _ in range(n)]
    d["shared_r_disk"] = [np.eye(3).tolist() for _ in range(n)]
    d["sensed_dv"] = [np.zeros(3).tolist() for _ in range(n)]
    d["sensed_omega"] = [np.zeros(3).tolist() for _ in range(n)]
    d["v"] = [np.zeros(3).tolist() for _ in range(n)]
    return data_factory.FittingDataset(pd.DataFrame(d), "mock")


class _FakeTable:
    """Lookup table stub returning a fixed inertial force (and zero v_i)."""

    def __init__(self, force=(0.0, 0.0, 6.0)):
        self.force = np.asarray(force, dtype=float)

    def get_rotor_forces(self, u_free, v_forward, r_disk, omega, is_ccw_blade):
        return self.force.copy(), np.zeros(3)

    def get_rotor_forces_sensed_wind(self, u_sensed, v_forward, r_disk, omega, is_ccw_blade):
        return self.force.copy(), np.zeros(3)


class TestComputeLosses(unittest.TestCase):

    def test_matches_hand_computed_squared_residual(self):
        # measured fz=5, predicted fz=6 → residual=[0,0,1] → r@r = 1 for every sample.
        ds = _make_mock_dataset(10, measured_force=(0.0, 0.0, 5.0))
        losses = PostAnalysis.compute_losses([ds], _FakeTable(force=(0.0, 0.0, 6.0)),
                                             is_ccw_rotor0=False)
        self.assertEqual(len(losses), 1)
        self.assertAlmostEqual(losses[0], 1.0, places=9)

    def test_zero_loss_when_prediction_matches(self):
        ds = _make_mock_dataset(5, measured_force=(0.0, 0.0, 5.0))
        losses = PostAnalysis.compute_losses([ds], _FakeTable(force=(0.0, 0.0, 5.0)),
                                             is_ccw_rotor0=False)
        self.assertAlmostEqual(losses[0], 0.0, places=12)

    def test_one_loss_per_dataset(self):
        datasets = [_make_mock_dataset(4), _make_mock_dataset(4), _make_mock_dataset(4)]
        losses = PostAnalysis.compute_losses(datasets, _FakeTable(), is_ccw_rotor0=False)
        self.assertEqual(len(losses), 3)

    def test_full_vector_residual_counts_all_components(self):
        # measured [0,0,5], predicted [3,4,5] → residual [3,4,0] → r@r = 25.
        ds = _make_mock_dataset(3, measured_force=(0.0, 0.0, 5.0))
        losses = PostAnalysis.compute_losses([ds], _FakeTable(force=(3.0, 4.0, 5.0)),
                                             is_ccw_rotor0=False)
        self.assertAlmostEqual(losses[0], 25.0, places=9)

    def test_sensed_wind_path_used_when_requested(self):
        ds = _make_mock_dataset(4, measured_force=(0.0, 0.0, 5.0), with_sensed_wind=True)
        losses = PostAnalysis.compute_losses([ds], _FakeTable(force=(0.0, 0.0, 6.0)),
                                             is_ccw_rotor0=False, use_sensed_wind=True)
        self.assertAlmostEqual(losses[0], 1.0, places=9)

    def test_sensed_wind_missing_raises(self):
        ds = _make_mock_dataset(4, with_sensed_wind=False)
        with self.assertRaises(ValueError):
            PostAnalysis.compute_losses([ds], _FakeTable(), is_ccw_rotor0=False,
                                        use_sensed_wind=True)

    def test_sample_step_subsamples(self):
        # Constant residual, so subsampling must not change the mean.
        ds = _make_mock_dataset(20, measured_force=(0.0, 0.0, 5.0))
        full = PostAnalysis.compute_losses([ds], _FakeTable(force=(0.0, 0.0, 6.0)),
                                           is_ccw_rotor0=False, sample_step=1)
        strided = PostAnalysis.compute_losses([ds], _FakeTable(force=(0.0, 0.0, 6.0)),
                                              is_ccw_rotor0=False, sample_step=5)
        self.assertAlmostEqual(full[0], strided[0], places=9)


class TestPlotLoss(unittest.TestCase):

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")

    def test_single_table_returns_fig_and_losses(self):
        datasets = [_make_mock_dataset(4), _make_mock_dataset(4)]
        fig, losses_by_table = PostAnalysis.plot_loss(
            datasets, _FakeTable(force=(0.0, 0.0, 6.0)), is_ccw_rotor0=False)
        self.assertEqual(list(losses_by_table.keys()), ["lookup_table"])
        self.assertEqual(len(losses_by_table["lookup_table"]), 2)
        self.assertIsNotNone(fig.axes)

    def test_named_tables_compared(self):
        datasets = [_make_mock_dataset(4, measured_force=(0.0, 0.0, 5.0))]
        tables = {"good": _FakeTable(force=(0.0, 0.0, 5.0)),
                  "bad": _FakeTable(force=(0.0, 0.0, 8.0))}
        fig, losses = PostAnalysis.plot_loss(datasets, tables, is_ccw_rotor0=False)
        self.assertAlmostEqual(losses["good"][0], 0.0, places=12)
        self.assertAlmostEqual(losses["bad"][0], 9.0, places=9)

    def test_list_of_tables_gets_index_names(self):
        datasets = [_make_mock_dataset(4)]
        fig, losses = PostAnalysis.plot_loss(
            datasets, [_FakeTable(), _FakeTable()], is_ccw_rotor0=False)
        self.assertEqual(set(losses.keys()), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
