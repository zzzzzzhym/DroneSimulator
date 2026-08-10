"""Post-fit analysis: score a lookup table's thrust prediction against measured data.

Run after a lookup table is built. Extends the plotting utilities (FitPlotter,
QuadraticFitPlotter) with a quantitative loss: for rotor 0 of each dataset it queries
the lookup table for the predicted force and compares it to the measured force in the
disk frame — the same residual SingleRotorObjective minimizes, so the numbers are
directly comparable to a fit's loss.

Supports several named lookup tables at once so their losses can be compared side by
side on the same data (e.g. a table before vs. after a BEMT refinement, or a quadratic
table vs. a BET table).
"""

import numpy as np
import matplotlib.pyplot as plt

import data_factory


def _residual_disk(dataset: data_factory.FittingDataset, i: int, lookup_table,
                   is_ccw_rotor0: bool, use_sensed_wind: bool) -> np.ndarray:
    """Disk-frame residual (f_lookup_table - f_measured) for rotor 0 at sample i."""
    r_disk = dataset.shared_r_disk[i]
    omega = dataset.omega_0[i]
    if use_sensed_wind:
        f_pred_inertial, _ = lookup_table.get_rotor_forces_sensed_wind(
            dataset.rotor_0_sensed_wind_velocity[i], dataset.v_forward_0[i],
            r_disk, omega, is_ccw_rotor0,
        )
    else:
        f_pred_inertial, _ = lookup_table.get_rotor_forces(
            dataset.u_free_0[i], dataset.v_forward_0[i],
            r_disk, omega, is_ccw_rotor0,
        )
    f_pred_disk = r_disk.T @ f_pred_inertial
    f_measured_disk = r_disk.T @ dataset.rotor_0_f_rotor_inertial_frame[i]
    return f_pred_disk - f_measured_disk


def _normalize_tables(lookup_tables) -> dict:
    """Accept a single table, a list, or a {name: table} dict; return a {name: table} dict."""
    if isinstance(lookup_tables, dict):
        return dict(lookup_tables)
    if isinstance(lookup_tables, (list, tuple)):
        return {str(k): t for k, t in enumerate(lookup_tables)}
    return {"lookup_table": lookup_tables}


class PostAnalysis:
    """Lookup-table-vs-measured loss analysis. Static methods; run after fitting."""

    @staticmethod
    def compute_losses(datasets: list[data_factory.FittingDataset], lookup_table,
                       is_ccw_rotor0: bool, use_sensed_wind: bool = False,
                       sample_step: int = 1) -> list[float]:
        """Mean squared disk-frame residual norm per dataset, for one lookup table.

        Residual is f_lookup_table - f_measured for rotor 0 (full 3 components, unweighted
        — the same metric as SingleRotorObjective), averaged over the sampled indices.

        Args:
            datasets: FittingDataset list (one wind condition each, typically).
            lookup_table: PropellerLookupTable.Reader.
            is_ccw_rotor0: rotor-0 spin direction, needed for the table's fy sign.
            use_sensed_wind: query via sensed wind (get_rotor_forces_sensed_wind) instead
                of background free-stream wind (get_rotor_forces).
            sample_step: stride for subsampling each dataset.

        Returns:
            list[float]: one loss per dataset, in dataset order.
        """
        losses = []
        for dataset in datasets:
            if use_sensed_wind and dataset.rotor_0_sensed_wind_velocity is None:
                raise ValueError(
                    "use_sensed_wind=True but a dataset lacks rotor_0_sensed_wind_velocity."
                )
            indices = range(0, len(dataset), sample_step)
            residuals = [
                _residual_disk(dataset, i, lookup_table, is_ccw_rotor0, use_sensed_wind)
                for i in indices
            ]
            losses.append(float(np.mean([r @ r for r in residuals])))
        return losses

    @staticmethod
    def plot_loss(datasets: list[data_factory.FittingDataset], lookup_tables,
                  is_ccw_rotor0: bool, use_sensed_wind: bool = False,
                  sample_step: int = 1, dataset_labels=None):
        """Compute each lookup table's per-dataset loss and plot it as a grouped bar chart.

        Args:
            datasets: FittingDataset list.
            lookup_tables: a single PropellerLookupTable.Reader, a list of them, or a
                {name: table} dict. Multiple tables are drawn as grouped bars so their
                losses can be compared on the same data.
            is_ccw_rotor0, use_sensed_wind, sample_step: see compute_losses.
            dataset_labels: optional x-axis labels (default: dataset index).

        Returns:
            (fig, losses_by_table): the Figure and a {name: list[float]} dict of losses.
        """
        tables = _normalize_tables(lookup_tables)
        labels = dataset_labels if dataset_labels is not None else [str(i) for i in range(len(datasets))]

        losses_by_table = {
            name: PostAnalysis.compute_losses(datasets, table, is_ccw_rotor0,
                                              use_sensed_wind, sample_step)
            for name, table in tables.items()
        }

        n_tables = max(len(tables), 1)
        x = np.arange(len(datasets))
        width = 0.8 / n_tables

        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(datasets)), 4))
        for k, (name, losses) in enumerate(losses_by_table.items()):
            offset = (k - (n_tables - 1) / 2) * width
            ax.bar(x + offset, losses, width, label=f"{name} (mean={np.mean(losses):.3f})")

        wind_label = "sensed wind" if use_sensed_wind else "background wind"
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean sq. residual (disk frame, N^2)")
        ax.set_title(f"Lookup-table loss vs measured ({wind_label})")
        ax.legend()
        fig.tight_layout()
        return fig, losses_by_table
