import data_factory
from learning.bemt_traditional_fit.fit_plotter import FitPlotter
from learning.bemt_traditional_fit.post_analysis import PostAnalysis
from learning.bemt_traditional_fit.engine_factory import EngineFactory


def _rotor0_is_ccw(model) -> bool:
    """Rotor-0 spin direction, for either a single-rotor or a full-vehicle model."""
    if hasattr(model, "is_ccw_rotor0"):
        return model.is_ccw_rotor0
    return model.params.is_ccw_blade[0]


class FittingSession:
    """User-facing session. It does NOT build the pipeline and does NOT know how to fit.

    Give it an "order" — a factory from engine_factory — and it asks that factory to
    produce the integrated engine (the pipeline: model + decision var + seed gen + solvers).
    From then on the manager only forwards the per-run inputs (datasets, init_guess) down
    to the engine and hosts post-processing. It owns neither the model/seed gen (the engine
    does) nor the datasets/init_guess (those flow through run()).

    Example:
        from learning.bemt_traditional_fit import engine_factory as ef
        mgr = FittingSession(ef.SingleRotorBandCorrectionFactory(blade, is_ccw_rotor0=False))
        params = mgr.run(datasets, init_guess=init_guess)
    """

    def __init__(self, factory: EngineFactory):
        # Take the order to the factory and get back the integrated engine.
        self.engine = factory.build()

    def run(self, datasets: list[data_factory.FittingDataset], init_guess=None, **run_opts):
        """Fit on datasets. Forwards init_guess and any run options to the engine as-is.

        run_opts are engine-specific (e.g. is_multiseed, is_fine_tune for FittingEngine);
        the manager passes them through without interpreting them.
        """
        return self.engine.fit(datasets, init_guess, **run_opts)

    def plot(self, datasets: list[data_factory.FittingDataset], dataset_idx: int = 0,
             lookup_table=None, is_using_lookup_table: bool = False,
             sample_step: int = 1, params=None, use_sensed_wind: bool = False):
        """Plot fit vs ground truth, and — when a lookup table is given — also run the
        post-fit loss analysis of that table against the datasets.

        Args:
            params: optional parameter vector to apply before plotting; if None, uses
                    whatever is currently set on the engine's model (e.g. from a prior run()).
            use_sensed_wind: forwarded to the post-analysis loss (sensed- vs background-wind
                    query); only used when lookup_table is given.

        Returns:
            Without lookup_table: the FitPlotter fit figures.
            With lookup_table: (fit_figs, loss_fig, losses_by_table), where loss_fig and
            losses_by_table come from PostAnalysis.plot_loss (lookup-table loss over all
            datasets).
        """
        model = self.engine.model
        if params is not None:
            model.apply_params(params)
        fit_figs = FitPlotter.plot_the_fit(
            model,
            datasets[dataset_idx],
            lookup_table=lookup_table,
            is_using_lookup_table=is_using_lookup_table,
            sample_step=sample_step,
        )
        if lookup_table is None:
            return fit_figs
        loss_fig, losses = PostAnalysis.plot_loss(
            datasets, lookup_table,
            is_ccw_rotor0=_rotor0_is_ccw(model),
            use_sensed_wind=use_sensed_wind,
            sample_step=sample_step,
        )
        return fit_figs, loss_fig, losses
