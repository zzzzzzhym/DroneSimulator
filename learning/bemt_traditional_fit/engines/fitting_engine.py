import numpy as np

import data_factory
from learning.bemt_traditional_fit.engines.engine import Engine
from learning.bemt_traditional_fit.seed_generator import MultiSeedGenerator, SingleSeedGenerator
from learning.bemt_traditional_fit.solver import Solver


class FittingEngine(Engine):
    """The fitting pipeline: owns the model, objective, decision space, seed generator and
    the config-built solvers, and runs them end to end.

    Callers hand it only the per-run inputs — the datasets to fit and an optional physical
    init_guess — via fit(); the engine supplies everything else from what it owns. The
    multi-start seed generator is owned here (not passed in); a single-start run builds its
    own SingleSeedGenerator from init_guess internally.
    """

    def __init__(self, model, objective, decision_var,
                 seed_gen: MultiSeedGenerator,
                 coarse_solver: Solver,
                 fine_solver: Solver,
                 single_solver: Solver,
                 single_fine_solver: Solver):
        self.model = model
        self.objective = objective
        self.decision_var = decision_var
        self.seed_gen = seed_gen
        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver
        self.single_solver = single_solver
        self.single_fine_solver = single_fine_solver

    def _loss(self, search_x, datasets):
        """Evaluate the objective on a search-space vector (converted to physical)."""
        return self.objective.get_loss(self.decision_var.to_physical(search_x), datasets)

    def _print_result(self, label: str, loss: float, x):
        print(f"{label}: loss={loss:.4f}  {self.decision_var.format(x)}")

    def _screen_seeds(self, seeds, datasets, n_keep):
        """Evaluate each seed once and return the n_keep lowest-loss ones."""
        evaluated = []
        for seed in seeds:
            loss = self._loss(seed, datasets)
            evaluated.append((loss, seed.copy()))
        evaluated.sort(key=lambda item: item[0])
        return evaluated[:n_keep]

    def fit(self, datasets: list[data_factory.FittingDataset], init_guess=None,
            is_multiseed: bool = True, is_fine_tune: bool = False) -> np.ndarray:
        """Run the pipeline.

        Args:
            datasets: data to fit.
            init_guess: optional physical-coefficient warm start. In multi-start it is
                prepended to the LHS seeds; in single-start it IS the start.
            is_multiseed: True → global LHS screen → coarse → fine; False → single-start.
            is_fine_tune: single-start only; selects the fine solver + resolution.
        """
        if not is_multiseed:
            return self.fit_single(datasets, init_guess, is_fine_tune=is_fine_tune)
        return self._fit_multiseed(datasets, init_guess)

    def _fit_multiseed(self, datasets: list[data_factory.FittingDataset],
                       init_guess=None) -> np.ndarray:
        """Three-stage multistart fitting: LHS screening → coarse → fine-tune."""
        # Stage 1: inexpensive global screening
        self.model.adjust_resolution(is_fine_tune=False)
        self.seed_gen.physical_seed = (
            self.decision_var.to_search(init_guess) if init_guess is not None else None
        )
        seeds = self.seed_gen.get_seeds(self.decision_var.bounds)
        selected = self._screen_seeds(seeds, datasets, self.seed_gen.n_keep)
        for rank, (loss, seed) in enumerate(selected, start=1):
            self._print_result(f"Selected seed {rank}", loss, seed)

        # Stage 2: coarse local optimization from the best seeds
        coarse_results = []
        for candidate, (_, seed) in enumerate(selected, start=1):
            print(f"Evaluating: {self.decision_var.format(seed)}")
            result = self.coarse_solver.run(lambda x: self._loss(x, datasets), seed)
            coarse_results.append((candidate, result))
        coarse_results.sort(key=lambda item: item[1].fun)
        best_candidate = coarse_results[0][0]
        best_coarse = coarse_results[0][1]

        print("\nCoarse tune results:")
        print("candidate | final loss | (best)")
        for candidate, result in sorted(coarse_results, key=lambda item: item[0]):
            marker = " | (best)" if candidate == best_candidate else ""
            print(f"{candidate} | {result.fun:.4f}{marker}")

        self._print_result("Best coarse result", best_coarse.fun, best_coarse.x_physical)

        # Stage 3: further iterate only the best coarse result
        self.model.adjust_resolution(is_fine_tune=True)
        fine_result = self.fine_solver.run(lambda x: self._loss(x, datasets), best_coarse.x_physical)
        self._print_result("Final result", fine_result.fun, fine_result.x_physical)
        physical = self.decision_var.to_physical(fine_result.x_physical)
        self.model.apply_params(physical)
        return physical

    def fit_single(self, datasets: list[data_factory.FittingDataset],
                   init_guess, is_fine_tune: bool = False, free_indices=None) -> np.ndarray:
        """Single-start fitting from a physical init_guess.

        free_indices optionally restricts the optimization to a subset of the decision
        vector (None → vary everything); the remaining components are held fixed at the
        seed (e.g. freeze the base aero coefficients and vary only the CL/CD correction
        offsets).

        Solver (iteration budget, tolerances) and BET resolution come from config via
        is_fine_tune: the single-fine solver and fine resolution when True, else the coarse
        single solver and coarse resolution.
        """
        if init_guess is None:
            raise ValueError("fit_single requires an init_guess (the single start point).")
        self.model.adjust_resolution(is_fine_tune)
        template = self.single_fine_solver if is_fine_tune else self.single_solver

        seed_gen = SingleSeedGenerator(init_guess, free_indices=free_indices)
        seed_search = np.asarray(
            self.decision_var.to_search(seed_gen.get_seeds(self.decision_var.bounds)[0]),
            dtype=float,
        )
        free_indices = seed_gen.free_indices
        if free_indices is None:
            free_indices = range(len(seed_search))
        free_indices = list(free_indices)

        def embed(free_x):
            full = seed_search.copy()
            full[free_indices] = free_x
            return full

        # Reduced-dimension solver over just the free components; dimension depends on the
        # free set, so it can't be one of the pre-built full-dim solvers. Reuse the template
        # solver's method/options (the config-derived budget and tolerances).
        free_bounds = [self.decision_var.bounds[i] for i in free_indices]
        solver = Solver(free_bounds, method=template.method, options=dict(template.options))

        if len(free_indices) < len(seed_search):
            print("Single-start fit — free: "
                  + ", ".join(self.decision_var.search_names[i] for i in free_indices))
        print("Initial guess: " + self.decision_var.format(seed_search))

        step_counter = {"count": 0}

        def callback(zk):
            free_x = solver.denormalize(zk)
            step_counter["count"] += 1
            print(f"Step {step_counter['count']:3d}: {self.decision_var.format(embed(free_x))}")

        result = solver.run(lambda free_x: self._loss(embed(free_x), datasets),
                            seed_search[free_indices], callback=callback)

        full_search = embed(result.x_physical)
        physical = self.decision_var.to_physical(full_search)
        if not result.success:
            # Nelder-Mead commonly reports success=False when it hits maxiter; the best
            # point found is still useful (and a staged pipeline must carry it forward).
            print("Optimization did not fully converge:", result.message)
        self._print_result("Fitted parameters", result.fun, full_search)
        self.model.apply_params(physical)
        return physical
