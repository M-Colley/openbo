"""Benchmark MO-TAF source references: own front vs target incumbent, with/without decay.

Compares ``source_reference_mode="front"`` (each source measures hypervolume improvement
against its own observed Pareto front) with ``"target_incumbent"`` (against the front of
its predictions at the target's observed points, as in Wistuba et al.'s TAF), each with
and without the population-weight decay, plus plain MOBO as the baseline.

Task family: a BoTorch multi-objective test problem (negated, so every objective is
maximized) composed with a monotone power warp of each input, x -> x ** a. The target
uses a = 1. A "related" pool holds three sources with a drawn log-uniformly from
[0.75, 1.33] per dimension; the "mixed" pool keeps two of them and adds one misleading
source (a = 0.35) whose Pareto set lies elsewhere. Each source trajectory mimics a BO
run: random exploration plus up to 20 near-front points, with per-objective
Matern-5/2 hyperparameters fitted by BoTorch and exported in the MO-TAF artifact schema.

Results are written one JSON file per run, so an interrupted benchmark resumes where it
stopped. Example:

    uv run python scripts/benchmark_mo_taf_reference.py --out-dir test_results/mo_taf_reference
    uv run python scripts/benchmark_mo_taf_reference.py --out-dir test_results/mo_taf_reference --summarize-only
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions.multi_objective import DTLZ2, BraninCurrin
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from openbo.optimizers.mobo_botorch import pareto_front, run_mobo_botorch
from openbo.optimizers.mobo_taf import run_mobo_taf

PROBLEMS = ["branincurrin", "dtlz2"]
SCENARIOS = ["related", "mixed"]
METHODS = [
    "front+decay",
    "front+nodecay",
    "target_incumbent+nodecay",
    "target_incumbent+decay",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MO-TAF source-reference modes against plain MOBO."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("test_results/mo_taf_reference"))
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--decay-rate", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--summarize-only", action="store_true", help="Skip running; summarize saved runs."
    )
    return parser.parse_args()


def make_problem(name: str):
    """Negated (maximization) BoTorch problem and its reference point."""
    if name == "branincurrin":
        prob = BraninCurrin(negate=True).to(torch.double)
    elif name == "dtlz2":
        prob = DTLZ2(dim=4, num_objectives=2, negate=True).to(torch.double)
    else:
        raise ValueError(f"unknown problem {name!r}")
    return prob, prob.ref_point.tolist()


def task_fn(name: str, warp: np.ndarray):
    """Family member: the problem evaluated at x ** warp."""
    prob, _ = make_problem(name)
    warp = np.asarray(warp, dtype=np.float64)

    def f(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.atleast_2d(x), 0.0, 1.0) ** warp
        with torch.no_grad():
            return prob(torch.tensor(x, dtype=torch.double)).numpy()

    return f


def fit_gp_state(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit one ScaleKernel(Matern52) GP per objective; export in the loader's schema."""
    entries = []
    xt = torch.tensor(x, dtype=torch.double)
    for m in range(y.shape[1]):
        yt = torch.tensor(y[:, m : m + 1], dtype=torch.double)
        gp = SingleTaskGP(
            xt,
            yt,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[1])),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        entries.append(
            {
                "kernel_type": "matern52",
                "lengthscale": gp.covar_module.base_kernel.lengthscale.detach()
                .reshape(-1)
                .tolist(),
                "variance": float(gp.covar_module.outputscale),
                "noise": float(gp.likelihood.noise),
                "mean_constant": float(gp.mean_module.constant),
            }
        )
    return {"objectives": entries}


def build_pool(root: Path, name: str, scenario: str, d: int) -> None:
    """Write one source pool (gp_states/ + trajectories/) for a problem and scenario."""
    rng = np.random.default_rng(1234)
    warps = [np.exp(rng.uniform(np.log(0.75), np.log(1.33), d)) for _ in range(3)]
    if scenario == "mixed":
        warps = warps[:2] + [np.full(d, 0.35)]
    for sub in ("gp_states", "trajectories"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for k, w in enumerate(warps):
        # BO-like trajectory: random exploration plus up to 20 near-front points.
        f = task_fn(name, w)
        x_explore = rng.random((10 * d + 10, d))
        x_dense = rng.random((4000, d))
        nd = np.flatnonzero(is_non_dominated(torch.tensor(f(x_dense))).numpy())
        nd = rng.choice(nd, size=min(20, nd.size), replace=False)
        x = np.vstack([x_explore, x_dense[nd]])
        y = f(x)
        (root / "trajectories" / f"src{k}.json").write_text(
            json.dumps(
                {
                    "x_values": x.tolist(),
                    "y_values": y.tolist(),
                    "pareto_front": pareto_front(y).tolist(),
                }
            )
        )
        (root / "gp_states" / f"src{k}.json").write_text(
            json.dumps({"gp_state": fit_gp_state(x, y)})
        )


def run_one(job: tuple) -> str:
    """Run one (problem, scenario, method, seed) job; skip it if its output exists."""
    name, scenario, method, seed, pool_dir, out_dir, n_init, n_iter, decay_rate = job
    out = Path(out_dir) / f"{name}_{scenario}_{method}_{seed}.json"
    if out.exists():
        return str(out)
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")

    prob, ref = make_problem(name)
    f = task_fn(name, np.ones(prob.dim))
    bounds = [(0.0, 1.0)] * prob.dim
    if method == "mobo":
        res = run_mobo_botorch(f, bounds, ref, n_init=n_init, n_iter=n_iter, seed=seed)
    else:
        mode, decay = method.split("+")
        res = run_mobo_taf(
            f, bounds, ref, pool_dir, n_init=n_init, n_iter=n_iter,
            source_reference_mode=mode,
            decay_rate=decay_rate if decay == "decay" else 0.0,
            seed=seed,
        )
    out.write_text(json.dumps([name, scenario, method, seed, res.hypervolume_history.tolist()]))
    return str(out)


def run_all(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in PROBLEMS:
        d = make_problem(name)[0].dim
        for scenario in SCENARIOS:
            pool = args.out_dir / "pools" / f"{name}_{scenario}"
            if not (pool / "gp_states").exists():
                build_pool(pool, name, scenario, d)
            for seed in range(args.n_seeds):
                for method in METHODS:
                    jobs.append((name, scenario, method, seed, str(pool), str(args.out_dir),
                                 args.n_init, args.n_iter, args.decay_rate))
        for seed in range(args.n_seeds):
            jobs.append((name, "-", "mobo", seed, "", str(args.out_dir),
                         args.n_init, args.n_iter, args.decay_rate))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = list(ex.map(run_one, jobs))
    print(f"completed {len(done)} runs")


def _se(v: np.ndarray) -> float:
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def summarize(args: argparse.Namespace) -> None:
    """Print log10 hypervolume regret per method, and paired differences vs plain MOBO."""
    runs: dict[tuple[str, str, str], dict[int, np.ndarray]] = defaultdict(dict)
    for path in args.out_dir.glob("*.json"):
        name, scenario, method, seed, hv = json.loads(path.read_text())
        runs[(name, scenario, method)][seed] = np.asarray(hv)

    n0 = args.n_init
    checkpoints = [n0 + 1, n0 + 5, n0 + 10, n0 + args.n_iter]  # in evaluations
    for name in PROBLEMS:
        max_hv = float(make_problem(name)[0].max_hv)
        base_runs = runs.get((name, "-", "mobo"), {})
        if not base_runs:
            print(f"\n== {name}: no plain-MOBO runs yet")
            continue

        def regret(rs: dict[int, np.ndarray], seeds: list[int]) -> np.ndarray:
            return np.log10(np.maximum(max_hv - np.array([rs[s] for s in seeds]), 1e-12))

        print(f"\n== {name}: log10 HV regret (mean +- se over seeds; lower is better)")
        print(f"{'scenario':9s}{'method':26s}"
              + "".join(f"{'eval ' + str(c):>15s}" for c in checkpoints)
              + "   paired vs MOBO: early      final")
        seeds = sorted(base_runs)
        base = regret(base_runs, seeds)
        print(f"{'-':9s}{'mobo':26s}"
              + "".join(f"{base[:, c - 1].mean():9.3f}+-{_se(base[:, c - 1]):.3f}" for c in checkpoints))
        for scenario in SCENARIOS:
            for method in METHODS:
                rs = runs.get((name, scenario, method), {})
                common = [s for s in seeds if s in rs]
                if not common:
                    continue
                a, b = regret(rs, common), regret(base_runs, common)
                # "early" averages the first ten suggestions; "final" is the last evaluation.
                early = a[:, n0 : n0 + 10].mean(1) - b[:, n0 : n0 + 10].mean(1)
                final = a[:, -1] - b[:, -1]
                print(f"{scenario:9s}{method:26s}"
                      + "".join(f"{a[:, c - 1].mean():9.3f}+-{_se(a[:, c - 1]):.3f}" for c in checkpoints)
                      + f"   {early.mean():+.3f}+-{_se(early):.3f}  {final.mean():+.3f}+-{_se(final):.3f}"
                      + f"  (n={len(common)})")


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        run_all(args)
    summarize(args)


if __name__ == "__main__":
    main()
