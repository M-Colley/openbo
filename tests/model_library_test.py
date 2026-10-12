"""Tests for the model-library switch (BoTorch or botorch-rust, ``BO_MODEL_LIBRARY``).

The botorch-rust cases run when ``botorch_rs`` is importable and complete (see
``openbo.model_library``) and are skipped otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import openbo
from openbo import model_library
from openbo.acquisition import taf_mo_ehvi
from openbo.acquisition.taf_mo_ehvi import build_source_hvi_term
from openbo.optimizers import mobo_botorch, mobo_taf
from openbo.optimizers.mobo_taf import (
    MOTAFConfig,
    MOTAFSequentialOptimizer,
    _load_mo_source_surrogates,
)

RS_AVAILABLE = model_library._botorch_rs_missing() is None
needs_rs = pytest.mark.skipif(not RS_AVAILABLE, reason="botorch_rs is not installed (complete)")

D = 3
REF = [-1.0, -1.0]
BOUNDS = [(0.0, 1.0)] * D
SRC = str(Path(openbo.__file__).resolve().parents[1])


def objective(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.stack(
        [1.0 - np.sum((x - 0.3) ** 2, axis=1), 1.0 - np.sum((x - 0.7) ** 2, axis=1)], axis=1
    )


def write_source(root: Path, name: str, shift: float, seed: int, n: int = 12) -> None:
    rng = np.random.default_rng(seed)
    for sub in ("gp_states", "trajectories"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    x = rng.random((n, D))
    y = objective(x + shift)
    (root / "trajectories" / f"{name}.json").write_text(
        json.dumps({"x_values": x.tolist(), "y_values": y.tolist()}), encoding="utf-8"
    )
    entries = [
        {"kernel_type": "matern52", "lengthscale": [0.3, 0.4, 0.5], "variance": 1.2,
         "noise": 0.01, "mean_constant": 0.1},
        {"kernel_type": "rbf", "lengthscale": [0.5, 0.3, 0.6], "variance": 0.8,
         "noise": 0.02, "mean_constant": -0.1},
    ]
    (root / "gp_states" / f"{name}.json").write_text(
        json.dumps({"gp_state": {"objectives": entries}}), encoding="utf-8"
    )


# ---------------------------------------------------------------- resolution


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "botorch"),
        ("", "botorch"),
        (" BoTorch ", "botorch"),
        ("botorch_rs", "botorch_rs"),
        ("botorch-rust", "botorch_rs"),
        ("botorch-rs", "botorch_rs"),
        ("auto", "auto"),
    ],
)
def test_normalize_request(value, expected):
    assert model_library.normalize_request(value) == expected


def test_invalid_request_is_rejected():
    with pytest.raises(ValueError, match="BO_MODEL_LIBRARY"):
        model_library.normalize_request("torch")


def test_every_module_states_the_bound_library():
    bound = model_library.MODEL_LIBRARY
    assert bound in model_library.LIBRARIES
    for module in (openbo, mobo_botorch, mobo_taf, taf_mo_ehvi):
        assert module.MODEL_LIBRARY == bound
    root = "botorch_rs" if bound == "botorch_rs" else "botorch"
    assert mobo_botorch.SingleTaskGP.__module__.split(".")[0] == root
    assert mobo_taf.optimize_acqf.__module__.split(".")[0] == root
    assert taf_mo_ehvi.PosteriorMeanModel.__module__.split(".")[0] == root


def test_botorch_rs_missing_falls_back_to_botorch(monkeypatch):
    monkeypatch.setitem(sys.modules, "botorch_rs", None)  # import fails
    for request in ("botorch_rs", "auto", "botorch-rust"):
        library, reason = model_library.resolve(request)
        assert library == "botorch"
        assert "not installed" in reason and "falling back to BoTorch" in reason
    assert model_library.resolve("botorch") == ("botorch", "BO_MODEL_LIBRARY=botorch")


@needs_rs
def test_incomplete_botorch_rs_falls_back_to_botorch(monkeypatch):
    import botorch_rs.acquisition.composite as composite

    monkeypatch.delattr(composite, "WeightedLogMeanExp")
    library, reason = model_library.resolve("botorch_rs")
    assert library == "botorch"
    assert "WeightedLogMeanExp" in reason


def _import_in_subprocess(value: str | None) -> tuple[list[str], str]:
    env = dict(os.environ)
    env.pop(model_library.ENV_VAR, None)
    if value is not None:
        env[model_library.ENV_VAR] = value
    env["PYTHONPATH"] = os.pathsep.join([SRC, env.get("PYTHONPATH", "")])
    code = (
        "import openbo.optimizers.mobo_taf as t, openbo.optimizers.mobo_botorch as b, "
        "openbo.acquisition.taf_mo_ehvi as e, openbo as o; "
        "print(o.MODEL_LIBRARY, t.MODEL_LIBRARY, b.MODEL_LIBRARY, e.MODEL_LIBRARY, "
        "b.SingleTaskGP.__module__.split('.')[0])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.split(), out.stderr


@pytest.mark.parametrize("value", [None, "BoTorch"])
def test_import_binds_botorch_by_default(value):
    libs, stderr = _import_in_subprocess(value)
    assert libs == ["botorch"] * 4 + ["botorch"]
    assert "openbo:" not in stderr


def test_invalid_environment_value_keeps_botorch_and_says_so():
    libs, stderr = _import_in_subprocess("pytorch")
    assert libs == ["botorch"] * 4 + ["botorch"]
    assert "BO_MODEL_LIBRARY must be one of" in stderr


@needs_rs
@pytest.mark.parametrize("value", ["botorch-rust", "auto"])
def test_import_binds_botorch_rs(value):
    libs, _ = _import_in_subprocess(value)
    assert libs == ["botorch_rs"] * 4 + ["botorch_rs"]


# ---------------------------------------------------------------- source terms


def test_posterior_mean_term_equals_generic_deterministic_term(tmp_path):
    """PosteriorMeanModel(model_list) gives exactly the GenericDeterministicModel(mean_fn)
    term it replaces."""
    write_source(tmp_path, "a", shift=0.05, seed=1)
    (src,) = _load_mo_source_surrogates(tmp_path, library="botorch")
    with_model = build_source_hvi_term(src, np.array(REF), library="botorch")
    src.model = None
    with_mean_fn = build_source_hvi_term(src, np.array(REF), library="botorch")
    x = torch.rand(16, 1, D, dtype=torch.double, generator=torch.Generator().manual_seed(0))
    xg1, xg2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    v1, v2 = with_model(xg1), with_mean_fn(xg2)
    assert torch.equal(v1, v2)
    (g1,) = torch.autograd.grad(v1.sum(), xg1)
    (g2,) = torch.autograd.grad(v2.sum(), xg2)
    assert torch.equal(g1, g2)


@needs_rs
def test_source_terms_agree_across_libraries(tmp_path):
    write_source(tmp_path, "a", shift=0.05, seed=1)
    (bt,) = _load_mo_source_surrogates(tmp_path, library="botorch")
    (rs,) = _load_mo_source_surrogates(tmp_path, library="botorch_rs")
    assert type(rs.model).__module__.startswith("botorch_rs")
    x = torch.rand(16, 1, D, dtype=torch.double, generator=torch.Generator().manual_seed(1))
    np.testing.assert_allclose(rs.posterior_mean(x[:, 0]), bt.posterior_mean(x[:, 0]), rtol=1e-12)
    t_bt = build_source_hvi_term(bt, np.array(REF), library="botorch")
    t_rs = build_source_hvi_term(rs, np.array(REF), library="botorch_rs", dim=D)
    xg1, xg2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    v_bt, v_rs = t_bt(xg1), t_rs(xg2)
    np.testing.assert_allclose(v_rs.detach(), v_bt.detach(), rtol=1e-9, atol=1e-12)
    (g_bt,) = torch.autograd.grad(v_bt.sum(), xg1)
    (g_rs,) = torch.autograd.grad(v_rs.sum(), xg2)
    np.testing.assert_allclose(g_rs, g_bt, rtol=1e-8, atol=1e-10 * float(g_bt.abs().max()))


# ---------------------------------------------------------------- MO-TAF on both libraries


def _optimizer(tmp_path: Path, library: str, seed: int = 0, **kw) -> MOTAFSequentialOptimizer:
    return MOTAFSequentialOptimizer(
        MOTAFConfig(
            bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, n_init=5, num_restarts=4,
            raw_samples=32, seed=seed, model_library=library, **kw,
        )
    )


@needs_rs
def test_mo_taf_runs_on_botorch_rs_with_the_native_composite(tmp_path):
    write_source(tmp_path, "a", shift=0.05, seed=1)
    write_source(tmp_path, "b", shift=-0.1, seed=2)
    opt = _optimizer(tmp_path, "botorch_rs")
    assert opt.model_library == "botorch_rs"
    opt.bootstrap(objective)
    opt.n_suggestions = 1  # as during the first suggest()
    train_x, train_y, _ = opt._unit_train_tensors()
    acq = opt._build_acquisition(opt._fit_model(train_x, train_y), train_x)
    opt.n_suggestions = 0
    assert type(acq).__name__ == "WeightedLogMeanExp"
    # The native blend is openbo's formula on the same terms.
    x = torch.rand(6, 1, D, dtype=torch.double, generator=torch.Generator().manual_seed(2))
    # The active terms (positive weights; the target first, with weight 1).
    assert acq.weights[0] == opt.last_target_weight == 1.0
    assert sorted(acq.weights[1:]) == sorted(w for w in opt.last_source_weights if w > 0)
    expected = taf_mo_ehvi.taf_mo_ehvi_acquisition(
        x, acq.acq_functions[0], acq.acq_functions[1:], np.asarray(acq.weights[1:]),
        target_weight=acq.weights[0],
    )
    np.testing.assert_allclose(acq(x), expected.detach(), rtol=1e-12, atol=1e-12)
    for _ in range(3):
        x_next = opt.suggest()
        opt.observe(x_next, objective(x_next))
    result = opt.result()
    assert result.final_state["model_library"] == "botorch_rs"
    assert result.x_obs.shape == (8, D)


@needs_rs
def test_mo_taf_on_botorch_rs_is_reproducible(tmp_path):
    write_source(tmp_path, "a", shift=0.05, seed=1)
    runs = []
    for _ in range(2):
        opt = _optimizer(tmp_path, "botorch_rs", seed=3)
        opt.bootstrap(objective)
        xs = []
        for _ in range(2):
            x_next = opt.suggest()
            opt.observe(x_next, objective(x_next))
            xs.append(x_next)
        runs.append(np.vstack(xs))
    np.testing.assert_array_equal(runs[0], runs[1])


def _suggestion_values(tmp_path: Path, seed: int) -> tuple[float, float]:
    """BoTorch's MO-TAF acquisition value at its own suggestion and at botorch-rust's, for
    the same observations (log scale)."""
    bt = _optimizer(tmp_path, "botorch", seed=seed)
    rs = _optimizer(tmp_path, "botorch_rs", seed=seed)
    bt.bootstrap(objective)
    rs.observe(bt.x_obs, bt.y_obs)
    x_bt = bt.suggest()
    x_rs = rs.suggest()
    bt.n_suggestions = 1  # the same weights as the suggestion just made
    train_x, train_y, _ = bt._unit_train_tensors()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        acq = bt._build_acquisition(bt._fit_model(train_x, train_y), train_x)
        X = torch.tensor(np.stack([x_bt, x_rs]), dtype=torch.double)  # 2 x 1 x d (unit cube)
        with torch.no_grad():
            v = acq(X)
    return float(v[0]), float(v[1])


@needs_rs
def test_mo_taf_suggestions_reach_botorchs_optimum(tmp_path):
    """Over several seeds, BoTorch's own acquisition function values botorch-rust's
    suggestions as highly as its own (the libraries' Monte-Carlo samples differ, so single
    suggestions differ by MC noise; on average they must agree)."""
    write_source(tmp_path, "a", shift=0.05, seed=1)
    write_source(tmp_path, "b", shift=-0.1, seed=2)
    gaps = []
    for seed in range(4):
        v_bt, v_rs = _suggestion_values(tmp_path, seed)
        gaps.append(v_bt - v_rs)  # > 0: botorch-rust's suggestion is worse for BoTorch
    gaps = np.asarray(gaps)
    # Each suggestion within 0.25 (log scale) of BoTorch's optimum, and no systematic gap.
    assert gaps.max() < 0.25, gaps
    assert gaps.mean() < 0.1, gaps
