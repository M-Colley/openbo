"""Source-equation fixtures and actual constrained FPBI solves."""

from types import SimpleNamespace

import pytest
import torch

from openbo.decision_making import (
    fpbi_geometry,
    knee_region,
    sample_fpbi,
    sample_model_fpbi,
    select_preferred,
)


def front(dtype=torch.double):
    return torch.tensor([[0.0, 1.0], [0.2, 0.3], [0.6, 0.15], [1.0, 0.0]], dtype=dtype)


def quadratic(X):
    x = X[..., 0]
    return torch.stack((x.square(), (1 - x).square()), -1)


def test_signed_distance_is_not_absolute_and_alpha_zero_is_not_all_front():
    # All five points are nondominated, normalized ranges are exactly [0,1].
    Y = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.6, 0.0],
            [0.1, 0.1, 0.1],
        ],
        dtype=torch.double,
    )
    assert knee_region(Y, alpha=0, maximize=False).tolist() == [
        True,
        True,
        True,
        False,
        True,
    ]
    assert knee_region(Y, alpha=1, maximize=False).tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_filter_before_normalization_duplicates_original_indices_and_ties():
    Y = torch.tensor([[0.0, 1.0], [0.3, 0.3], [0.3, 0.3], [1.0, 0.0], [50.0, 50.0]])
    assert knee_region(Y, maximize=False).tolist() == [False, True, False, False, False]
    assert select_preferred(Y, [1, 1], maximize=False).item() == 1
    line = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert select_preferred(line, [1, 1], maximize=False).item() == 0


def test_batch_preferences_orientation_and_dtype():
    Y = torch.stack([front(), front()])
    p = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
    result = select_preferred(Y, p, alpha=0, maximize=False)
    assert result.tolist() == [0, 3]
    assert result.dtype == torch.long and result.device == Y.device
    assert torch.equal(result, select_preferred(-Y, p, alpha=0))
    assert knee_region(Y).shape == (2, 4)


def test_flat_columns_identical_rows_single_objective_and_extreme_ranges():
    flat = torch.tensor([[3.0, 7.0], [3.0, 7.0]], dtype=torch.double)
    assert knee_region(flat).tolist() == [True, False]
    assert select_preferred(flat, [0.0, 2.0]).item() == 0
    assert knee_region(torch.tensor([[1.0], [3.0], [2.0]])).tolist() == [
        False,
        True,
        False,
    ]
    huge = torch.tensor(
        [[-1e308, 1e308], [0.0, 0.0], [1e308, -1e308]], dtype=torch.double
    )
    assert knee_region(huge, maximize=False).tolist() == [True, True, True]
    assert select_preferred(huge, [3.0, 1.0], maximize=False).item() == 0


def test_large_front_uses_bounded_pairwise_storage():
    x = torch.linspace(0, 1, 1350)
    Y = torch.stack((x, 1 - x, torch.zeros_like(x)), -1)
    assert knee_region(Y, alpha=0, maximize=False).all()


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_bad_alpha(alpha):
    with pytest.raises((ValueError, TypeError)):
        knee_region(front(), alpha=alpha)


@pytest.mark.parametrize(
    "Y",
    [
        torch.zeros(0, 3),
        torch.zeros(0, 4, 3),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[1, 2]]),
    ],
)
def test_bad_objective_tensors(Y):
    with pytest.raises((ValueError, TypeError)):
        knee_region(Y)


@pytest.mark.parametrize(
    "p", [[0.0, 0.0], [-1.0, 1.0], [float("inf"), 1.0], [1.0], [[1.0, 1.0]]]
)
def test_bad_preferences(p):
    with pytest.raises(ValueError):
        select_preferred(front(), p)


def test_paper_grid_is_asymmetric_and_direction_not_normalized():
    E = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    G = fpbi_geometry(E, n_steps=4, maximize=False)
    torch.testing.assert_close(G.center, torch.tensor([0.5, 0.5], dtype=torch.double))
    torch.testing.assert_close(
        G.direction, torch.tensor([-0.5, -0.5], dtype=torch.double)
    )
    expected = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]], dtype=torch.double
    )
    torch.testing.assert_close(G.anchors, expected)
    assert not G.used_fallback_basis
    torch.testing.assert_close(fpbi_geometry(-E, n_steps=4).anchors, G.anchors)


def test_batched_geometry_flat_axes_and_degenerate_extrema():
    E = torch.tensor(
        [[0.0, 1.0, 4.0], [1.0, 0.0, 4.0], [0.5, 0.5, 4.0]], dtype=torch.double
    )
    G = fpbi_geometry(torch.stack([E, E]), n_steps=2, maximize=False)
    assert G.anchors.shape == (2, 4, 3)
    assert G.basis.shape == (2, 3, 2)
    assert G.degenerate.shape == (2,)
    assert not G.active_objectives[:, 2].any()
    assert not G.anchors[..., 2].any()
    assert not G.basis[..., 1].any()
    identical = fpbi_geometry(torch.ones(3, 3), n_steps=2)
    assert identical.degenerate
    assert not identical.anchors.any()


def test_generalized_cross_basis_and_rank_deficient_fallback():
    E = torch.ones(4, 4, dtype=torch.double) - torch.eye(4, dtype=torch.double)
    G = fpbi_geometry(E, n_steps=2, maximize=False)
    torch.testing.assert_close(G.basis.T @ G.basis, torch.eye(3, dtype=torch.double))
    torch.testing.assert_close(
        G.basis.T @ G.direction, torch.zeros(3, dtype=torch.double)
    )
    assert G.anchors.shape == (8, 4)
    # Two unique extrema in four nonflat objectives need a documented extension.
    rank_deficient = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.double,
    )
    G = fpbi_geometry(rank_deficient, n_steps=2, maximize=False)
    assert G.used_fallback_basis
    torch.testing.assert_close(G.basis.T @ G.basis, torch.eye(3, dtype=torch.double))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_fpbi_solves_quadratic_front_and_preserves_dtype(dtype):
    bounds = torch.tensor([[0.0], [1.0]], dtype=dtype)
    R = sample_fpbi(
        quadratic,
        bounds,
        n_steps=3,
        num_restarts=2,
        maximize=False,
        feasibility_tol=2e-5 if dtype == torch.float32 else 1e-6,
    )
    assert R.X.dtype == R.Y.dtype == dtype
    assert R.X.device == bounds.device
    assert len(R.X) >= 4
    torch.testing.assert_close(R.Y, quadratic(R.X))
    assert R.ray_succeeded.all()
    # Interior samples prove the implementation actually solves new designs.
    assert ((R.X[:, 0] > 0.1) & (R.X[:, 0] < 0.9)).any()
    for diagnostic in R.diagnostics:
        if diagnostic.accepted:
            assert diagnostic.solver_success and diagnostic.max_violation <= 2e-5
    # At normalized anchor(.5,.5), true optimum is x=.5 and kappa=.5.
    centered = sample_fpbi(
        quadratic,
        bounds,
        n_steps=2,
        num_restarts=1,
        maximize=False,
        feasibility_tol=2e-5 if dtype == torch.float32 else 1e-6,
    )
    accepted = [
        d
        for d in centered.diagnostics
        if d.kind == "ray" and d.index == 0 and d.accepted
    ]
    assert max(d.kappa for d in accepted) == pytest.approx(0.5, abs=1e-4)


def test_nonlinear_constraints_apply_to_extremes_and_rays():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    R = sample_fpbi(
        quadratic,
        bounds,
        n_steps=2,
        num_restarts=2,
        maximize=False,
        inequality_constraints=lambda X: X[..., :1] - 0.2,
    )
    assert (R.X >= 0.2 - 1e-6).all()
    assert (R.extreme_X >= 0.2 - 1e-6).all()
    assert R.ray_succeeded.all()


def test_concave_front_requires_negative_unrestricted_kappa():
    def concave(X):
        x = X[..., 0]
        return torch.stack((x.sqrt(), (1 - x).sqrt()), -1)

    R = sample_fpbi(
        concave,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        n_steps=2,
        num_restarts=1,
        maximize=False,
    )
    accepted = [
        d for d in R.diagnostics if d.kind == "ray" and d.index == 0 and d.accepted
    ]
    assert max(d.kappa for d in accepted) == pytest.approx(1 - 2**0.5, abs=1e-5)


def test_equality_constraints_and_posterior_mean_adapter():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    model = SimpleNamespace(posterior=lambda X: SimpleNamespace(mean=quadratic(X)))
    R = sample_model_fpbi(
        model,
        bounds,
        n_steps=2,
        num_restarts=1,
        maximize=False,
        equality_constraints=lambda X: X[:, 1:] - 0.4,
    )
    torch.testing.assert_close(R.X[:, 1], torch.full_like(R.X[:, 1], 0.4))
    torch.testing.assert_close(R.Y, quadratic(R.X))


def test_failure_is_checked_and_nonfinite_objective_rejected(monkeypatch):
    import openbo.decision_making._fpbi as module

    original = module.minimize

    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        result.status = 9
        result.message = "injected failure"
        return result

    monkeypatch.setattr(module, "minimize", failed)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(RuntimeError, match="extreme"):
        sample_fpbi(quadratic, bounds, n_steps=2, num_restarts=1, maximize=False)
    with pytest.raises(ValueError, match="nonfinite"):
        sample_fpbi(lambda X: torch.full((len(X), 2), float("nan")), bounds)


def test_failed_rays_are_recorded_and_not_promoted(monkeypatch):
    import openbo.decision_making._fpbi as module

    original = module.minimize

    def fail_ray(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(kwargs["bounds"]) == 2:
            result.success = False
            result.status = 9
            result.message = "injected ray failure"
        return result

    monkeypatch.setattr(module, "minimize", fail_ray)
    R = sample_fpbi(
        quadratic,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        n_steps=2,
        num_restarts=1,
        maximize=False,
    )
    assert not R.ray_succeeded.any()
    assert len(R.X) == 2  # only successful individual extreme solves remain
    assert all(not d.accepted for d in R.diagnostics if d.kind == "ray")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_steps": 0},
        {"n_steps": True},
        {"max_rays": 1},
        {"focus": [float("nan"), 0.0]},
        {"focus": [2.0, 2.0]},
    ],
)
def test_invalid_fpbi_options(kwargs):
    with pytest.raises(ValueError):
        sample_fpbi(
            quadratic,
            torch.tensor([[0.0], [1.0]], dtype=torch.double),
            num_restarts=1,
            maximize=False,
            **kwargs,
        )


def test_low_precision_and_overflowing_boxes_rejected():
    with pytest.raises(ValueError, match="rescale"):
        fpbi_geometry(torch.tensor([[-1e308, 1e308], [1e308, -1e308]], dtype=torch.double))
    with pytest.raises(TypeError, match="float32 or float64"):
        sample_fpbi(quadratic, torch.tensor([[0.0], [1.0]], dtype=torch.float16))
    with pytest.raises(ValueError, match="widths"):
        sample_fpbi(quadratic, torch.tensor([[-1e308], [1e308]], dtype=torch.double))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensor_boundary():
    Y = front().cuda()
    assert knee_region(Y).is_cuda
    assert select_preferred(Y, [1.0, 1.0]).is_cuda
    R = sample_fpbi(
        quadratic,
        torch.tensor([[0.0], [1.0]], dtype=torch.double, device="cuda"),
        n_steps=2,
        num_restarts=1,
        maximize=False,
    )
    assert R.X.is_cuda and R.Y.is_cuda and R.geometry.anchors.is_cuda
