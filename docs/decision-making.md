# Knee-region decisions and FPBI

`openbo.decision_making` implements the two-stage decision rule and focus
point boundary intersection (FPBI) sampler from [Schmitt et al. (2022),
*Incorporating Human Preferences in Decision Making for Dynamic Multi-Objective
Optimization in Model Predictive Control*](https://doi.org/10.3390/inventions7030046).
This is a front-selection and front-generation module. It does not implement
an MPC controller or add stability guarantees to Bayesian optimization.

## Select from measured or predicted outcomes

```python
import torch
from openbo.decision_making import knee_region, select_preferred

# Paper costs: lower values are better. The library defaults to maximization.
Y = torch.tensor([[0., 1.], [.2, .3], [.6, .15], [1., 0.]], dtype=torch.double)
mask = knee_region(Y, alpha=0.5, maximize=False)
index = select_preferred(Y, [70., 30.], alpha=0.5, maximize=False)
chosen_outcomes = Y[index]
```

Both functions accept `Y` of shape `(..., n, m)` and preserve its device.
`knee_region` returns a boolean mask of shape `(..., n)`. `select_preferred`
returns `int64` original row indices of shape `Y.shape[:-2]`. Preferences have
shape `(..., m)`, broadcast to the objective tensor's batch shape, and must be
finite, nonnegative and nonzero. They need not sum to one. Ties select the
first original row; duplicate outcome vectors keep their first occurrence.

The implementation first excludes dominated rows, then converts maximized
outcomes to costs if needed and applies the paper's range normalization
(equation 8). Flat objective columns map to zero. Let `Z` be the normalized
front and `q` its first row of greatest Euclidean norm. The **signed** distance
is `(sum(Z[q]) - sum(Z[i])) / sqrt(m)` (equation 10). A row belongs to the
knee region when its distance is at least `alpha * max_distance` (equation
11). `alpha` must lie in `[0, 1]`. In particular, `alpha=0` can still exclude
negative-distance rows; replacing this calculation with an absolute distance
would implement a different method. Within the region, selection minimizes
the preference-weighted normalized costs (equation 13).

The method uses the supplied front approximation to estimate its ranges.
Missing extremes, changing ranges and changing `alpha` can change a decision.
For FPBI, a zero range across the observed individual extrema does not prove
that the objective is constant elsewhere; interior tradeoffs may be missed.
Preferences describe the hyperplane in normalized outcome space; percentages
are not promises about achieved objective shares. Knee filtering can exclude
the preferred extreme even for a 100–0 preference.

## Generate a front by continuous optimization

```python
from openbo.decision_making import sample_fpbi

def costs(X):
    x = X[..., 0]
    return torch.stack((x.square(), (1 - x).square()), dim=-1)

bounds = torch.tensor([[0.], [1.]], dtype=torch.double)  # BoTorch (2, d)
result = sample_fpbi(costs, bounds, n_steps=5, maximize=False, seed=0)
chosen = select_preferred(result.Y, [50., 50.], maximize=False)
design = result.X[chosen]
```

`objective(X)` accepts a tensor of shape `(n, d)` and returns finite floating
outcomes of shape `(n, m)`, with at least two objectives. The sampler solves
one optimization problem at a time. Bounds use `float32` or `float64` and
must have finite positive widths. Torch evaluation and result tensors use the
bounds' dtype and device; SciPy SLSQP runs on the CPU. No gradients propagate
through the optimizer. Float64 is recommended for numerical precision;
float32 uses a larger finite-difference step and stopping tolerance.
If the difference between finite extreme objective endpoints overflows,
FPBI rejects the geometry and asks for objective rescaling.

First, SLSQP optimizes each individual objective using multiple deterministic
Sobol starts. The resulting individual extreme outcomes define the FPBI
geometry (equations 16–26). The farthest pair determines the midpoint and
first plane direction. The search direction points from that midpoint toward
`focus`, which defaults to Utopia (zero). **`focus` is supplied in normalized
cost coordinates; it is independent of `select_preferred`'s preferences.**

For `r = n_steps`, the paper's grid coefficients are
`gamma[k] = (k - r/2) * farthest_pair_distance/r`, for `k=1,...,r`.
This intentionally asymmetric grid includes the positive endpoint and omits
the negative endpoint. There are `r**(m-1)` nominal ray subproblems. The
`max_rays=10000` default rejects larger grids before optimization starts.

For each anchor, the sampler minimizes `-kappa` subject to
`anchor + kappa * direction >= normalized_costs(X)`, plus the original bounds
and nonlinear constraints (equation 27). `kappa` is unrestricted, including
negative values. Inequalities are used rather than ray equalities, as in the
paper. `inequality_constraints(X)` follows the SciPy/BoTorch convention
**greater than or equal to zero**; `equality_constraints(X)` must equal zero.
Both may return a scalar or a tensor of constraint values for the supplied
single-row input. For example:

```python
result = sample_fpbi(
    costs, bounds, n_steps=5, maximize=False,
    inequality_constraints=lambda X: torch.stack((X[:, 0] - .2, .8 - X[:, 0]), -1),
)
```

Returned `FPBIResult` fields are:

| Field | Meaning |
| --- | --- |
| `X`, `Y` | Successful feasible designs/outcomes, deduplicated and nondominated among returned samples |
| `extreme_X`, `extreme_Y` | Individual-objective solutions used for geometry |
| `geometry` | `FPBIGeometry`, including normalization, plane, direction and anchors |
| `diagnostics` | Every solver attempt, status, message, acceptance, residual and ray `kappa` |
| `ray_succeeded` | One boolean for every nominal ray |

Solver success and explicit feasibility are both required. Failure to solve
an individual extreme raises `RuntimeError`. Failed rays are omitted from
the returned front and remain visible in diagnostics; inspect
`result.ray_succeeded`. Local SLSQP and finite restarts do not establish a
global Pareto front. Dominance filtering only checks the returned samples.

`fpbi_geometry(extreme_Y, ...)` is also available independently. It accepts
batched extreme tensors `(..., m, m)`, with row `i` representing the outcome
from optimizing objective `i`. It returns a batched `FPBIGeometry`. Its
`basis` has shape `(..., m, m-1)` with directions as columns, and `anchors`
has shape `(..., r**(m-1), m)`. With flat objectives the implementation works
in the active dimensions and pads unused basis columns with zeros, leading
to repeated anchors that the sampler solves only once. A completely flat
front returns individual extreme samples and marks the geometry degenerate;
there are no meaningful ray successes. Rank-deficient nonflat extrema use a
deterministic orthogonal basis completion, reported by
`used_fallback_basis`. These are explicit extensions for cases where the
paper's cross-product formula has a zero denominator.

## Use a BoTorch-compatible model

```python
from openbo.decision_making import sample_model_fpbi

# Any object with model.posterior(X).mean, including compatible botorch_rs models.
result = sample_model_fpbi(model, bounds, n_steps=5, maximize=True)
```

This optimizes posterior **means**, without accounting for uncertainty in the
front or selecting a BO acquisition. For a model with batch dimensions, pass
`model_batch_index=(...)` to choose one problem explicitly. It does not flatten
model batches into points or silently combine independent problems.

Run the example with `python examples/decision_making.py` and the focused
checks with `python -m pytest tests/test_decision_making.py -q`.

## Measured evidence and current limits

See [what worked and what did not](decision-making-evidence.md) for functional
validation, negative final-utility and FPBI comparisons, runtime regressions,
applicable uses and the upstream draft. The methods remain optional research
utilities; the measurements do not establish a general robustness or speed
improvement.
