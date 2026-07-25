"""Multi-objective Transfer Acquisition Function (TAF-EHVI).

Multi-objective sibling of ``openbo.acquisition.taf``. The mechanism is deliberately the
same one, lifted from scalar improvement to hypervolume improvement:

    single objective (taf.py)                multi objective (this module)
    -------------------------------------    -------------------------------------------
    EI_target(x)                             qLogNEHVI_target(x)
    softplus(mu_i(x) - ref_i)                log HVI of mu_i(x) vs source i's Pareto front
    TAF-M: Epanechnikov over meta-features    same, with MO meta-features
    TAF-R: mis-ordered observation PAIRS      mis-classified Pareto RELATIONS between pairs
    [w_t*T + sum w_i*S_i] / [w_t + sum w_i]  same, evaluated in log space

Two properties of the scalar version are preserved exactly because they are what make the
mechanism work:

1. A source contributes through its posterior MEAN only; its variance is discarded. A source
   is a deterministic "prior utility surface", not a calibrated belief.
2. Sources whose weight is <= 0 leave BOTH the numerator and the denominator, so a rejected
   source cannot dilute the target term (matches ``taf_m_acquisition``).

The blend is computed as ``logsumexp`` of the log-weighted terms minus ``log`` of the weight
sum. Because ``logsumexp([log a, log b]) == log(a + b)`` exactly, this is the logarithm of the
scalar version's weighted average -- not an approximation of it. Log space is required rather
than merely convenient: the non-log hypervolume improvement of a dominated candidate is
identically zero with a zero gradient, which strands a gradient-based acquisition optimizer,
whereas the log form keeps a finite, monotonically decreasing "how far behind the front"
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from numpy.typing import NDArray

# Reused verbatim from the single-objective module so both stay in lockstep.
from openbo.acquisition.taf import compute_taf_m_weights, epanechnikov_weight

__all__ = [
    "MOSourceTaskSurrogate",
    "compute_taf_m_weights",
    "epanechnikov_weight",
    "pareto_relation",
    "compute_taf_r_pareto_weights",
    "build_source_hvi_term",
    "taf_mo_ehvi_acquisition",
    "mo_meta_features",
]

# qLogEHVI smoothing. tau_max controls accuracy (1e-3 keeps the recovered hypervolume
# improvement within ~5e-4 relative error, versus ~7e-3 at the 1e-2 default); tau_relu is
# left at its default because shrinking it inflates the gradient norm at points sitting
# exactly on the front boundary (it scales as 1/tau_relu).
_TAU_MAX = 1e-3
_FAT = True


@dataclass
class MOSourceTaskSurrogate:
    """Reconstructed multi-objective source-task surrogate used by TAF-EHVI.

    Counterpart of ``SourceTaskSurrogate``. ``mean_fn`` must be a differentiable torch
    callable mapping ``(..., d) -> (..., M)``; only the mean is ever consulted.
    """

    name: str
    mean_fn: Callable[[torch.Tensor], torch.Tensor]
    pareto_front: NDArray[np.float64]
    meta_features: NDArray[np.float64]
    reference_front: NDArray[np.float64] | None = None

    def front(self) -> NDArray[np.float64]:
        """Front the source measures improvement against."""
        return self.pareto_front if self.reference_front is None else self.reference_front

    def posterior_mean(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Source posterior mean at ``x`` as numpy, shape (n, M)."""
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.double)
        with torch.no_grad():
            mu = self.mean_fn(x_t)
        return np.asarray(mu.cpu().numpy(), dtype=np.float64)


def mo_meta_features(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multi-objective analogue of ``_default_meta_features``: [d, mean_m, std_m for each m]."""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.ndim != 2:
        raise ValueError("x_values must have shape (n, d).")
    if y_values.ndim != 2 or y_values.shape[0] != x_values.shape[0]:
        raise ValueError("y_values must have shape (n, M) matching x_values rows.")
    feats = [float(x_values.shape[1])]
    for m in range(y_values.shape[1]):
        feats.append(float(np.mean(y_values[:, m])))
        feats.append(float(np.std(y_values[:, m])))
    return np.asarray(feats, dtype=np.float64)


def pareto_relation(a: NDArray[np.float64], b: NDArray[np.float64], eps: float = 1e-12) -> int:
    """Pareto relation between two objective vectors (maximization).

    Returns +1 if ``a`` dominates ``b``, -1 if ``b`` dominates ``a``, 0 if incomparable.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_ge = np.all(a >= b - eps)
    b_ge = np.all(b >= a - eps)
    a_gt = np.any(a > b + eps)
    b_gt = np.any(b > a + eps)
    if a_ge and a_gt:
        return 1
    if b_ge and b_gt:
        return -1
    return 0


def compute_taf_r_pareto_weights(
    source_surrogates: list[MOSourceTaskSurrogate],
    x_obs: NDArray[np.float64],
    y_obs: NDArray[np.float64],
    rho: float,
    min_informative_pairs: int = 1,
) -> NDArray[np.float64]:
    """TAF-R weights from Pareto-dominance agreement on current target observations.

    Multi-objective counterpart of ``compute_taf_r_weights``. For each observation pair the
    target and the source each assert one of {a dominates b, b dominates a, incomparable};
    per-pair disagreement is 0.0 for an identical relation, 0.5 when one asserts a strict
    dominance and the other calls it incomparable, and 1.0 for opposite strict dominances.

    The denominator counts only INFORMATIVE pairs -- those where the target or the source
    asserts a strict dominance. This matters more as M grows: the share of mutually
    incomparable pairs rises like 2^(1-M), and those pairs are vacuous agreements. Averaging
    over all pairs would let an exactly inverted source keep almost the full kernel weight at
    M >= 4, whereas the informative-pair denominator drives it to exactly 1.0 (weight 0).

    Degenerate cases follow the single-objective module exactly: no sources -> empty, fewer
    than two observations -> normalized uniform (no evidence yet), a source that orders
    nothing -> weight 0.0 (it carries no ranking information, so it must NOT receive the
    weight a small distance would imply -- this is the multi-objective reading of the scalar
    module's "no comparable pairs" case), and all weights underflowing -> zeros so the
    acquisition falls back to the target term alone.
    """
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if x_obs.ndim != 2:
        raise ValueError("x_obs must have shape (n, d).")
    if y_obs.ndim != 2 or y_obs.shape[0] != x_obs.shape[0]:
        raise ValueError("y_obs must have shape (n, M) and match x_obs rows.")

    n_sources = len(source_surrogates)
    n = y_obs.shape[0]
    eps = 1e-12
    if n_sources == 0:
        return np.zeros(0, dtype=np.float64)
    # With <2 observations there is no dominance evidence yet; fall back to a
    # uniform distribution (normalized, matching compute_taf_m_weights' contract).
    if n < 2:
        return np.ones(n_sources, dtype=np.float64) / n_sources

    target_rel = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            target_rel[i, j] = pareto_relation(y_obs[i], y_obs[j])

    weights: list[float] = []
    for source in source_surrogates:
        mu_source = source.posterior_mean(x_obs)
        disagreement = 0.0
        informative = 0
        source_strict = 0
        for i in range(n):
            for j in range(i + 1, n):
                t_rel = int(target_rel[i, j])
                s_rel = pareto_relation(mu_source[i], mu_source[j])
                if s_rel != 0:
                    source_strict += 1
                if t_rel == 0 and s_rel == 0:
                    # Neither side orders this pair: vacuous agreement, carries no evidence.
                    continue
                informative += 1
                if t_rel == s_rel:
                    continue
                if t_rel != 0 and s_rel != 0:
                    disagreement += 1.0  # opposite strict dominances
                else:
                    disagreement += 0.5  # strict vs incomparable
        if source_strict == 0 or informative < max(1, int(min_informative_pairs)):
            # The source ordered nothing (a flat or degenerate surrogate), or there is too
            # little evidence to judge it. Either way it carries no ranking information, so
            # it gets zero weight -- the same call the single-objective module makes when
            # its comparable-pair count is zero. Without this a flat source would score only
            # the 0.5 "strict vs incomparable" penalty on each pair and keep most of its
            # weight forever, despite saying nothing.
            weights.append(0.0)
            continue
        distance = float(disagreement / informative)
        weights.append(epanechnikov_weight(distance, rho))

    weights_arr = np.asarray(weights, dtype=np.float64)
    total_weight = float(weights_arr.sum())
    if total_weight <= eps:
        return np.zeros(n_sources, dtype=np.float64)
    return weights_arr / total_weight


def build_source_hvi_term(
    source: MOSourceTaskSurrogate,
    ref_point: NDArray[np.float64],
) -> qLogExpectedHypervolumeImprovement:
    """Build the deterministic log-hypervolume-improvement term for one source.

    The source's mean function is wrapped in a ``GenericDeterministicModel``, whose posterior
    has exactly zero variance, so expected hypervolume improvement collapses to the plain
    hypervolume improvement of the predicted objective vector -- the multi-objective reading
    of ``softplus(mu_i(x) - ref_i)``. A ``StochasticSampler`` of size 1 is passed explicitly:
    the lazily-created default draws 128 identical copies of a deterministic mean, and a
    normal sampler is rejected outright by the ensemble posterior.
    """
    ref = np.asarray(ref_point, dtype=np.float64)
    front = np.asarray(source.front(), dtype=np.float64)
    if front.ndim != 2 or front.shape[1] != ref.shape[0]:
        raise ValueError(
            f"source '{source.name}' front must have shape (P, {ref.shape[0]}), got {front.shape}."
        )

    ref_t = torch.tensor(ref, dtype=torch.double)
    front_t = torch.tensor(front, dtype=torch.double)
    # A front point that does not dominate the reference contributes no hypervolume; drop it
    # so the partitioning is built from meaningful cells only.
    keep = torch.all(front_t > ref_t, dim=-1)
    front_t = front_t[keep]
    if front_t.shape[0] == 0:
        raise ValueError(
            f"source '{source.name}' has no front point dominating the reference point."
        )

    model = GenericDeterministicModel(source.mean_fn, num_outputs=int(ref.shape[0]))
    partitioning = NondominatedPartitioning(ref_point=ref_t, Y=front_t)
    return qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_t.tolist(),
        partitioning=partitioning,
        sampler=StochasticSampler(sample_shape=torch.Size([1])),
        tau_max=_TAU_MAX,
        fat=_FAT,
    )


def taf_mo_ehvi_acquisition(
    x: torch.Tensor,
    target_term: Callable[[torch.Tensor], torch.Tensor] | None,
    source_terms: list[Callable[[torch.Tensor], torch.Tensor]],
    source_weights: NDArray[np.float64],
    target_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the TAF-EHVI blend for a batch of candidates.

    ``target_term`` and each entry of ``source_terms`` must return LOG-scale acquisition
    values of shape ``(b,)`` for an input of shape ``(b, q, d)``. The result is

        log( [w_t * T(x) + sum_i w_i * S_i(x)] / [w_t + sum_i w_i] )

    i.e. the logarithm of the single-objective module's weighted average, computed in log
    space. Sources with non-positive weight are dropped from numerator and denominator alike.
    """
    source_weights = np.asarray(source_weights, dtype=np.float64)
    if source_weights.ndim != 1:
        raise ValueError("source_weights must have shape (n_sources,).")
    if len(source_terms) != source_weights.shape[0]:
        raise ValueError("source_terms and source_weights length mismatch.")

    log_terms: list[torch.Tensor] = []
    denominator = 0.0

    target_weight = float(target_weight)
    if target_weight > 0.0:
        if target_term is None:
            raise ValueError("target_term is required when target_weight > 0.")
        log_terms.append(float(np.log(target_weight)) + target_term(x))
        denominator += target_weight

    for term, w_i in zip(source_terms, source_weights):
        w = float(w_i)
        if w <= 0.0:
            continue
        log_terms.append(float(np.log(w)) + term(x))
        denominator += w

    if not log_terms:
        raise ValueError(
            "TAF-EHVI has no active terms: target_weight <= 0 and every source weight <= 0."
        )

    stacked = torch.stack(log_terms, dim=-1)
    return torch.logsumexp(stacked, dim=-1) - float(np.log(denominator))
