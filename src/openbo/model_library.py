"""The model library behind openbo's multi-objective BoTorch optimizers: BoTorch or botorch-rust.

botorch-rust (https://github.com/M-Colley/botorch-rust, Python package ``botorch_rs``) is a
Rust reimplementation of BoTorch with the same API paths. The environment variable
``BO_MODEL_LIBRARY`` selects the library when openbo is imported:

* ``botorch`` (default, also when unset or empty): BoTorch and GPyTorch;
* ``botorch_rs`` (also ``botorch-rust``): botorch-rust;
* ``auto``: botorch-rust when it is installed and complete, otherwise BoTorch.

botorch-rust is bound only when it provides everything the multi-objective optimizers and
MO-TAF need (``PosteriorMeanModel``, ``StochasticSampler``, the native composite
``WeightedLogMeanExp``, ...). Otherwise openbo falls back to BoTorch and prints why.
``MODEL_LIBRARY`` states the library that was actually bound (``"botorch"`` or
``"botorch_rs"``) and ``MODEL_LIBRARY_REASON`` how it was chosen; the modules
``openbo.optimizers.mobo_botorch``, ``openbo.optimizers.mobo_taf`` and
``openbo.acquisition.taf_mo_ehvi`` re-export ``MODEL_LIBRARY`` and bind their BoTorch names
(``SingleTaskGP``, ``optimize_acqf``, ...) from it. ``MOBoTorchConfig.model_library``
overrides the choice for one optimizer.

The library is part of an experiment's design: results of the two libraries agree in
distribution, not bit for bit (botorch-rust draws its own random numbers), so a study should
not switch libraries midway.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

ENV_VAR = "BO_MODEL_LIBRARY"
BOTORCH = "botorch"
BOTORCH_RS = "botorch_rs"
AUTO = "auto"
LIBRARIES = (BOTORCH, BOTORCH_RS)
REQUESTS = (BOTORCH, BOTORCH_RS, AUTO)
_ALIASES = {"botorch-rust": BOTORCH_RS, "botorch_rust": BOTORCH_RS, "botorch-rs": BOTORCH_RS}

# (name, BoTorch module, botorch_rs module or None, attribute). botorch_rs mirrors BoTorch's
# module paths, with GPyTorch's modules under ``botorch_rs`` itself.
_NAMES = [
    ("SingleTaskGP", "botorch.models", "botorch_rs.models", "SingleTaskGP"),
    ("ModelListGP", "botorch.models", "botorch_rs.models", "ModelListGP"),
    ("Standardize", "botorch.models.transforms.outcome", "botorch_rs.models.transforms.outcome",
     "Standardize"),
    ("MaternKernel", "gpytorch.kernels", "botorch_rs.kernels", "MaternKernel"),
    ("RBFKernel", "gpytorch.kernels", "botorch_rs.kernels", "RBFKernel"),
    ("ScaleKernel", "gpytorch.kernels", "botorch_rs.kernels", "ScaleKernel"),
    ("ExactMarginalLogLikelihood", "gpytorch.mlls", "botorch_rs.mlls",
     "ExactMarginalLogLikelihood"),
    ("fit_gpytorch_mll", "botorch.fit", "botorch_rs.fit", "fit_gpytorch_mll"),
    ("optimize_acqf", "botorch.optim", "botorch_rs.optim", "optimize_acqf"),
    ("SobolQMCNormalSampler", "botorch.sampling.normal", "botorch_rs.sampling.normal",
     "SobolQMCNormalSampler"),
    ("StochasticSampler", "botorch.sampling.stochastic_samplers",
     "botorch_rs.sampling.stochastic_samplers", "StochasticSampler"),
    ("qLogNoisyExpectedHypervolumeImprovement", "botorch.acquisition.multi_objective.logei",
     "botorch_rs.acquisition.multi_objective.logei", "qLogNoisyExpectedHypervolumeImprovement"),
    ("qLogExpectedHypervolumeImprovement", "botorch.acquisition.multi_objective.logei",
     "botorch_rs.acquisition.multi_objective.logei", "qLogExpectedHypervolumeImprovement"),
    ("PosteriorMeanModel", "botorch.models.deterministic", "botorch_rs.models.deterministic",
     "PosteriorMeanModel"),
    ("GenericDeterministicModel", "botorch.models.deterministic",
     "botorch_rs.models.deterministic", "GenericDeterministicModel"),
    ("NondominatedPartitioning",
     "botorch.utils.multi_objective.box_decompositions.non_dominated",
     "botorch_rs.utils.multi_objective.box_decompositions.non_dominated",
     "NondominatedPartitioning"),
    ("Hypervolume", "botorch.utils.multi_objective.hypervolume",
     "botorch_rs.utils.multi_objective.hypervolume", "Hypervolume"),
    ("is_non_dominated", "botorch.utils.multi_objective.pareto",
     "botorch_rs.utils.multi_objective.pareto", "is_non_dominated"),
    # MO-TAF's blend of acquisition functions: native in botorch-rust (BoTorch evaluates
    # openbo's torch.nn.Module instead).
    ("WeightedLogMeanExp", None, "botorch_rs.acquisition.composite", "WeightedLogMeanExp"),
]


def normalize_request(value: str | None) -> str:
    """``botorch``, ``botorch_rs`` or ``auto`` for a ``BO_MODEL_LIBRARY`` value (empty or
    ``None``: ``botorch``)."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return BOTORCH
    text = _ALIASES.get(text, text)
    if text not in REQUESTS:
        raise ValueError(
            f"{ENV_VAR} must be one of {list(REQUESTS)} (or 'botorch-rust'), got {value!r}."
        )
    return text


def _botorch_rs_missing() -> str | None:
    """Why botorch_rs cannot run the multi-objective optimizers and MO-TAF (``None`` if it
    can)."""
    try:
        botorch_rs = importlib.import_module("botorch_rs")
    except ImportError as e:
        return f"botorch_rs is not installed ({e})"
    except Exception as e:  # a broken build must not break openbo's import
        return f"botorch_rs failed to import ({type(e).__name__}: {e})"
    missing = []
    for _, _, module_name, attribute in _NAMES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            missing.append(f"{module_name}.{attribute}")
            continue
        if not hasattr(module, attribute):
            missing.append(f"{module_name}.{attribute}")
    if missing:
        version = getattr(botorch_rs, "__version__", "unknown")
        return f"botorch_rs {version} lacks {', '.join(missing)}"
    return None


def resolve(requested: str | None) -> tuple[str, str]:
    """The library to bind for a request (a ``BO_MODEL_LIBRARY`` value): ``(library,
    reason)``. A request for botorch_rs that it cannot serve falls back to BoTorch."""
    request = normalize_request(requested)
    if request == BOTORCH:
        return BOTORCH, f"{ENV_VAR}={request}"
    missing = _botorch_rs_missing()
    if missing is not None:
        return BOTORCH, f"{ENV_VAR}={request}, but {missing}; falling back to BoTorch"
    version = getattr(importlib.import_module("botorch_rs"), "__version__", "unknown")
    return BOTORCH_RS, f"{ENV_VAR}={request}, botorch_rs {version}"


def namespace(library: str) -> SimpleNamespace:
    """The BoTorch names openbo's optimizers use, from ``library``. ``WeightedLogMeanExp`` is
    ``None`` for BoTorch (openbo blends acquisition functions in a ``torch.nn.Module``)."""
    if library not in LIBRARIES:
        raise ValueError(f"unknown model library {library!r}; expected one of {LIBRARIES}")
    names = {}
    for name, bt_module, rs_module, attribute in _NAMES:
        module_name = bt_module if library == BOTORCH else rs_module
        names[name] = None if module_name is None else _import_name(module_name, attribute)
    return SimpleNamespace(library=library, **names)


def _import_name(module_name: str, attribute: str) -> object:
    """``from module_name import attribute`` (an ImportError if either is missing)."""
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as e:
        raise ImportError(f"cannot import name {attribute!r} from {module_name!r}") from e


def select(requested: str | None = None) -> tuple[str, str]:
    """``(library, reason)`` of an optimizer: the import-time choice for ``None``, else
    :func:`resolve` (with a printed message when it falls back to BoTorch)."""
    if requested is None:
        return MODEL_LIBRARY, MODEL_LIBRARY_REASON
    library, reason = resolve(requested)
    _report(requested, library, reason)
    return library, reason


def _report(requested: str | None, library: str, reason: str) -> None:
    if normalize_request(requested) != BOTORCH and library == BOTORCH:
        print(f"openbo: {reason}", file=sys.stderr, flush=True)


def _resolve_environment() -> tuple[str, str]:
    """The import-time choice; an invalid ``BO_MODEL_LIBRARY`` keeps BoTorch (and says so)
    rather than breaking ``import openbo``."""
    value = os.environ.get(ENV_VAR)
    try:
        library, reason = resolve(value)
    except ValueError as e:
        library, reason = BOTORCH, f"{e} Using BoTorch"
        print(f"openbo: {reason}", file=sys.stderr, flush=True)
        return library, reason
    _report(value, library, reason)
    return library, reason


# The import-time choice (see the module documentation).
MODEL_LIBRARY, MODEL_LIBRARY_REASON = _resolve_environment()
