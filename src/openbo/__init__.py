"""OpenBO package."""

# The model library of the BoTorch-based multi-objective optimizers (BoTorch or
# botorch-rust, chosen with BO_MODEL_LIBRARY when openbo is imported; see
# openbo.model_library).
from openbo.model_library import MODEL_LIBRARY
from openbo.test_functions.synthetic import (
    ackley,
    branin,
    hartmann6,
    rastrigin,
    rosenbrock,
    sphere,
)

__all__ = ["MODEL_LIBRARY", "branin", "sphere", "ackley", "rastrigin", "rosenbrock", "hartmann6"]
