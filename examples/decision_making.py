"""Continuous FPBI sampling followed by the paper's knee/preference decision."""

import torch

from openbo.decision_making import knee_region, sample_fpbi, select_preferred


def costs(X: torch.Tensor) -> torch.Tensor:
    x = X[..., 0]
    return torch.stack((x.square(), (1 - x).square()), dim=-1)


def main() -> None:
    torch.set_printoptions(precision=4)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    result = sample_fpbi(
        costs, bounds, n_steps=5, num_restarts=3, maximize=False, seed=0
    )
    mask = knee_region(result.Y, alpha=0.5, maximize=False)
    index = select_preferred(result.Y, [70.0, 30.0], alpha=0.5, maximize=False)
    print("Sampled designs:\n", result.X)
    print("Cost outcomes:\n", result.Y)
    print("Knee region:", mask.tolist())
    print("Selected design:", result.X[index].tolist())
    print("Selected costs:", result.Y[index].tolist())
    print(
        "Successful rays:",
        int(result.ray_succeeded.sum()),
        "/",
        len(result.ray_succeeded),
    )


if __name__ == "__main__":
    main()
