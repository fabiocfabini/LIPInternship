import torch

def mse_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.pow(input - target, 2).sum(dim=1) * weights

def mae_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """return torch.abs(input - target) * weights
    """
    return torch.abs(input - target).sum(dim=1) * weights