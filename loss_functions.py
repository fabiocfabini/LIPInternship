import torch

def mse_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ij,i->ij', (input - target)**2, weights).mean()

def mae_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ij,i->ij', torch.abs(input - target), weights).mean()