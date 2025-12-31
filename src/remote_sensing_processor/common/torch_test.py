"""Torch testing functions."""

import torch


def cuda_test() -> bool:
    """Test if CUDA is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()
