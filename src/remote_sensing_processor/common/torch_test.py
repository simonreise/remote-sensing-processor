import torch


def cuda_test():
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        return 'CUDA or MPS is not available'