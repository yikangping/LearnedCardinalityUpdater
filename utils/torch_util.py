import torch


def get_torch_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)
    return device
