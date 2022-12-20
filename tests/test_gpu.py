import torch

def test_is_gpu_available():
    print(torch.cuda.is_available())