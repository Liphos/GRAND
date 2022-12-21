"""To test all the gpu needed functions"""
import torch

def test_is_gpu_available():
    """Test if gpu is available"""
    print(torch.cuda.is_available())