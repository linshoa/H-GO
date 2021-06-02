import torch
from torch import nn
from torch.nn import functional as F

class Mish(nn.Module):
    def __init__(self):
       super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x