import math
import torch


def inverse_sigmoid(x):
    return math.log(x/(1-x))


def inverse_sigmoid_tensor(x):
    return torch.log(x/(1-x))