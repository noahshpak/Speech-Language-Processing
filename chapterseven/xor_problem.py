import torch

"""
I quickly coded this up to test out the XOR solution from section 7.2.1
"""


def add_bias(v):
    return torch.cat([v, torch.tensor([[1]])])


def xor_perceptron(x):
    h1 = [1,1,0]                # weights for H1
    h2 = [1,1,-1]               # weights for H2
    W = torch.tensor([h1, h2])
    y = torch.tensor([[1, -2, 0]])
    h_out = torch.relu(W @ x)
    add_y_bias = add_bias(h_out)
    return torch.relu(y @ add_y_bias)