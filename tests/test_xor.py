import torch
import pytest

from chapterseven import xor_problem


def test_xor():
    examples_to_answers = {
        xor_problem.add_bias(torch.tensor([[0], [0]])): torch.tensor([[0]]),
        xor_problem.add_bias(torch.tensor([[0], [1]])): torch.tensor([[1]]),
        xor_problem.add_bias(torch.tensor([[1], [0]])): torch.tensor([[1]]),
        xor_problem.add_bias(torch.tensor([[1], [1]])): torch.tensor([[0]])
    }

    for ex, a in examples_to_answers.items():
        assert xor_problem.xor_perceptron(ex) == a

