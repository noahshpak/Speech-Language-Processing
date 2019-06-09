import torch
from typing import List


class SkipGrams:
    MAX_VOCAB = 20000

    def __init__(self, data: List[str], context_size: int):
        self.data = data
        self.context_size = context_size

