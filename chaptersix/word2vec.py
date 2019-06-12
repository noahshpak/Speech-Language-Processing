import torch.nn as nn
import torch.utils.data
import torch.functional as F

from typing import List, Dict


class CBOW(nn.Module):
    pass


class SkipGrams(nn.Module):
    def __init__(self, vocabulary: Dict, embedding_size: int, window_size: int):
        self.window_size = window_size
        self.num_outputs = window_size * 2

        self.embedding_size = embedding_size
        self.vocabulary = vocabulary

        self.embedding = nn.Embedding(len(vocabulary), embedding_size)

    def forward(self, batch_of_context_vectors):
        embeds = self.embeddings(batch_of_context_vectors).view((1, -1))
        log_probs = F.log_softmax(batch_of_context_vectors, dim=1)
        return log_probs
