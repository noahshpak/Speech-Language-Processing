import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from typing import List, Dict
from chaptersix.dataset import create_cbow_dataset


class CBOW(nn.Module):
    def __init__(self, vocabulary_size, context_size, embedding_dim):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((inputs.size(0), self.context_size*2, -1))
        logits = self.out(torch.sum(embeds, dim=1))  # sum all context embeddings
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs


class SkipGrams(nn.Module):
    def __init__(self, vocabulary: Dict, embedding_dim: int, context_size: int):
        super().__init__()
        self.window_size = context_size
        self.num_outputs = context_size * 2

        self.embedding_size = embedding_dim
        self.vocabulary = vocabulary

        self.embedding = nn.Embedding(len(vocabulary), embedding_dim)

    def forward(self, batch_of_context_vectors):
        embeds = self.embeddings(batch_of_context_vectors).view((batch_of_context_vectors.size(0), -1))
        log_probs = F.log_softmax(batch_of_context_vectors, dim=1)
        return log_probs



if __name__ == '__main__':
    data = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()
    context_size = 2
    embedding_dim = 50
    x_train, y_train, vocab, word_to_ix = create_cbow_dataset(data, context_size=context_size)

    model = CBOW(vocabulary_size=len(vocab), context_size=context_size, embedding_dim=embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.002)

    bs = 10

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    loss_function = nn.NLLLoss(reduction='mean')

    for epoch in range(1000):
        for xb, yb in train_dl:
            model.zero_grad()
            pred = model(xb)
            loss = loss_function(pred, yb)
            loss.backward()
            optimizer.step()
            if epoch % 10: print(loss)


