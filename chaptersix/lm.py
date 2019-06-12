"""
A neural probabilistic language model
http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from dataset import create_lm_dataset

class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, intermediate_rep_dim):
        super(NNLM, self).__init__()
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(context_size*embedding_dim, intermediate_rep_dim)
        self.out = nn.Linear(intermediate_rep_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.size(0), -1))
        out = F.hardtanh(self.linear(embeds))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def get_model(embedding_dim, context_size, data, lr=0.005, intermediate_rep_dim=64):
    x_train, y_train, vocab, word_to_ix = create_lm_dataset(data, context_size=context_size)

    lm_model = NNLM(vocab_size=len(vocab),
                    embedding_dim=embedding_dim,
                    context_size=context_size,
                    intermediate_rep_dim=intermediate_rep_dim)

    return lm_model, optim.SGD(lm_model.parameters(), lr=lr), x_train, y_train, vocab, word_to_ix


if __name__ == '__main__':
    data = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    model, optimizer, x_train, y_train, vocab, word_to_ix = get_model(embedding_dim=64, context_size=3, data=data)

    bs = 10

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    loss_function = nn.NLLLoss(reduction='mean')

    for epoch in range(1000):
        for xb,yb in train_dl:
            model.zero_grad()
            pred = model(xb)
            loss = loss_function(pred, yb)
            loss.backward()
            optimizer.step()
            if epoch % 10: print(loss)
