import torch.utils.data
from typing import List

data = []
context_size = 5
MAX_VOCAB = 20000


def create_lm_dataset(tokens: List[str], context_size: int):
    """
    :param data: list of tokenized sentences [['the', 'brown', 'fox'], ['I', 'am', 'free']
    :param context_size: number of context words to provide the model as input (c -> w)
    :return: V: vocabulary, X: Tensor(num_samples, context_size), Y: Tensor(num_samples, 1)
    """
    vocab = set(t for t in tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    X, y = [], []
    for i in range(context_size, len(tokens) - context_size):
        context = tokens[i-context_size:i]
        target = tokens[i]
        X.append(context)
        y.append(target)

    # convert to indices
    x_train = torch.tensor([[word_to_ix[w] for w in context] for context in X], dtype=torch.long)
    y_train = torch.tensor([word_to_ix[t] for t in y], dtype=torch.long)
    return x_train, y_train, vocab, word_to_ix


def create_cbow_dataset(tokens: List[str], context_size: int):
    vocab = set(t for t in tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    X, y = [], []
    for i in range(context_size, len(tokens) - context_size):
        context = tokens[i - context_size:i] + tokens[i+1:i+1+context_size]
        target = tokens[i]
        X.append(context)
        y.append(target)

    # convert to indices
    x_train = torch.tensor([[word_to_ix[w] for w in context] for context in X], dtype=torch.long)
    y_train = torch.tensor([word_to_ix[t] for t in y], dtype=torch.long)
    return x_train, y_train, vocab, word_to_ix