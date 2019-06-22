import pytest

from chaptersix import word2vec


def test_init():
    assert word2vec.SkipGrams(["Hello", "World"], embedding_dim=20, context_size=2) is not None

