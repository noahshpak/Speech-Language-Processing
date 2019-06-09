import pytest

from chaptersix import word2vec


def test_init():
    assert word2vec.SkipGrams(["Hello", "World"], 1) is not None

