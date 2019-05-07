import pytest
from chapterthree import ngrams

def test_unigrams():
    corpus = [ "I am Sam", "Sam I am", "I do not like green eggs and ham"]
    lm = ngrams.LM(n=1)
    lm.train(corpus)
    probability = 1 / 10
    assert lm.proba("Sam", ())  == probability

def test_bigrams():
    corpus = [ "I am Sam", "Sam I am", "I do not like green eggs and ham"]
    lm = ngrams.LM(n=2)
    lm.train(corpus)
    assert lm.proba("I", ("<s>",)) == 2/3