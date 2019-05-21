import pytest
from chapterfour import classifier

def test_classifier():
    documents = [
        ["good", "good", "good", "great", "great", "great"],
        ["poor", "great", "great"],
        ["good", "poor", "poor", "poor"],
        ["good", "poor", "poor", "poor", "poor", "poor", "great", "great"],
        ["poor", "poor"]
    ]

    labels = [
        "pos",
        "pos",
        "neg",
        "neg",
        "neg"
    ]

    model = classifier.NaiveBayesClassifier(documents, labels)
    assert model.class_prior == {'neg': 3, 'pos': 2}
    assert model.class_word_counts == {
        'neg': {
            'good': 2,
            'great': 2,
            'poor': 10
        },
        'pos': {
            'good': 3,
            'great': 5,
            'poor': 1
        }
    }

    
    binary = classifier.NaiveBayesClassifier(documents, labels, model='binary')
    assert binary.class_word_counts == {
        'neg': {
            'good': 2,
            'great': 1,
            'poor': 3
        },
        'pos': {
            'good': 1,
            'great': 2,
            'poor': 1
        }
    }

    
