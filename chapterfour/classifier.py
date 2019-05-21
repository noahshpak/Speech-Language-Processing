import math
from typing import List
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    """
    Multinomial NB & Binary NM
    Inputs are token
    """
    def __init__(self, documents: List[List[str]], labels: List[str], model='multi'):
        """
        Parameters
        ----------
        documents : List[List[str]]
            The training set of tokens `documents`.
        y: List[str]
            The labels for each document ordered such that labels[i] corresponds to document[i]
        model: str
            One of `multi` for multinomial NB or `binary` for binarized NB
        """
        if len(documents) != len(labels):
            raise ValueError("Documents and Labels are not a 1-to-1 mapping.")

        self.documents = documents
        self.labels = labels
        self.vocab = set(w for d in documents for w in d)
        self.classes = sorted(list(set(labels)))
        self.class_prior = Counter(labels)
        self.class_documents = defaultdict(list)
        self.class_document_list = defaultdict(list)
        self.model = model

        for c, d in zip(labels, documents):
            self.class_documents[c] += d  # concatenate all documents of the same class
            self.class_document_list[c].append(d) 

        self.train()


    def train(self):
        if self.model == 'multi':
            self.class_word_counts = {
                c: Counter(self.class_documents[c]) 
                for c in self.classes
            }
            self.class_word_prob = {
                c: { w: math.log((self.class_word_counts[c][w] + 1) / (len(self.class_documents[c]) + len(self.vocab)))  # add-1 smoothing
                     for w in self.class_documents[c]
                } for c in self.classes
            }
        elif self.model == 'binary':
            self.class_word_counts = {c: defaultdict(int)  for c in self.classes}
            for c in self.classes:
                for doc in self.class_document_list[c]:
                    for w in set(doc):
                        self.class_word_counts[c][w] += 1

            self.class_word_prob = {
                c: { w:  math.log((self.class_word_counts[c][w] + 1) / (len(self.class_documents[c]) + len(self.vocab)))  # add-1 smoothing
                     for w in self.class_documents[c]
                } for c in self.classes
            }

    def predict(self, doc: List[str]):
        probas = {c: 0 for c in self.classes}
        for c in self.classes:
            probas[c] += self.class_prior[c] / len(self.labels)
            for pword in doc:
                if pword in self.vocab:
                    probas[c] += self.class_word_prob[c][pword]
        return probas

