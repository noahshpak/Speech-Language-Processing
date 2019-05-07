import spacy
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

BEGIN_TOKEN = "<s>"
END_TOKEN = "</s>"

class LM:
    def __init__(self, n: int):
        self.n = n
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, sentence: str) -> List[str]:
        begin = [BEGIN_TOKEN] * self.n
        end = [END_TOKEN] * self.n
        return begin + [token.text.lower() for token in self.nlp(sentence)] + end

    def train(self, corpus: List[str]) -> Dict:
        tokens = [t for sentence in corpus for t in self.tokenize(sentence)]
        if self.n == 1:
            return {t: count/len(tokens) for t,count in Counter(tokens).items()}
        if self.n > len(tokens):
            raise ValueError(f"Corpus of length {len(tokens)} is not long enough for ngram of size {self.n}")
        ngrams, context_grams = defaultdict(int), defaultdict(int)
        for i in range(len(tokens)-self.n):
            ngrams[tuple(tokens[i:i+self.n])] += 1
            context_grams[tuple(tokens[i:i+self.n-1])] += 1
        else:
            context_grams[tuple(tokens[i+1:])] += 1

        self.ngrams = ngrams
        self.context_grams = context_grams

    def proba(self, w: str, ctx: Tuple[str]):
        ngram = tuple(list(ctx) + [w])
        if not self.context_grams[ctx]: return 0
        return self.ngrams[ngram] / self.context_grams[ctx]
    
    def sequence_probability(self, sentence: str):
        tokens = self.tokenize(sentence)
        prob = 1
        for i in range(len(tokens)-self.n):
            prob *= self.proba(tokens[i], tuple(tokens[i:i+self.n]))
        return prob

