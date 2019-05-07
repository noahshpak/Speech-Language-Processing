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
        begin = [BEGIN_TOKEN] * (self.n-1) if self.n > 1 else [BEGIN_TOKEN]
        end = [END_TOKEN] * (self.n-1) if self.n > 1 else [END_TOKEN]
        return begin + [token.text.lower() for token in self.nlp(sentence)] + end

    def train(self, corpus: List[str]) -> Dict:
        tokens = [t for sentence in corpus for t in self.tokenize(sentence)]
        if self.n == 1:
            self.unigrams = {t: count/len(tokens) for t,count in Counter(tokens).items()}
        if self.n > len(tokens):
            raise ValueError(f"Corpus of length {len(tokens)} is not long enough for ngram of size {self.n}")
        ngrams, context_grams = defaultdict(int), defaultdict(int)
        for i in range(len(tokens)-self.n):
            ngrams[tuple(tokens[i:i+self.n])] += 1
            context_grams[tuple(tokens[i:i+self.n-1])] += 1
        else:
            context_grams[tuple(tokens[i+self.n:])] += 1

        self.ngrams = ngrams
        self.context_grams = context_grams

    def proba(self, w: str, ctx: Tuple[str]):
        if self.n == 1:
            return self.unigrams.get(w.lower(), 0.0) 
        ngram = tuple([c.lower() for c in list(ctx)] + [w.lower()])
        if not self.context_grams[ctx]: return 0.0
        return self.ngrams[ngram] / self.context_grams[ctx]
    
    def sequence_probability(self, sentence: str):
        tokens = self.tokenize(sentence)
        prob = 1.0
        for i in range(len(tokens)-self.n):
            prob *= self.proba(tokens[i], tuple(tokens[i:i+self.n]))
        return prob
        
