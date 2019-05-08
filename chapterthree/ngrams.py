import spacy
import numpy as np
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
        self.tokens = [t for sentence in corpus for t in self.tokenize(sentence)]
        self.vocab = list(set(self.tokens))
        if self.n == 1:
            self.unigrams = {t: count/len(self.tokens) for t,count in Counter(self.tokens).items()}
            return
        
        if self.n > len(self.tokens):
            raise ValueError(f"Corpus of length {len(self.tokens)} is not long enough for ngram of size {self.n}")
        
        self.ngrams, self.context_grams = defaultdict(int), defaultdict(int)
        for i in range(len(self.tokens)-self.n):
            self.ngrams[tuple(self.tokens[i:i+self.n])] += 1
            self.context_grams[tuple(self.tokens[i:i+self.n-1])] += 1
        else:
            self.context_grams[tuple(self.tokens[i+self.n:])] += 1

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

    def next_most_likely(self, seed: Tuple[str]) -> Tuple[str, float]:
        seed = tuple(t.lower() for t in seed)
        next_word_space = np.array([self.proba(w, seed) for w in self.vocab])
        most_likely_idx = np.argmax(next_word_space)
        most_likely_prob = next_word_space[most_likely_idx]
        same_prob = [(self.vocab[i], prob) for i, prob in enumerate(next_word_space) if prob == most_likely_prob]
        return same_prob[np.random.randint(0, len(same_prob))]
        
