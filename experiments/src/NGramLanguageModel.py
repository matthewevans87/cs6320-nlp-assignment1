import math
from typing import Any, Dict, List, Optional, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

START_TOKEN = '<S>'
UNKNOWN_TOKEN = '<UNK>'

class NGramLanguageModel():
    """
    A basic n-gram language model. 
    Statistics are computed upon instantiation.
    """

    def __init__(self, n: int, smoothing: int, unknown_epsilon: float, training_data: Iterable[str], validation_data: Iterable[str]):
        """
        Initialize the base model.
        
        Args:
            n: the size of the n-grams
            smoothing: the smoothing factor, e.g., 0 for none, 1 for laplace, etc.
            unknown_epsilon: the log2 count distance below which a token is mapped to <UNK>
            training_data: Iterable of training strings (can be loaded line by line)
            validation_data: Iterable of validation strings (can be loaded line by line)
        """
        self.training_data: Iterable[str] = training_data
        self.validation_data: Iterable[str] = validation_data
        self.unknown_epsilon: float = unknown_epsilon
        self.n: int = n
        if (n <= 0):
            raise ValueError('n must be greater than 0')
        self.smoothing: int = smoothing
        if (smoothing < 0):
            raise ValueError('smoothing must be greater or equal to 0')
        self.vocabulary: set[str]
        self.vocabulary_size: int
        self.init_vocabulary()
        
        self.counts: Dict[Tuple[str, ...], int]
        self.all_tokens_count: int
        self.init_statistics()
    
    
    def get_probability(self, *tokens: str) -> float:
        """
        Get the probability of a token given zero or more other tokens.
        
        Args:
            *tokens: The target token and optional givens to get probability for
            
        Returns:
            The probability of the token given the context, or 0.0 if not found
        """
        
        if len(tokens) != self.n:
            raise ValueError('Length of tokens must match n-gram length')
        
        # TODO: Handle <UNK> case
        
        # handle unigram case
        if self.n == 1:
            return (self.counts.get(tokens, 0) + self.smoothing) / (self.all_tokens_count + (self.smoothing * self.vocabulary_size))
        
        # handle ngram case
        else:
            token = tokens[0]
            preceding = tokens[1:]
            a = self.counts.get((token,), 0) + self.smoothing
            b = self.counts.get(preceding, 0) + (self.smoothing * self.vocabulary_size)
            return a / b
    
    
    def init_vocabulary(self) -> None:
        
        highest_frequency = 0
        token_counts: dict[str, int] = dict()
        for line in self.training_data:
            for token in self.parse_tokens(line):
                token_counts[token] = token_counts.get(token, 0) + 1
                if token_counts[token] > highest_frequency:
                    highest_frequency = token_counts[token]
                
        # Handle mapping tokens to <UNK>
        p = math.log2(highest_frequency)
        t = 2 ** (p - self.unknown_epsilon)
        
        unk_count = 0
        for token, count in list(token_counts.items()):
            if count < t:
                unk_count += count
                del token_counts[token]
        
        token_counts[UNKNOWN_TOKEN] = unk_count
        
        self.vocabulary = set(token_counts.keys())
        self.vocabulary_size = len(self.vocabulary)
    
        
    def init_statistics(self) -> None:
        """
        Build the statistics to be used by the model
        """
        
        counts: Dict[Tuple[str, ...], int] = dict()
        
        for line in self.training_data:
            # Pad the preceding tokens with the start token
            preceding = ["<S>"] * self.n
            for token in self.parse_tokens(line):
                if token not in self.vocabulary:
                    token = UNKNOWN_TOKEN
                preceding.append(token)
                if len(preceding) > self.n:
                    preceding.pop(0)
                key: Tuple[str, ...] = tuple(preceding)
                counts[key] = counts.get(key, 0) + 1
               
        self.counts = counts
        self.all_tokens_count = sum(counts.values())


    def get_perplexity(self) -> float:
        # build validation set n-grams
        validation_ngrams: list[tuple[str, ...]] = []
        
        for line in self.validation_data:
            # Pad the preceding tokens with the start token
            preceding = [START_TOKEN] * self.n
            for token in self.parse_tokens(line):
                if token not in self.vocabulary:
                    token = UNKNOWN_TOKEN
                preceding.append(token)
                if len(preceding) > self.n:
                    preceding.pop(0)
                ngram: Tuple[str, ...] = tuple(preceding)
                validation_ngrams.append(ngram)
        
        pp = np.exp(1/self.vocabulary_size * np.sum(- np.log([self.get_probability(*tokens) for tokens in validation_ngrams])))
        return pp
    
    
    def parse_tokens(self, line: str) -> list[str]:
        """
        Parse and split a line into tokens.
        """
        # For now, just split on whitespace characters.
        # Count consider more sophisticated approach such as using "stop words", etc.
        return line.lower().split()