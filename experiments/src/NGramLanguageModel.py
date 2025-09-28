import math
from typing import Any, Dict, List, Optional, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

START_TOKEN = '<S>'
UNKNOWN_TOKEN = '<UNK>'

# Type Aliases
NGram = Tuple[str, ...]
Token = str
Dataset = Iterable[str]


class NGramLanguageModel():
    """
    A basic n-gram language model. 
    Statistics are computed upon instantiation.
    """

    def __init__(self, n: int, smoothing: int, coverage: float, training_data: Dataset):
        """
        Initialize the model.
        
        Args:
            n: the size of the n-grams
            smoothing: the smoothing factor, e.g., 0 for none, 1 for laplace, etc.
            unknown_epsilon: the log2 count distance below which a token is mapped to <UNK>
            training_data: Iterable of training strings (can be loaded line by line)
            validation_data: Iterable of validation strings (can be loaded line by line)
        """
        self.training_data: Dataset = training_data
        if (coverage < 0 or coverage > 1):
            raise ValueError('coverage must be between 0 and 1')
        self.coverage: float = coverage
        self.n: int = n
        if (n <= 0):
            raise ValueError('n must be greater than 0')
        self.smoothing: int = smoothing
        if (smoothing < 0):
            raise ValueError('smoothing must be greater or equal to 0')
        
        self.vocabulary: set[Token] = NGramLanguageModel._get_vocabulary(training_data, coverage)
        self.ngram_counts: Dict[NGram, int] = NGramLanguageModel._get_ngram_counts(training_data, self.vocabulary, n)
        self.all_tokens_count: int = sum(self.ngram_counts.values())
        self.context_counts: Dict[NGram, int] = NGramLanguageModel._get_context_counts(self.ngram_counts) if n > 1 else {}

        self.DEBUG = False
    
    def get_probability(self, *tokens: Token) -> float:
        """
        Get the probability of a token given zero or more other tokens.
        
        Args:
            *tokens: The target token and optional givens to get probability for
            
        Returns:
            The probability of the token given the context, or 0.0 if not found
        """
        
        if len(tokens) != self.n:
            raise ValueError('Length of tokens must match n-gram length')
        
        vocabulary_size = len(self.vocabulary)
        
        # handle unigram case
        if self.n == 1:
            a = (self.ngram_counts.get(tokens, 0) + self.smoothing)
            b = (self.all_tokens_count + (self.smoothing * vocabulary_size))
            return a / b if b > 0 else 0.0
        
        # handle ngram case
        else:
            preceding = tokens[:-1]

            # Count of the full n-gram tokens
            a = self.ngram_counts.get(tokens, 0) + self.smoothing
            
            # Count of the (n-1) preceding tokens
            b = self.context_counts.get(preceding, 0) + (self.smoothing * vocabulary_size)

            if self.DEBUG:
                print(str(a) + " / " + str(b) + " - " + ' '.join(preceding) + " [" + ' '.join(tokens) + "]")

            return a / b if b > 0 else 0.0


    def get_perplexity(self, datum: str) -> float:
        ngrams: list[NGram] = []

        ngrams.extend(NGramLanguageModel._get_ngrams_from_line(datum, self.vocabulary, self.n))
        nll = - np.log([self.get_probability(*tokens) for tokens in ngrams])
        pp: np.float64 = np.exp(np.mean(nll))

        return pp
    
    def get_mean_perplexity(self, data: Dataset) -> float:
        pps: list[float] = []
        for line in data:
            pp = self.get_perplexity(line)
            pps.append(pp)
        return float(np.mean(pps))

    @staticmethod
    def _get_context_counts(ngram_counts: Dict[NGram, int]) -> Dict[NGram, int]:
        """
        Calculates (n-1)-gram context counts from n-gram counts.
        """
        context_counts: Dict[NGram, int] = dict()
        if not ngram_counts:
            return context_counts
            
        for ngram, count in ngram_counts.items():
            context = ngram[1:]
            context_counts[context] = context_counts.get(context, 0) + count
        return context_counts
    
    @staticmethod
    def _get_vocabulary(data: Dataset, coverage: float) -> set[Token]:
        
        token_counts: dict[str, int] = dict()
        for line in data:
            for token in NGramLanguageModel._parse_tokens(line):
                token_counts[token] = token_counts.get(token, 0) + 1
                
        token_counts = NGramLanguageModel.apply_unk_by_coverage(token_counts, coverage)
        
        vocabulary = set(token_counts.keys())
        return vocabulary
    
    @staticmethod
    def apply_unk_by_coverage(token_counts, coverage):
        total_mass = sum(token_counts.values())
        sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        updated_token_counts: dict[str, int] = dict()
        cumulative_mass = 0
        unknown_count = 0
        for token, count in sorted_token_counts:
            if cumulative_mass / total_mass < coverage:
                updated_token_counts[token] = count
                cumulative_mass += count
            else:
                unknown_count += count
        updated_token_counts[UNKNOWN_TOKEN] = unknown_count
        return updated_token_counts

    @staticmethod
    def _get_ngram_counts(data: Dataset, vocabulary: set[Token], n: int) -> Dict[NGram, int]:
        """
        Build the statistics to be used by the model
        """
        
        counts: Dict[NGram, int] = dict()

        for line in data:
            ngrams = NGramLanguageModel._get_ngrams_from_line(line, vocabulary, n)
            for key in ngrams:
                counts[key] = counts.get(key, 0) + 1
               
        return counts
    
    
    @staticmethod
    def _parse_tokens(line: str) -> list[Token]:
        """
        Parse and split a line into tokens.
        """
        # For now, just split on whitespace characters.
        # Might consider more sophisticated approach such as using "stop words", etc.
        return line.lower().split()

    
    @staticmethod
    def _get_ngrams_from_line(line: str, vocabulary: set[Token], n: int) -> List[NGram]:
        """
        Get the n-grams from a line of text.
        
        Args:
            line: The line of text to parse

        Returns:
            A list of n-grams extracted from the line
        """
        
        ngrams: list[NGram] = []
        # Pad the preceding tokens with the start token
        preceding = [START_TOKEN] * n
        for token in NGramLanguageModel._parse_tokens(line):
            if token not in vocabulary:
                token = UNKNOWN_TOKEN
            preceding.append(token)
            if len(preceding) > n:
                preceding.pop(0)
            ngram: NGram = tuple(preceding)
            ngrams.append(ngram)
                
        return ngrams