from typing import Iterator, Tuple

import numpy as np


class PurgedTimeSeriesSplit:
    """
    Time Series Cross Validator with Purging and Embargo.
    Prevents data leakage by removing training samples that overlap with test evaluation time.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        
    def split(self, X: np.ndarray, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(1, self.n_splits + 1):
            train_end = i * test_size - self.purge_gap
            test_start = i * test_size
            test_end = (i + 1) * test_size
            
            if train_end <= 0:
                continue
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
