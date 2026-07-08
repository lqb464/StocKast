import numpy as np
from src.features.purged_cv import PurgedTimeSeriesSplit

def test_purged_cv_no_leakage():
    X = np.arange(100)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=5)
    
    for train_idx, test_idx in cv.split(X):
        # Assert no overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        
        # Assert temporal order (all train is before test)
        assert np.max(train_idx) < np.min(test_idx)
        
        # Assert gap is respected
        assert np.min(test_idx) - np.max(train_idx) >= 5
