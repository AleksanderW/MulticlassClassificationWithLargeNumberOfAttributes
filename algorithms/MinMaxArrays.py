import numpy as np

def find_min_max_pairs(subtable):
    subtable = np.array(subtable)
    min_probs = np.min(subtable, axis=0)
    max_probs = np.max(subtable, axis=0)
    min_max_pairs = np.stack((min_probs, max_probs), axis=-1)
    return min_max_pairs.tolist()