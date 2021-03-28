import numpy as np
import Levenshtein


def levenshteinDistance(y_true, y_pred):
    return np.mean([
        Levenshtein.distance(true, pred)
        for true, pred in zip(y_true, y_pred)])
