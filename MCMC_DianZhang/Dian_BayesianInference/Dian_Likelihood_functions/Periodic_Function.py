from typing import List
import numpy as np
class Periodic_Func2D:
    def __init__(self, Freq: np.array, X_range: List[List[float]]):
        self.Freq = Freq
        self.X_range = X_range

    def f(self, X):
        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range):
                return np.sin(2 * np.pi * self.Freq[0] * X[0]) * np.cos(2 * np.pi * self.Freq[1] * X[1])
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = np.sin(2 * np.pi * self.Freq[0] * X[in_range, 0]) * np.cos(2 * np.pi * self.Freq[1] * X[in_range, 1])
            return Res
