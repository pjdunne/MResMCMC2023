from typing import List
import numpy as np
class Periodic_Func2D:
    def __init__(self, Freq: np.array, X_range: List[List[float]]):

        """

        The Periodic Function

        Arguments
        ---------
        Freq (np.array): The frequency of each axis of the Periodic Function
        X_range (np.array): The range of each axis of the Periodic Function

        Returns
        -------
        None


        """


        self.Freq = Freq
        self.X_range = X_range

    def f(self, X: np.array):

        """
        
        The main functon of the Periodic Function

        Arguments
        ---------
        X (np.array): The inputed value of the variable of the Periodic Fucntion

        Returns
        -------
        (np.array): The output value of the Periodic Function repect to the inputed value X
        
        
        """


        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range):
                return (np.sin(2 * np.pi * self.Freq[0] * X[0]) * np.cos(2 * np.pi * self.Freq[1] * X[1]) + 1)/2
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = (np.sin(2 * np.pi * self.Freq[0] * X[in_range, 0]) * np.cos(2 * np.pi * self.Freq[1] * X[in_range, 1]) + 1)/2
            return Res
