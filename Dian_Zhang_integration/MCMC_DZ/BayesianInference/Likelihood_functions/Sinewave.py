from typing import List
import numpy as np

class Sinewave_function:
    def __init__(self, params: np.array, X_range: List[List[float]]):

        """

        The Sinusoidal Function

        Arguments
        ---------
     
        Returns
        -------
        None


        """

        self.params = params
        self.X_range = X_range

    def f(self, X: np.array):

        """
        
        The main functon of the Sinewave Function

        Arguments
        ---------
        X (np.array): The inputed value of the variable of the Sinusoidal Fucntion

        Returns
        -------
        (np.array): The output value of the Sinusoidal Function respected to the inputed value X
        
        
        """

        [self.B, self.C] = self.params
        if (self.C<(-np.pi) or self.C>np.pi):
            if (len(X.shape)==1):
                return 0
            else:
                return np.zeros(X.shape[0])

        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range): 
                return (np.sin(self.B * (np.sqrt(X[0] ** 2 + X[1] ** 2) - self.C)) + 1)
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = (np.sin(self.B * (np.sqrt(X[in_range, 0] ** 2 + X[in_range, 1] ** 2) - self.C)) + 1)
            return Res