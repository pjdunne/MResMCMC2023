#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:03:32 2023

@author: firework
"""
from typing import List
import numpy as np
class Sinusoidal:
    def __init__(self, A: np.array, B: np.array, C: np.array, X_range: List[List[float]]):

        """

        The Sinusoidal Function

        Arguments
        ---------
     
        Returns
        -------
        None


        """


        self.A = A
        self.B = B
        self.C = C
        self.X_range = X_range

    def f(self, X: np.array):

        """
        
        The main functon of the Sinusoidal Function

        Arguments
        ---------
        X (np.array): The inputed value of the variable of the Sinusoidal Fucntion

        Returns
        -------
        (np.array): The output value of the Sinusoidal Function respected to the inputed value X
        
        
        """


        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range):
                return (self.A[0] * np.cos(self.B[0] * X[1] + self.C[0]))
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = (self.A[0] * np.cos(self.B[0] * X[1] + self.C[0]))
            return Res
