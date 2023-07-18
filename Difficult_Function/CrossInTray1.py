from typing import List
import numpy as np
import scipy.special as spc

class CrossInTray_function:
    def __init__(self, A: float, B: float, C: float, X_range: List[List[float]]):

        """

        The CrossInTray Function

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
        
        The main functon of the CrossInTray Function

        Arguments
        ---------
        X (np.array): The inputed value of the variable of the CrossInTray Fucntion

        Returns
        -------
        (np.array): The output value of the CrossInTray Function respected to the inputed value X
        
        
        """


        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range): 
                return (self.B * (np.abs(np.sin(X[0]) * np.sin(X[1]) * np.exp(np.abs(100 - np.sqrt(X[0]**2 + X[1]**2) / np.pi))) + 1)**self.C + self.A)
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = (self.B * (np.abs(np.sin(X[in_range, 0]) * np.sin(X[in_range, 1]) * np.exp(np.abs(100 - np.sqrt(X[in_range, 0]**2 + X[in_range, 1]**2) / np.pi))) + 1)**self.C + self.A)
            return Res


class CrossInTray_likeli:
    def __init__(self, FD, A: float, X_range: List[List[float]]):

        """

        The CrossInTray Function

        Arguments
        ---------
        FD : the fake data for the CrossInTray
        C (float): the value of the paramter C
        X_range (List[List[float]]): the range of the CrossInTray Function
     
        Returns
        -------
        None

        """


        self.FD = FD
        self.A = A
        self.X_range = X_range

    def l(self, params)->float:

        """
        
        The likelihood function of the Sinewave Function

        Arguments
        ---------
        params (np.array): the value of the parameters of the Sinewave function.

        Returns
        -------
        (float): the likelihood value of the inputed parameter values of the Sinewave function.

        """

        self.B = params[0]
        self.C = params[1]
        if (self.B<0 or self.C<0 or self.C>np.pi):
            return 0
        self.lambda_theta = np.floor(self.pdf(np.column_stack((self.FD.x.flatten(), self.FD.y.flatten())))*((self.FD.Data_range[0][1]-self.FD.Data_range[0][0])*(self.FD.Data_range[1][1]-self.FD.Data_range[1][0])/(self.FD.bins)**2)*self.FD.scaler)
        likeli = np.power(self.lambda_theta, self.FD.FakeData)*np.exp(-self.lambda_theta)/(spc.factorial(self.FD.FakeData))
        return np.sum(likeli)
        
    def pdf(self, X: np.array):

        """
        
        The main functon of the Sinewave Function

        Arguments
        ---------
        X (np.array): The inputed value of the variable of the Sinewave Fucntion

        Returns
        -------
        (np.array): The output value of the Sinewave Function respected to the inputed value X
        
        
        """


        if (len(X.shape)==1):
            X0_in_range = np.logical_and(X[0] >= self.X_range[0][0], X[0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[1] >= self.X_range[1][0], X[1] <= self.X_range[1][1])
            if (X0_in_range & X1_in_range): 
                return (self.B * (np.abs(np.sin(X[0]) * np.sin(X[1]) * np.exp(np.abs(100 - np.sqrt(X[0]**2 + X[1]**2) / np.pi))) + 1)**self.C + self.A)
            else:
                return 0
        else:
            X0_in_range = np.logical_and(X[:, 0] >= self.X_range[0][0], X[:, 0] <= self.X_range[0][1])
            X1_in_range = np.logical_and(X[:, 1] >= self.X_range[1][0], X[:, 1] <= self.X_range[1][1])
            in_range = np.logical_and(X0_in_range, X1_in_range)
            Res = np.zeros(X.shape[0])
            Res[in_range] = (self.B * (np.abs(np.sin(X[in_range, 0]) * np.sin(X[in_range, 1]) * np.exp(np.abs(100 - np.sqrt(X[in_range, 0]**2 + X[in_range, 1]**2) / np.pi))) + 1)**self.C + self.A)
            return Res