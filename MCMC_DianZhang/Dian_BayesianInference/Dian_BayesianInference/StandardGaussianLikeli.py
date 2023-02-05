import numpy as np
from typing import List

class Likelihood_StandardGaussian:
    def __init__(self, Dim:int, Dataset: List[List[float]]) -> None:

        """
        
        The likelihood function of the Standard Gaussian distribution

        Arguments
        ---------
        Dim: the dimension of the inputted parameter
        Dataset: the dataset used to generate the Likelihood function

        Returns
        -------
        None


        """


        self.Dim = Dim
        self.Dataset = Dataset
    
    def f(self, x: List[float], mu: List[float]) -> float:

        """
        The probability density function of the Multidimensional Standard Gaussian Distribution

        Arguments
        ---------
        x: the random variable
        mu: the mean value of the standard gaussian distribution

        Returns
        -------
        res: the probability density value of the random variable x


        """


        res = 1
        for i in  range(0, self.Dim):
            res *= (np.exp(-((x[i]-mu[i])**2)/2))/(np.sqrt(2*np.pi))
        return res

    def L(self, mu: List[float]) -> float:

        """
        
        The Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution

        Retruns
        -------
        res: the value of the likelihood function


        """


        res = 1
        for i in range(len(self.Dataset)):
            res *= self.f(self.Dataset[i], mu)
        return res
    
    def dLdmu(self, mu: List[float]) -> List[float]:

        """

        The derivative of the Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution

        Retruns
        -------
        res: the value of the derivative of the likelihood function


        """


        res = self.L(mu)
        Res = []
        for i in range(len(mu)):
            subres = res
            for k in range(len(self.Dataset)):
                subres *= (self.Dataset[k][i]-mu[i])
            Res.append(subres)
        return Res