import numpy as np
from typing import List

class Likelihood_MultidimensionalGaussian:
    def __init__(self, Dim:int, Sigma: List[List[float]], Dataset: List[List[float]]) -> None:

        """
        
        The likelihood function of the Multidimensional Gaussian distribution

        Arguments
        ---------
        Dim (int): the dimension of the inputted parameter
        Sigma (List[List[float]]): variance covariance matrix of the Gaussian Likelihood function
        Dataset (List[List[float]]): the dataset used to generate the Likelihood function

        Returns
        -------
        None


        """


        self.Dim = Dim
        self.Sigma = np.array(Sigma)
        self.detSigma = np.linalg.det(self.Sigma)
        assert ((self.Sigma==np.transpose(self.Sigma)).all())and(self.detSigma!=0), "The Sigma(Variance-Covariance Matrix) should be non-singular and symmetric"
        self.invSigma = np.linalg.inv(self.Sigma)
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


        self.x = np.array(x)
        self.mu = np.array(mu)
        res = ((2*np.pi)**(-self.Dim/2))*(self.detSigma**(-1/2))*np.exp(-((self.x-self.mu)@self.invSigma@(self.x-self.mu).T)/2)
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
        Res = self.invSigma@(np.array(self.Dataset[0])-np.array(mu))*res
        
        for k in range(1, len(self.Dataset)):
            Res = self.invSigma@(np.array(self.Dataset[k])-np.array(mu))*Res
        return list(Res)