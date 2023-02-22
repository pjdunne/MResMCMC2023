import numpy as np
from typing import List

class log_Likelihood_GaussianMixture:
    def __init__(self, Dim:int, pi:List[List[float]], Sigma: List[List[float]], Dataset: List[List[float]]) -> None:

        """
        
        The likelihood function of the Gaussian Mixture

        Arguments
        ---------
        Dim (int): the dimension of the inputted parameter
        pi (List[List[float]]): the mixing coefficient
        Sigma (List[List[float]]): variance covariance matrix of the Gaussian Likelihood function
        Dataset (List[List[float]]): the dataset used to generate the Likelihood function

        Returns
        -------
        None


        """

        self.Dim = Dim
        self.pi = np.array(pi)
        self.Sigma = np.array(Sigma)
        self.detSigma = np.linalg.det(self.Sigma)
        assert ((self.Sigma==np.transpose(self.Sigma)).all())and(self.detSigma!=0), "The Sigma(Variance-Covariance Matrix) should be non-singular and symmetric"
        self.invSigma = np.linalg.inv(self.Sigma)
        self.Dataset = Dataset
    
    def logf(self, x: List[float], mu: List[float]) -> float:

        """
        The log joint probability density function of the Gaussian Mixture

        Arguments
        ---------
        x: the random variable
        mu: the mean value of the standard gaussian distribution

        Returns
        -------
        res: the log probability density value of the random variable x


        """


        self.x = np.array(x)
        self.mu = np.array(mu)
        res = sum(np.log(self.pi)*(-(1/2))*(self.Dim*np.log(2*np.pi) + np.log(self.detSigma) + ((self.x-self.mu)@self.invSigma@(self.x-self.mu).T)))
        return res

    def f(self, x: List[float], mu: List[float]) -> float:

        """
        The probability density function of the Gaussian Mixture

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
        res = self.pi*((2*np.pi)**(-self.Dim/2))*(self.detSigma**(-1/2))*np.exp(-((self.x-self.mu)@self.invSigma@(self.x-self.mu).T)/2)
        return res

    def logL(self, mu: List[float]) -> float:

        """
        
        The log Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution

        Retruns
        -------
        res: the value of the log likelihood function


        """


        res = 0
        for i in range(len(self.Dataset)):
            res += self.logf(self.Dataset[i], mu)

        return res

    def dlogLdmu(self, mu: List[float]) -> List[float]:

        """

        The derivative of the log Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution

        Retruns
        -------
        res: the value of the derivative of the log likelihood function


        """

        Res = 0
        
        for k in range(0, len(self.Dataset)):
            Res += ((self.f(self.Dataset[k],mu)) / (sum(self.f(self.Dataset,mu))))*(self.invSigma@(np.array(self.Dataset[k])-np.array(mu)))

        return list(Res)