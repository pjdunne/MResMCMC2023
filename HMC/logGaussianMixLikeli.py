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
        self.pi = pi
        self.Sigma = np.array(Sigma)
        self.detSigma = np.linalg.det(self.Sigma)
        assert ((self.Sigma==np.transpose(self.Sigma)).all())and(self.detSigma!=0), "The Sigma(Variance-Covariance Matrix) should be non-singular and symmetric"
        self.invSigma = np.linalg.inv(self.Sigma)
        self.Dataset = Dataset
    
    def logf(self, x: List[float], mu: List[float], pi: list[float]) -> float:

        """
        The log joint probability density function of the Gaussian Mixture

        Arguments
        ---------
        x: the random variable
        mu: the mean value of the standard gaussian distribution
        pi: the mixing coefficient

        Returns
        -------
        res: the log probability density value of the random variable x


        """


        self.x = np.array(x)
        self.mu = np.array(mu)
        self.pi = np.array(pi)
        res = sum(np.log(pi)*(-(1/2))*(self.Dim*np.log(2*np.pi) + np.log(self.detSigma) + ((self.x-self.mu)@self.invSigma@(self.x-self.mu).T)))
        return res

    def logL(self, mu: List[float], pi: List[float]) -> float:

        """
        
        The log Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution
        pi: the mixing coefficient

        Retruns
        -------
        res: the value of the log likelihood function


        """


        res = 0
        for i in range(len(self.Dataset)):
            res += self.logf(self.Dataset[i], mu, pi)

        return res

    def dlogLdmu(self, mu: List[float]) -> List[float]:

        """

        The derivative of the log Likelihood function

        Arguments
        ---------
        mu: the mean value of Multidimensional Gaussian Distribution
        pi: the mixing coefficient

        Retruns
        -------
        res: the value of the derivative of the log likelihood function


        """

        Res = sum(self.invSigma@(np.array(self.Dataset[0])-np.array(mu)))
        
        for k in range(1, len(self.Dataset)):
            Res += sum(self.invSigma@(np.array(self.Dataset[k])-np.array(mu)))

        return list(Res)