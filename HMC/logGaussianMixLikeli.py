import numpy as np
from typing import List

class log_Likelihood_GaussianMixture:
    def __init__(self, Dim:int, nComp:int, weights: List[float], Sigma: List[List[float]], Dataset: List[List[float]]) -> None:

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
        self.weights = weights
        self.nComp = nComp
        self.Sigma = np.array(Sigma)
        self.detSigma = np.linalg.det(self.Sigma)
        assert ((self.Sigma==np.transpose(self.Sigma)).all())and(self.detSigma!=0), "The Sigma(Variance-Covariance Matrix) should be non-singular and symmetric"
        self.invSigma = np.linalg.inv(self.Sigma)
        self.Dataset = Dataset
    
    def f(self, x: List[float], mu: List[float]) -> float:

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


        x = np.array(x)
        mu = np.array(mu)
        means = np.reshape(mu[:self.Dim*self.nComp], (self.Dim, self.nComp))
        for j in range(self.nComp):
            res += self.weights[j]*((2*np.pi)**(-self.Dim/2))*(self.detSigma**(-1/2))*np.exp(-((x-means[j])@self.invSigma@(x-means[j]).T)/2)
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
            res += np.log(self.f(self.Dataset[i], mu))

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
        
        for i in range(0, len(self.Dataset)):
            res = 0
            for k in range(self.nComp):
                res += (self.weights[k]*(self.f(self.Dataset[i],mu))) / (self.f(self.Dataset[i],mu)*[self.weights[j] for j in range(self.nComp)])*(self.invSigma@(np.array(self.Dataset[i])-np.array(mu)))
            Res += res

        return list(Res)