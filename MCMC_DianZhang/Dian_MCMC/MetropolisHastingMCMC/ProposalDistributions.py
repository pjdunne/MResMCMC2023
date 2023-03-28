from typing import List
import numpy as np

class UniformProposalDistribution:
    def __init__(
        self, 
        radius: float,
        Dim: float,
        seed=123
    ) -> None:

        """
        
        The Uniform Proposal distribution

        Argumrnts
        ---------
        radius: the radius of the uniform distribution at each dimension
        Dim: the total dimension of the proposal distribution
        seed: the random seed of the distribution

        Returns
        -------
        None

        
        """


        # Setting the random seed of the numpy
        np.random.seed(seed)
        # Setting the parameters of the proposal distribution
        self.radius = radius
        self.Dim = Dim
    
    def qProb(self, theta_nPlus1: List[float], theta_n: List[float]) -> float:

        """

        Generating the probability density function value of the proposal distribution

        Arguments
        ---------
        theta_nPlus1: the proposal value of the parameter
        theta_n: the current value of the parameter

        Returns
        -------
        p: The probability density function value of the proposal distribution


        """


        self.p = (1/(2*self.radius))**self.Dim
        return self.p
    
    def qSample(self, theta: List[float]) -> List[float]:

        """
        Drawing the proposal value of the parameter from the proposal distribution with given the current value of the parameter

        Arguments
        ---------
        theta_n: the current value of the parameter

        Returns
        -------
        theta_nPlus1: The proposal value of the parameter
        """


        self.theta_nPlus1 = [np.random.uniform(low=t-self.radius, high=t+self.radius, size=1)[0] for t in theta]
        return self.theta_nPlus1



class GaussianProposalDistribution:
    def __init__(
        self,
        sd: float,
        Dim: float,
        seed=123
    ) -> None:

        """
        
        Gaussian Proposal distribution

        Arguments
        ---------
        sd: the standard deviation of the gaussian distribution at each dimension
        Dim: the total dimension of the proposal distribution
        seed: the random seed of the distribution
        """


        # Setting the random seed of the numpy
        np.random.seed(seed)
        # Setting the parameters of the proposal distribution
        self.sd = sd
        self.Dim = Dim

    def qProb(self, theta_nPlus1: List[float], theta_n: List[float]) -> float:

        """

        Generating the probability density function value of the proposal distribution

        Arguments
        ---------
        theta_nPlus1: the proposal value of the parameter
        theta_n: the current value of the parameter

        Returns
        -------
        p: The probability density function value of the proposal distribution


        """


        self.p = 1
        for i in range(0, self.Dim):
            self.p *= np.exp(-(((theta_n[i]-theta_nPlus1[i])/self.sd)**2)/2)/(self.sd*np.sqrt(2*np.pi))
        return self.p
    
    def qSample(self, theta: List[float]) -> List[float]:

        """
        Drawing the proposal value of the parameter from the proposal distribution with given the current value of the parameter

        Arguments
        ---------
        theta_n: the current value of the parameter

        Returns
        -------
        theta_nPlus1: The proposal value of the parameter
        """


        self.theta_nPlus1 = [np.random.normal(loc=t, scale=self.sd, size=1)[0] for t in theta]
        return self.theta_nPlus1
