from typing import List, Callable
import numpy as np

class UniformProposalDistribution:
    def __init__(
        self, 
        radius: float,
        Dim: int,
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
        self.name = "UniformProposalDistribution"
        self.informations = {"PD_radius": radius}
    
    def pdf(self, theta_1: (np.array), theta_0: (np.array)) -> float:

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

    def log_pdf(self, theta_1: (np.array), theta_0: (np.array)) -> float:

        """

        Generating the log probability density function value of the proposal distribution

        Arguments
        ---------
        theta_nPlus1: the proposal value of the parameter
        theta_n: the current value of the parameter

        Returns
        -------
        p: The probability density function value of the proposal distribution


        """


        self.p = (1/(2*self.radius))**self.Dim
        return np.log(self.p)
    
    def sampling(self, theta: (np.array)) -> (np.array):

        """
        Drawing the proposal value of the parameter from the proposal distribution with given the current value of the parameter

        Arguments
        ---------
        theta_n: the current value of the parameter

        Returns
        -------
        theta_nPlus1: The proposal value of the parameter
        """


        return np.random.uniform(low=(theta-self.radius), high=(theta+self.radius))



class GaussianProposalDistribution:
    def __init__(
        self,
        sd: np.array,
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
        self.sd = np.asarray(sd)
        self.Dim = Dim
        self.name = "GaussianProposalDistribution"
        self.informations = {"PD_sd": sd}

    def pdf(self, theta_1: (np.array), theta_0: (np.array)) -> float:

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


        self.p = np.exp(-(((theta_0-theta_1)/self.sd)**2)/2) / (self.sd * np.sqrt(2*np.pi))
        return np.prod(self.p)

    def log_pdf(self, theta_1: (np.array), theta_0: (np.array)) -> float:

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


        self.p = np.exp(-(((theta_0-theta_1)/self.sd)**2)/2)/(self.sd*np.sqrt(2*np.pi))
        return np.sum(np.log(self.p))
    
    def sampling(self, theta: (np.array)) -> (np.array):

        """
        Drawing the proposal value of the parameter from the proposal distribution with given the current value of the parameter

        Arguments
        ---------
        theta_n: the current value of the parameter

        Returns
        -------
        theta_nPlus1: The proposal value of the parameter
        """


        return np.random.normal(loc=theta, scale=self.sd)


class HamiltonianProposalFunction:

    def __init__(
            self,
            rho: Callable,
            epsilon: float,
            L: int,
            log_likelihood = False
    ):
        self.rho = rho
        self.epsilon = epsilon
        self.L = L
        self.log_likelihood = log_likelihood
        self.name = "HamiltonianProposalFunction"
        self.informations = {"PF_epsilon": epsilon, "PF_L": L}
    
    def U(
        self,
        theta
    ) -> float:

        """
        The potential energy function: U(theta)
        This should be inputed as the target distribution to the MHMCMC
        And you should set the log_likelihood=True

        Arguments
        ---------
        theta: the inputted value of the parameter theta

        Returns
        -------
        The potential energy: -log(probability distribution of theta)


        """

        if self.log_likelihood:
            return self.rho(theta)
        else:
            return np.log(self.rho(theta))

    def dU_dtheta(
        self,
        theta: (np.array)
    ):

        """
        
        The derivative of the potential energy function

        Arguments
        ---------
        theta: the inputted value of the parameter theta

        Returns
        -------
        The derivative of the potential energy respect to theta: d(-log(probability distribution of theta))/d(theta)


        """

        Res = np.array([])

        for i in range(theta.shape[0]):
            theta_pos = theta.copy()
            theta_neg = theta.copy()

            theta_pos[i] += self.epsilon
            theta_neg[i] -= self.epsilon
            Resi = (-self.U(theta_pos)+self.U(theta_neg))/(2*self.epsilon)
            if (Resi!=0 and Resi!=np.nan):
                Res = np.append(Res, Resi)
            else:
                Res = np.append(Res, 0)
        return Res
    
    def log_pdf(self, theta_1: (np.array), theta_0: (np.array)) -> float:

        """
        
        The kinetic energy function

        Arguments
        ---------
        theta: the inputted value of the parameter theta

        Returns
        -------
        The kinetic energy respect to theta


        """
    
        return -(np.sum((self.p1 if (theta_1==self.theta1).all() else self.p0) ** 2))/2


    def sampling(self, theta: np.array) -> (np.array):

        """
        Drawing the proposal value of the parameter from the proposal distribution with given the current value of the parameter

        Arguments
        ---------
        theta_n: the current value of the parameter

        Returns
        -------
        theta_nPlus1: The proposal value of the parameter


        """

        # the ”positions” which are independent standard normal variables
        self.p1 = np.random.default_rng().normal(0,1,theta.shape[0])
        self.theta1 = theta
        self.p0 = self.p1
        # At the beginning, take a half step for momentum.
        self.p1 = self.p1 - (self.epsilon*self.dU_dtheta(self.theta1)/2)
        for i in range(0, self.L):
            # Take a full step for the position
            self.theta1 = self.theta1 + self.epsilon*self.p1
            # Unless at the end of the trajectory, take a full step for the momentum
            self.p1 = (self.p1 - self.epsilon*self.dU_dtheta(self.theta1)) if (i<(self.L-1)) else self.p1
        # At the end, take a half step for momentum.
        self.p1 = self.p1 - (self.epsilon*self.dU_dtheta(self.theta1)/2)
        # To make the proposal symmetric, negate momentum at end of trajectory
        self.p1 = -self.p1
        return self.theta1
