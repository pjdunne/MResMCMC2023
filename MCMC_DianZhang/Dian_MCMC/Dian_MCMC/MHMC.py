from typing import Callable
import numpy as np

class MHMC:  
    def __init__(
    self, 
    rho: Callable,
    seed=123
    ) -> None:
    
        """
        
        Metropolis Hasting Monte Carlo Algorithm
        
        Arguments
        ----------
        rho (Callable): the target distribution of the parameter
        seed (int): the random seed of the distribution
        
        Returns
        -------
        None


        """
    
        # Setting the random seed of the numpy
        np.random.seed(seed)
        # Saving the inputted target distribution
        self.rho = rho

    def generate(
        self,
        theta0,
        qProb: Callable,
        qSamp: Callable,
        steps: int,
        OutputAcceptanceRate = True
    ):

        """

        Generating the values of the parameter from the target distribution

        Arguments
        ----------
        theta0: the initial value of the parameter
        qProb (Callable): probability of the proposal distribution of the parameter
        qSamp (Callable): draw the sample with the proposal distribution
        steps (int): run the MCMC for n steps
        OutputAcceptanceRate (bool): deciding whether to output the acceptance rate of the Metropolis-Hasting Algorithm

        Returns
        -------
        Theta: the values of the parameter accepted by the Metropolis-Hasting Monte Carlo Algorithm


        """

        self.theta_n = theta0
        if OutputAcceptanceRate:
            self.acceptanceRate = 0
        self.Theta = []
        for _ in range(0, steps):
            self.Theta.append(list(self.theta_n))
            # Updating the parameter from the proposal disribution
            self.theta_nPlus1 = qSamp(self.theta_n)
            self.alpha = min(1, (self.rho(self.theta_nPlus1)*qProb(self.theta_nPlus1,self.theta_n))/(self.rho(self.theta_n)*qProb(self.theta_n,self.theta_nPlus1)))
            # Deciding whether to reject the update of the parameter
            self.u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(self.alpha>=self.u):
                self.theta_n = self.theta_nPlus1
                if OutputAcceptanceRate:
                    self.acceptanceRate += 1
            else:
                pass
        self.Theta.append(list(self.theta_n))
        if OutputAcceptanceRate:
            return self.Theta, (self.acceptanceRate/steps)
        else:
            return self.Theta
