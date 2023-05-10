from typing import Callable
import numpy as np
import time

class MHMC:  
    def __init__(
    self, 
    rho: Callable,
    log_likelihood=False,
    seed=123
    ) -> None:
    
        """
        
        Metropolis Hasting Monte Carlo Algorithm
        
        Arguments
        ----------
        rho (Callable): the target distribution of the parameter
        log_likelihood (bool): whether the inputed target distribution is log likelihood function
        seed (int): the random seed of the distribution
        
        Returns
        -------
        None


        """
    
        # Setting the random seed of the numpy
        np.random.seed(seed)

        self.log_likelihood = log_likelihood

        # Saving the inputted target distribution
        self.rho = rho

    def generate(
        self,
        theta0,
        qProb: Callable,
        qSamp: Callable,
        steps: int,
        OutputAcceptanceRate = True,
        OutputRunTime = 0
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

        theta_0 = np.asarray(theta0)
        if OutputRunTime>=steps:
            OutputRunTime = steps
        if OutputAcceptanceRate:
            acceptanceRate = 0
        if OutputRunTime:
            RunTime = []
            start_time = time.perf_counter()
        Thetas = np.array([theta_0])
        for s in range(0, steps):
            # Updating the parameter from the proposal disribution
            theta_1 = qSamp(theta_0)
            if self.log_likelihood:
                alpha = min(1, np.exp(self.rho(theta_1) + (qProb(theta_1, theta_0)) - (self.rho(theta_0) + (qProb(theta_0, theta_1)))))
            else:
                alpha = min(1, (self.rho(theta_1)*qProb(theta_1, theta_0))/(self.rho(theta_0)*qProb(theta_0, theta_1)))

            # Deciding whether to reject the update of the parameter
            u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(alpha>=u):
                theta_0 = theta_1
                if OutputAcceptanceRate:
                    acceptanceRate += 1
            if OutputRunTime:
                if (s%OutputRunTime)==0:
                    RunTime.append(time.perf_counter() - start_time)
            Thetas = np.append(Thetas, np.array([theta_0]), axis=0)
        Res = {}
        Res["Thetas"] = Thetas
        if OutputAcceptanceRate:
            Res["Acceptance_Rate"] = acceptanceRate/steps
        if OutputRunTime:
            Res["Run_Times"]  = RunTime
        return Res