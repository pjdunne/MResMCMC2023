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
        ProposalFunction,
        steps: int,
        MaxTime = 0,
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
        MaxTime (int): the max time taken by the algorithm
        OutputAcceptanceRate (bool): deciding whether to output the acceptance rate of the Metropolis-Hasting Algorithm
        OutputRunTime (int): the Run time taken to finish n*OutputRunTime steps

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
        Thetas = np.array([theta_0], dtype=np.float64)
        for s in range(0, steps):
            # Updating the parameter from the proposal disribution
            theta_1 = ProposalFunction.sampling(theta_0)
            rho_theta_0 = self.rho(theta_0)
            rho_theta_1 = self.rho(theta_1)
            if (rho_theta_0!=0 and rho_theta_0!=np.nan):
                if (rho_theta_1!=0 and rho_theta_1!=np.nan):
                    if self.log_likelihood:
                        log_alpha = min(0, (rho_theta_1 + ProposalFunction.log_pdf(theta_1, theta_0) - rho_theta_0 - ProposalFunction.log_pdf(theta_0, theta_1)))
                        alpha = np.exp(log_alpha)
                    else:
                        alpha = min(1, (rho_theta_1*ProposalFunction.pdf(theta_1, theta_0))/(rho_theta_0*ProposalFunction.pdf(theta_0, theta_1)))
                else:
                    alpha = 0
            elif (rho_theta_1!=0 and rho_theta_1!=np.nan):
                alpha = 1
            else:
                alpha = 0.5

            # Deciding whether to reject the update of the parameter
            u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(alpha>=u):
                theta_0 = theta_1
                if OutputAcceptanceRate:
                    acceptanceRate += 1
            Thetas = np.append(Thetas, np.array([theta_0]), axis=0)
            if OutputRunTime:
                if ((s+1)%OutputRunTime)==0:
                    RunTime.append(time.perf_counter() - start_time)
            if MaxTime:
                if ((time.perf_counter()-start_time)>MaxTime):
                    steps = s+1
                    break
        Res = {}
        Res["Thetas"] = Thetas
        Res["ProposalFunction"] = ProposalFunction.name
        for key in ProposalFunction.informations:
            Res[key] = ProposalFunction.informations[key]
        if OutputAcceptanceRate:
            Res["Acceptance_Rate"] = acceptanceRate/steps
        if OutputRunTime:
            Res["Run_Times"]  = RunTime
        return Res