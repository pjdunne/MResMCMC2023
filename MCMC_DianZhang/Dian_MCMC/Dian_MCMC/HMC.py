from typing import List, Callable
import numpy as np
class HMC:
    def __init__(
        self,
        rho: Callable,
        drho_dtheta: Callable,
        seed=123
    ) -> None:

        """
        
        Hamiltonian Monte Carlo Algorithm

        Arguments
        ---------
        rho: the target distribution of the parameter
        drho_dtheta: the derivative of the target distribution repect to the parameter
        seed: the random seed of the distribution

        Returns
        -------
        None

        """

        # Setting the random seed of the numpy
        np.random.seed(seed)
        # Saving the inputted target distribution
        self.rho = rho
        self.drho_dtheta = drho_dtheta

    def U(
        self,
        theta: List[float]
    ) -> float:

        """
        The potential energy function: U(theta)

        Arguments
        ---------
        theta: the inputted value of the parameter theta

        Returns
        -------
        The potential energy: -log(probability distribution of theta)


        """


        return (- np.log(self.rho(theta)))
    
    def dU_dtheta(
        self,
        theta: List[float]
    ) -> List[float]:

        """
        
        The derivative of the potential energy function

        Arguments
        ---------
        theta: the inputted value of the parameter theta

        Returns
        -------
        The derivative of the potential energy respect to theta: d(-log(probability distribution of theta))/d(theta)


        """


        self.DRho = self.drho_dtheta(theta)
        self.Rho = self.rho(theta)*(-1)
        return np.array([self.DR/self.Rho for self.DR in self.DRho])

    def predict(
        self,
        theta0: List[float],
        epsilon: float,
        L: int,
        steps: int,
        OutputAcceptanceRate = True
    ):

        """
        Generating the values of the parameter from the target distribution
        
        Arguments
        ----------
        theta0: the initial value of the parameter
        epsilon: stepwith of each updates of the proposal value of the parameter
        L: the number of leapfrog steps in the trajectory
        steps: run the MCMC for n steps
        OutputAcceptanceRate: deciding whether to output the acceptance rate of the Metropolis-Hasting Algorithm
        
        Returns
        -------
        Theta: the values of the parameter accepted by the Hamiltonian Monte Carlo Algorithm
        
        """


        self.theta_n = np.array(theta0)
        if OutputAcceptanceRate:
            self.acceptanceRate = 0
        self.Theta = []     
        for _ in range(0, steps):
            self.Theta.append(list(self.theta_n))
            # the ”positions” which are independent standard normal variables
            self.p_nPlus1 = np.random.default_rng().normal(0, 1, len(self.theta_n))
            self.theta_nPlus1 = self.theta_n
            self.p_n = self.p_nPlus1
            # At the beginning, take a half step for momentum.
            self.p_nPlus1 = self.p_nPlus1 - (epsilon*self.dU_dtheta(self.theta_nPlus1)/2)
            for i in range(0,L):
                # Take a full step for the position
                self.theta_nPlus1 = self.theta_nPlus1 + epsilon*self.p_nPlus1
                # Unless at the end of the trajectory, take a full step for the momentum
                self.p_nPlus1 = (self.p_nPlus1 - epsilon*self.dU_dtheta(self.theta_nPlus1)) if (i<(L-1)) else self.p_nPlus1
            # At the end, take a half step for momentum.
            self.p_nPlus1 = self.p_nPlus1 - (epsilon*self.dU_dtheta(self.theta_nPlus1)/2)
            # To make the proposal symmetric, negate momentum at end of trajectory
            self.p_nPlus1 = -self.p_nPlus1
            # At start and end of trajectory, evaluate potential and kinetic energies
            self.U_n = self.U(self.theta_n)
            self.K_n = 0
            for self.pValue in self.p_n:
                self.K_n += self.pValue**2
            self.K_n /= 2
            self.U_nPlus1 = self.U(self.theta_nPlus1)
            self.K_nPlus1= 0
            for self.pValue in self.p_nPlus1:
                self.K_nPlus1 += self.pValue**2
            self.K_nPlus1 /= 2
            # At end of trajectory deciding whether accept or reject the state , returning either the position at the end of the trajectory or the initial position
            self.alpha = np.exp((self.U_n-self.U_nPlus1)+(self.K_n-self.K_nPlus1))
            self.u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(self.alpha>self.u):
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