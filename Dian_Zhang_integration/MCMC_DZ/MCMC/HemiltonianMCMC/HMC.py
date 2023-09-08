from typing import List, Callable
import numpy as np
class HMC:
    def __init__(
        self,
        rho: Callable,
        drho_dtheta=None,
        epsilon=1e-8,
        log_likelihood=False,
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
        self.epsilon = epsilon
        self.log_likelihood = log_likelihood

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


        if self.log_likelihood:
            return - self.rho(theta)
        else:
            return - np.log(self.rho(theta))
    
    def dU_dtheta(
        self,
        theta: (np.array)
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


        if self.log_likelihood or (self.drho_dtheta==None): 
            DU_dtheta = []
            for i in range(len(theta)):
                theta_pos = theta
                theta_neg = theta
                theta_pos[i] += self.epsilon
                theta_neg[i] -= self.epsilon
                DU_dtheta.append((self.U(theta_pos)-self.U(theta_neg))/2*self.epsilon)
            return np.asarray(DU_dtheta)
        else:
            DRho = self.drho_dtheta(theta)
            Rho = self.rho(theta)*(-1)
            return np.asarray([DR/Rho for DR in DRho])

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


        self.theta0 = np.asarray(theta0)
        if OutputAcceptanceRate:
            self.acceptanceRate = 0
        self.Theta = []     
        for _ in range(0, steps):
            self.Theta.append(list(self.theta0))
            # the ”positions” which are independent standard normal variables
            self.p1 = np.random.default_rng().normal(0, 1, len(self.theta_n))
            self.theta1 = self.theta0
            self.p0 = self.p1
            # At the beginning, take a half step for momentum.
            self.p1 = self.p1 - (epsilon*self.dU_dtheta(self.theta1)/2)
            for i in range(0,L):
                # Take a full step for the position
                self.theta1 = self.theta1 + epsilon*self.p1
                # Unless at the end of the trajectory, take a full step for the momentum
                self.p1 = (self.p1 - epsilon*self.dU_dtheta(self.theta_nPlus1)) if (i<(L-1)) else self.p1
            # At the end, take a half step for momentum.
            self.p1 = self.p1 - (epsilon*self.dU_dtheta(self.theta1)/2)
            # To make the proposal symmetric, negate momentum at end of trajectory
            self.p1 = -self.p1
            # At start and end of trajectory, evaluate potential and kinetic energies
            self.U0 = self.U(self.theta0)
            self.K0 = 0
            for self.pValue in self.p0:
                self.K0 += self.pValue**2
            self.K0 /= 2
            self.U1 = self.U(self.theta1)
            self.K1= 0
            for self.pValue in self.p1:
                self.K1 += self.pValue**2
            self.K1 /= 2
            # At end of trajectory deciding whether accept or reject the state , returning either the position at the end of the trajectory or the initial position
            self.alpha = np.exp((self.U0-self.U1)+(self.K0-self.K1))
            self.u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(self.alpha>self.u):
                self.theta0 = self.theta1
                if OutputAcceptanceRate:
                    self.acceptanceRate += 1
            else:
                pass
            self.Theta.append(list(self.theta0))
        if OutputAcceptanceRate:
            return self.Theta, (self.acceptanceRate/steps)
        else:
            return self.Theta
