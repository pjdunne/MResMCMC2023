from typing import List, Callable
import numpy as np
class HMC:
    def __init__(
        self,
        rho: Callable, # the target distribution of the parameter
        drho_dtheta: Callable, # the derivative of the target distribution
    ) -> None:
        self.rho = rho
        self.drho_dtheta = drho_dtheta

    def U(
        self,
        theta: List[float]
    ) -> float: # the potential energy function: U(theta) = -log(probability distribution of theta)
        return (- np.log(self.rho(theta)))
    
    def dU_dtheta(
        self,
        theta: List[float]
    ) -> List[float]: # the derivative of the potential energy function
        self.DRho = self.drho_dtheta(theta)
        self.Rho = self.rho(theta)*(-1)
        return np.array([self.DR/self.Rho for self.DR in self.DRho])
    def Testing(self, theta):
        return self.U(theta), self.dU_dtheta(theta)

    def predict(
        self,
        theta0: List[float], # initial value of the parameter
        lr: float, # learning rate
        L: int, # the number of leapfrog steps in the trajectory
        epoch: int # trian the MCMC for n epoch
    ):
        theta_n = np.array(theta0)        
        for _ in range(0, epoch):
            # the ”positions” which are independent standard normal variables
            p_nPlus1 = np.random.default_rng().normal(0, 1, len(theta_n))
            theta_nPlus1 = theta_n
            p_n = p_nPlus1
            # At the beginning, take a half step for momentum.
            p_nPlus1 = p_nPlus1 - (lr*self.dU_dtheta(theta_nPlus1)/2)
            for i in range(0,L):
                # Take a full step for the position
                theta_nPlus1 = theta_nPlus1 + lr*p_nPlus1
                # Unless at the end of the trajectory, take a full step for the momentum
                p_nPlus1 = (p_nPlus1 - lr*self.dU_dtheta(theta_nPlus1)) if (i<(L-1)) else p_nPlus1
            # At the end, take a half step for momentum.
            p_nPlus1 = p_nPlus1 - (lr*self.dU_dtheta(theta_nPlus1)/2)
            # To make the proposal symmetric, negate momentum at end of trajectory
            p_nPlus1 = -p_nPlus1
            # At start and end of trajectory, evaluate potential and kinetic energies
            U_n = self.U(theta_n)
            K_n = 0
            for pValue in p_n:
                K_n += pValue**2
            K_n /= 2
            U_nPlus1 = self.U(theta_nPlus1)
            K_nPlus1= 0
            for pValue in p_nPlus1:
                K_nPlus1 += pValue**2
            K_nPlus1 /= 2
            # At end of trajectory deciding whether accept or reject the state , returning either the position at the end of the trajectory or the initial position
            alpha = np.exp((U_n-U_nPlus1)+(K_n-K_nPlus1))
            u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(alpha>u):
                theta_n = theta_nPlus1
            else:
                pass
        return list(theta_n)