from typing import List
import numpy as np

# Uniform Proposal distribution
class UniformProposalDistribution:
    def __init__(
        self, 
        radius: float,
        Dim: float
    ) -> None:
        self.radius = radius
        self.Dim = Dim
    
    def qProb(self, theta_nPlus1: List[float], theta_n: List[float]) -> float:
        return (1/(2*self.radius))**self.Dim
    
    def qSample(self, theta: List[float]) -> List[float]:
        return [np.random.uniform(low=t-self.radius, high=t+self.radius, size=1)[0] for t in theta]

# Gaussian Proposal distribution

class GausianProposalDistribution:
    def __init__(
        self,
        sd: float, # Standard deviation of the proposal distribution
        Dim: float
    ) -> None:
        self.sd = sd
        self.Dim = Dim

    def qProb(self, theta_nPlus1: List[float], theta_n: List[float]) -> float:
        res = 1
        for i in range(0, self.Dim):
            res *= np.exp(-(((theta_n[i]-theta_nPlus1[i])/self.sd)**2)/2)/(self.sd*np.sqrt(2*np.pi))
        return res
    
    def qSample(self, theta: List[float]) -> List[float]:
        return [np.random.normal(loc=t, scale=self.sd, size=1)[0] for t in theta]
