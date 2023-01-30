import numpy as np
from typing import List

class Likelihood_Gaussian:
    def __init__(self, Dim:int, Dataset: List[List[float]]) -> None:
        self.Dim = Dim
        self.Dataset = Dataset
    
    def f(self, x: List[float], mu: List[float]) -> float:
        res = 1
        for i in  range(0, self.Dim):
            res *= (np.exp(-((x[i]-mu[i])**2)/2))/(np.sqrt(2*np.pi))
        return res

    def L(self, mu: List[float]) -> float:
        res = 1
        for i in range(len(self.Dataset)):
            res *= self.f(self.Dataset[i], mu)
        return res
    
    def dLdmu(self, mu: List[float]) -> List[float]:
        res = self.L(mu)
        Res = []
        for i in range(len(mu)):
            subres = res
            for k in range(len(self.Dataset)):
                subres *= (self.Dataset[k][i]-mu[i])
            Res.append(subres)
        return Res