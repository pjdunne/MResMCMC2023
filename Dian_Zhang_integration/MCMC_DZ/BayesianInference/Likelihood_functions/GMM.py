from typing import List
import numpy as np

class Gaussian_PDF:
    def __init__(self, params: np.array, Sigma: np.array):

        """

        The Gaussian Probability Density Function

        Arguments
        ---------
     
        Returns
        -------
        None


        """

        self.params = params
        self.Sigma = Sigma
        self.sqrt_det_Sigma = np.sqrt(np.linalg.det(self.Sigma))
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def f(self, X: np.array):

        """
        
        The main functon of the Gaussian distribution

        Arguments
        ---------
        X (np.array): The inputed value of the random variable of the Gaussian Distribution

        Returns
        -------
        (np.array): The probability of the Gaussian Distribution with given parameter value respected to the inputed value X
        
        
        """

        k  = self.params.shape[0]
        if (len(X.shape)==1):
                return (np.power(2*np.pi, -k/2) * self.sqrt_det_Sigma * (np.exp(-0.5*((X-self.params) @ self.Sigma_inv) @ (X-self.params))))
        else:
            return np.power(2*np.pi, -k/2) * self.sqrt_det_Sigma * (np.exp(-0.5*np.sum(((X - self.params) @ self.Sigma_inv) * (X - self.params), axis=1)))


class Gaussian_Mixture_Model:
    def __init__(self, params: np.array, Sigmas: np.array):
        self.params = params
        Sigmas = np.asarray(Sigmas)
        self.model_number = Sigmas.shape[0]
        self.Gaussian_PDFs = []
        for i in range(self.model_number):
            self.Gaussian_PDFs.append(Gaussian_PDF(self.params[self.model_number+i*2:self.model_number+(i+1)*2], Sigmas[i]))

    def f(self, X: np.array):
        weights = self.params[:self.model_number]
        weights = np.exp(weights)
        weights /= np.sum(weights)
        self.Gaussian_PDFs[0].params = self.params[self.model_number: self.model_number+2]
        Res = weights[0] * self.Gaussian_PDFs[0].f(X)

        for i in range(1, self.model_number):
            self.Gaussian_PDFs[i].params = self.params[self.model_number+i*2: self.model_number+(i+1)*2]
            Res += weights[i] * self.Gaussian_PDFs[i].f(X)
        
        return Res