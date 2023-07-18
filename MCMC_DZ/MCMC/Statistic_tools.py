import numpy as np
import scipy.special as spc

class FakeDataGen2D_Poisson:
    def __init__(self, pdf, Data_range, bins, scaler=1):
        """

        The fake data generator

        Arguments
        ---------
        pdf (Callable): the probability density function
        Data_range (List[List[float]]): The range of the Fake data
        bins (int): the number of the bins
        
        """

        self.pdf = pdf
        self.scaler = scaler
        self.Data_range  = Data_range
        self.bins = bins
        # Generate a grid of x and y value
        x = np.linspace(Data_range[0][0], Data_range[0][1], bins)
        y = np.linspace(Data_range[1][0], Data_range[1][1], bins)
        self.x, self.y  = np.meshgrid(x, y)
        self.BinsValue = np.floor(self.pdf(np.column_stack((self.x.flatten(), self.y.flatten())))*((self.Data_range[0][1]-self.Data_range[0][0])*(self.Data_range[1][1]-self.Data_range[1][0])/(self.bins**2))*self.scaler)
        self.BinsValue = np.asanyarray([np.random.poisson(datai) for datai in self.BinsValue])

class LikeliFuncGen:
    def __init__(self, Data, pdf, Prior=None, shift=0):

        """

        The Likelihood Function Generator

        Arguments
        ---------

        Data : the data set for the pdf function function
     
        Returns
        -------
        None

        """


        self.Data = Data
        self.pdf = pdf
        self.Prior = Prior
        self.shift = shift
        if (self.Prior==None):
            def constantPrior(params):
                return 1
            self.Prior = constantPrior

    def Likelihood(self, params):
        """
        
        The Likelihood Function

        Arguments
        ---------
        params : the parameter values of the probability denstiy function

        Returns
        -------
        (float)
        
        """

        self.pdf.params = params
        self.lambda_theta = np.floor(self.pdf.f(np.column_stack((self.Data.x.flatten(), self.Data.y.flatten())))*((self.Data.Data_range[0][1]-self.Data.Data_range[0][0])*(self.Data.Data_range[1][1]-self.Data.Data_range[1][0])/(self.Data.bins**2))*self.Data.scaler)
        # range = np.logical_and(self.lambda_theta>0, self.Data.BinsValue>0)
        # likeli = np.zeros(self.lambda_theta.shape[0])
        # likeli[range] = (np.power(self.lambda_theta[range], self.Data.BinsValue[range]) * np.exp(-self.lambda_theta[range])/ spc.factorial(self.Data.BinsValue[range]))
        likeli = (np.power(self.lambda_theta, self.Data.BinsValue) * np.exp(-self.lambda_theta)/ spc.factorial(self.Data.BinsValue))
        return np.sum(likeli)
    
    def Posterior(self, params):

        """
        
        The Posterior Function

        Arguments
        ---------
        params : the parameter values of the probability denstiy function

        Returns
        -------
        (float)
        
        """
        
        return self.Likelihood(params)*self.Prior(params) + self.shift