import numpy as np
from scipy.stats import multivariate_normal

class Likeli_GMM_mus:
    
    def __init__(self, Dataset, Sigmas, weights, n_components):

        """

        initialize the Gaussian Mixture Model Likelihood Function with mus(means) as the variable

        Arguments
        ---------
        Dataset (np.array): the inputed dataset of the likelihood function
        Sigmas (np.array): the variance covariance matrix of each gaussian model
        weightes (np.array); the weight of each gaussian model
        n_components (int): the number of components of the Gaussian Mixture Model

        Returns
        -------
        None

        """

        self.Dataset = Dataset
        self.Sigmas = Sigmas
        self.weights = weights
        self.n_components = n_components
        self.n_dimensions = self.Dataset.shape[1]

    def calculate_log_likelihoods_mu(self, params):
        '''

        Calculating the log likelihood value

        Arguments
        params [list]: flattened 1D list containing the mean for each Gaussian component
                       in shape of [mean11, ..., mean1d, mean21, ..., meanKd]
                       where K is the number of components, and D is the number of dimensions.
        
        Returns
        log_likelihoods [np.array]: log-likelihood from parameters

        '''
        n_dims = self.n_dimensions
        n_comp = self.n_components
        
        # reshape the 1D parameter list into mean and covariance matrices for each component
        means = np.reshape(params[:n_dims*n_comp], (n_comp, n_dims))
            
        # initialize a numpy array of zeros to store log-likelihoods of each sample
        likelihoods = self.weights[0] * multivariate_normal.pdf(self.noisy_data, mean=means[0], cov=self.Sigmas[0])
        for j in range(1, n_comp):
            # calculate log-likelihood of each sample by
            # adding product of mixture coefficient and pdf at sample x_j for each Gaussian component k
            likelihoods += self.weights[j] * multivariate_normal.pdf(self.noisy_data, mean=means[j], cov=self.Sigmas[j])
        log_likelihoods = np.log(likelihoods) # store in log_likelihoods numpy array
        
        return log_likelihoods.sum() # return total log-likelihood of each sample given the parameters