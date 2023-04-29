import numpy as np

from scipy.stats import multivariate_normal

# import time
import cProfile
import pstats
from typing import List


class GaussianMixtureModel:
    '''
     This class implements a Gaussian Mixture Model which has several methods for 
     data generation, likelihood calculation, and visualization of the results.
     
    '''
    def __init__(self, n_dimensions, n_components, means, covs, weights):
        '''
        Initializes the class with the number of dimensions, number of components, 
        means, covariance matrices, and weights for each dimension.
        
        Arguments
        n_dimensions [int]: number of dimensions of dataset
        n_components [int]: number of components in the input data
        means [list]: means for each Gaussian component
        covs [list]: covariance matrices for each Gaussian component
        weights [list]: weights for each Gaussian component
        
        '''
        self.n_dimensions = n_dimensions
        self.n_components = n_components
        self.means = means
        self.covs = covs
        self.weights = weights
        # initialize the noisy_data attribute to None
        self.noisy_data = None

        
    def generate_data(self, n_samples, noise_scale=None):
        '''
        Generate data with Poisson flucuations
        
        Arguments
        X [np.ndarra]: initial dataset we generate
        noise_scale [float or None]: standard deviation of the Poisson noise added to the data
        
        Returns 
        None
        
        '''
        # t_start = time.process_time()

        X = np.zeros((n_samples, self.n_dimensions))
        rng = np.random.default_rng(12345)
        c = rng.choice(self.n_components, size=n_samples, p=self.weights)
        for i in range(self.n_components):
            idx = (c == i)
            X[idx, :] = rng.multivariate_normal(mean=self.means[i], cov=self.covs[i], size=np.sum(idx))

        # first create a numpy histogram from the GMM
        bins = 30
        data_hist, bin_edges = np.histogramdd(X, bins=bins)
        bin_centers = []
        for i in range(self.n_dimensions):
            bin_centers.append((bin_edges[i][:-1] + bin_edges[i][1:]) / 2.0)

        # add Poisson fluctuations to the value in each bin
        # it takes the expected values for the Poisson distribution 
        # and generates a new numpy array with Poisson fluctuations around these expected values
        noisy_data_hist = rng.poisson(lam=data_hist)

        noisy_data = np.array(np.meshgrid(*bin_centers)).T.reshape(-1,self.n_dimensions) # reshpae
        noisy_data = noisy_data.repeat(noisy_data_hist.flatten(), axis=0) # flattened
        noisy_data = noisy_data + rng.normal(scale=noise_scale, size=noisy_data.shape)
        self.noisy_data = noisy_data

        # t_stop = time.process_time()
        # print("Elapsed time generating data in seconds =", t_stop - t_start)

        return X, noisy_data

    def logpdf(self, x, mean, cov):
        '''
        the log pdf of multivariate normal

        x will need to contain more than one data point here
        '''
        x = np.array(x)
        # `eigh` assumes the matrix is Hermitian.
        vals, vecs = np.linalg.eigh(cov)
        logdet     = np.sum(np.log(vals))
        valsinv    = np.array([1./v for v in vals])
        # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
        # rank. The asterisk performs element-wise multiplication.
        U          = vecs * np.sqrt(valsinv)
        rank       = len(vals)
        dev        = x - mean
        # "maha" for "Mahalanobis distance".
        maha       = np.square(np.dot(dev, U)).sum(axis=1)
        log2pi     = np.log(2 * np.pi)
        return -0.5 * (rank * log2pi + maha + logdet)
    
    def pdf(self, x, mean, cov):
        '''
        calculating the pdf
        '''
        return np.exp(self.logpdf(x, mean, cov))


    def calculate_log_likelihoods_mu(self, params):
        '''
        Pass this function to MCMCs so that parmameters can be inputted.
        Calculates the total log-likelihood of the Gaussian Mixture Model for the given parameter.
        
            The likelihood is calculated as the product of individual pdfs at the observed samples:
            L(θ) = p(x_1, x_2, ..., x_N|θ) = Π_j=1^N p(x_j|θ)

            Since likelihood of each point may be very small, leading to extremely small products, 
            a log likelihood is often used in practice:
            logL(θ) = Σ_j=1^N log(p(x_j|θ)) 

        Arguments
        params [list]: flattened 1D list containing the mean and covariance for each Gaussian component
                       in shape of [mean1, mean2, ..., meanK, cov11, cov12, ..., cov1D, cov21, cov22, ..., covK1, ..., covKD]
                       where K is the number of components, and D is the number of dimensions.
        
        Returns
        log_likelihoods [np.array]: log-likelihood from parameters

        '''
        # t_start = time.process_time()

        n_dims = self.n_dimensions
        n_comp = self.n_components
        
        # reshape the 1D parameter list into mean and covariance matrices for each component
        means = np.reshape(params[:n_dims*n_comp], (n_comp, n_dims))
            
        # initialize a numpy array of zeros to store log-likelihoods of each sample
        log_likelihoods = 0 

        likelihood = 0
        for j in range(n_comp):
            # calculate log-likelihood of each sample by
            # adding product of mixture coefficient and pdf at sample x_j for each Gaussian component k
            likelihood += self.weights[j] * self.pdf(self.noisy_data, mean=means[j], cov=self.covs[j])
        log_likelihoods += np.log(likelihood) # store in log_likelihoods numpy array

        # t_stop = time.process_time()
        # print("Elapsed time calculating LLH in seconds =", t_stop - t_start)
        
        return np.sum(log_likelihoods) # return total log-likelihood of each sample given the parameters

    def logf_mu(self, params):
        '''
        Pass this function to MCMCs so that parmameters can be inputted.
        Calculates the total log-likelihood of the Gaussian Mixture Model for the given parameter.
        
            The likelihood is calculated as the product of individual pdfs at the observed samples:
            L(θ) = p(x_1, x_2, ..., x_N|θ) = Π_j=1^N p(x_j|θ)

            Since likelihood of each point may be very small, leading to extremely small products, 
            a log likelihood is often used in practice:
            logL(θ) = Σ_j=1^N log(p(x_j|θ)) 

        Arguments
        params [list]: flattened 1D list containing the mean and covariance for each Gaussian component
                       in shape of [mean1, mean2, ..., meanK, cov11, cov12, ..., cov1D, cov21, cov22, ..., covK1, ..., covKD]
                       where K is the number of components, and D is the number of dimensions.
        
        Returns
        log_likelihoods [np.array]: log-likelihood from parameters

        '''
        n_dims = self.n_dimensions
        n_comp = self.n_components
        
        # reshape the 1D parameter list into mean and covariance matrices for each component
        means = np.reshape(params[:n_dims*n_comp], (n_comp, n_dims))

        #change data type of the parameters using to calculate LLH

            
        # initialize a numpy array of zeros to store log-likelihoods of each sample
        log_likelihoods = 0 

        likelihood = 0
        for j in range(n_comp):
            # detSigma = np.linalg.det(self.covs[j])
            # invSigma = np.linalg.inv(self.covs[j])
            # # calculate log-likelihood of each sample by
            # # adding product of mixture coefficient and pdf at sample `self.noisy_data` for each Gaussian component j
            # likelihood += self.weights[j]*((2*np.pi)**(-n_dims/2))*(detSigma**(-1/2))*np.exp((-((self.noisy_data-means[j])@invSigma@(self.noisy_data-means[j]).T)/2),dtype=np.longdouble)
            likelihood += self.weights[j]*(np.exp(-means[j]+self.noisy_data-self.noisy_data*(np.log(self.noisy_data)-np.log(means[j]))))
        log_likelihoods += np.log(likelihood) # store in log_likelihoods numpy aikki rray
        
        return np.sum(log_likelihoods) # return total log-likelihood of each sample given the parameters


    def dfdx(self, mu):
        '''
        Central differencing scheme
        '''
        self.epsilon = 0.01
        n_dims = self.n_dimensions
        n_comp = self.n_components
        mu = np.array(mu)
        delta_theta = []
        res_mu_plus = []
        res_mu_minus = []
        for i in range(mu.shape[0]):
            E_plus = np.zeros(n_dims*n_comp)
            E_minus = np.zeros(n_dims*n_comp)
            E_plus += [(self.epsilon if j == i else 0) for j in range(n_dims*n_comp)]
            E_minus += [(-self.epsilon if j == i else 0) for j in range(n_dims*n_comp)]
            res_mu_plus.append(E_plus)
            res_mu_minus.append(E_minus)

        for i in range(n_dims*n_comp):
            delta_thetai = (self.calculate_log_likelihoods_mu(mu+res_mu_plus[i]) - self.calculate_log_likelihoods_mu(mu+res_mu_minus[i]))/(2*self.epsilon)
            delta_theta.append(delta_thetai)
            
        return delta_theta

    def dfdx_1(self, mu):
        '''
        Forward differencing scheme
        '''
        self.epsilon = 0.01
        n_dims = self.n_dimensions
        n_comp = self.n_components
        mu = np.array(mu)
        delta_theta = []
        res_mu_plus = []
        for i in range(mu.shape[0]):
            E_plus = np.zeros(n_dims*n_comp)
            E_plus += [(self.epsilon if j == i else 0) for j in range(n_dims*n_comp)]
            res_mu_plus.append(E_plus)

        for i in range(n_dims*n_comp):
            delta_thetai = (self.calculate_log_likelihoods_mu(mu+res_mu_plus[i]) - self.calculate_log_likelihoods_mu(mu))/(self.epsilon)
            delta_theta.append(delta_thetai)
            
        return delta_theta

    def dfdx_2(self, mu):
        '''
        Backward differencing scheme
        '''
        self.epsilon = 0.01
        n_dims = self.n_dimensions
        n_comp = self.n_components
        mu = np.array(mu)
        delta_theta = []
        res_mu_minus = []
        for i in range(mu.shape[0]):
            E_minus = np.zeros(n_dims*n_comp)
            E_minus += [(-self.epsilon if j == i else 0) for j in range(n_dims*n_comp)]
            res_mu_minus.append(E_minus)

        for i in range(n_dims*n_comp):
            delta_thetai = (self.calculate_log_likelihoods_mu(mu) - self.calculate_log_likelihoods_mu(mu+res_mu_minus[i]))/(self.epsilon)
            delta_theta.append(delta_thetai)
            
        return delta_theta

if __name__ == "__main__":
    dimensions = 2
    components = 2
    means = [[1, 1], [5, 5]]
    covs = [[[1, 0], [0, 1]], 
    [[2, 0], [0, 2]]]
    weights = [0.3, 0.7]
    n_samples = 200
    noise = 0.2
    gmm1 = GaussianMixtureModel(n_dimensions = dimensions, 
                            n_components = components, 
                            means = means, 
                            covs = covs, 
                            weights = weights)
    logLikeli = gmm1.calculate_log_likelihoods_mu

    def U(theta: List[np.longdouble]) -> np.longdouble:
        '''
        the potential energy function: 
        U(theta) = -log(probability distribution of theta)
        '''
        return - logLikeli(theta)

    def grad_U(theta: List[np.longdouble]) -> np.longdouble:
        '''
        the derivative of the potential energy function
        dU/dmu
        '''
        DRho = gmm1.dfdx_2(theta)
        Rho = - logLikeli(theta)
        return np.array([dRho/Rho for dRho in DRho])

    def HMC(epoch, L, epsilon, U, grad_U, current_theta):

        """
        
        Hamiltonian Monte Carlo algorithm

        Parameters
        ----------
        epoch: number of iteration of the algorithm
        L: number of steps of leap frog
        epsilon: step size for discrete approximation
        U: potential energy
        grad_U: derivative of potential energy
        current_theta: the current 'position'
        
        Returns
        -------
        theta_accept: if accpeted
        theta_reject: if rejected

        
        """
        
        theta_accept = []
        theta_reject = []

        for _ in range(epoch):
        
            theta = current_theta
            rho = np.random.normal(loc = 0, scale = 1, size= len(theta)) # sample random momentum
            current_rho = rho
            
            # make a half step for momentum at the beginning
            rho = rho - epsilon * grad_U(theta) / 2 

            # alternate full steps for position and momentum
            for i in range(1, L):

                #make a full step for the position
                theta = theta + epsilon * rho

                #make a full step for the momentum, except at end of trajectory
                if (i != L):
                    rho = rho - epsilon * grad_U(theta)
            
            # make a half step for momentum at the end
            rho = rho - epsilon * grad_U(theta) / 2

            # Negate momentum at end of trajectory to make the proposal symmetric
            rho = -rho

            # Evaluate potential and kinetic energies at start and end of trajectory (K kinetic energy, U potential energy)
            current_U = U(current_theta)
            current_K = sum(current_rho**2) / 2
            proposed_U = U(theta)
            proposed_K = sum(rho**2) / 2

            # Accept or reject the state at end of trajectory, returning either
            # the position at the end of the trajectory or the initial position

            if (np.random.uniform(0, 1) < np.exp((current_U - proposed_U + current_K - proposed_K),dtype=np.longdouble)):
                current_theta = theta
                #return (theta) # accept
                theta_accept.append(theta)
            else:
                theta_reject.append(theta)
                #return (current_theta) # reject

        return theta_accept, theta_reject


    with cProfile.Profile() as profile:
        X1, noisy_data1 = gmm1.generate_data(n_samples, noise_scale = noise)
        theta_accept, theta_reject = HMC(epoch=3000, L=20, epsilon=0.01, U=U, grad_U=grad_U, current_theta=np.array([0,0,0,0]))
        
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("results.prof")

    print(len(theta_accept)/(len(theta_accept)+len(theta_reject)))