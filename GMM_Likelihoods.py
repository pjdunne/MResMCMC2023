import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata


from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns


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

        
    def generate_data(self, n_samples, noise_scale=0.1):

        X = np.zeros((n_samples, self.n_dimensions))
        c = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        for i in range(self.n_components):
            idx = (c == i)
            X[idx, :] = np.random.multivariate_normal(mean=self.means[i], cov=self.covs[i], size=np.sum(idx))

        # first create a numpy histogram from the GMM
        bins = 30
        data_hist, bin_edges = np.histogramdd(X, bins=bins)
        bin_centers = []
        for i in range(self.n_dimensions):
            bin_centers.append((bin_edges[i][:-1] + bin_edges[i][1:]) / 2.0)

        # add Poisson fluctuations to the value in each bin
        noisy_data_hist = np.random.poisson(lam=data_hist)

        noisy_data = np.array(np.meshgrid(*bin_centers)).T.reshape(-1,self.n_dimensions)
        noisy_data = noisy_data.repeat(noisy_data_hist.flatten(), axis=0)
        noisy_data = noisy_data + np.random.normal(scale=noise_scale, size=noisy_data.shape)
        self.noisy_data = noisy_data

        return X, noisy_data

    
    
    def plot_scatter(self, X):
        '''
        Plots scatter plot for data.
        
        Arguments
        X [np.array]: dataset for which scatter plot will be plotted
        
        Returns
        None
                
        '''
        plt.rcParams["figure.figsize"] = (8, 6)
        
        n_dimensions = X.shape[1]
        
        # 2D data
        if n_dimensions == 2:
            plt.scatter(X[:, 0], X[:, 1], marker = 'o', s = 6, alpha = 0.5)
            plt.title("Scatter plot for Gaussian Mixture Model")
            plt.xlabel("$\mu1$")
            plt.ylabel("$\mu2$")
            plt.savefig('2Dscatter')
            plt.show()
        
        # 3D data
        elif n_dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker = 'o', s = 6, alpha = 0.5)
            plt.title("Scatter plot for Gaussian Mixture Model")
            ax.set_xlabel("$\mu1$")
            ax.set_ylabel("$\mu2$")
            ax.set_zlabel("$\mu3$")
            plt.savefig('3Dscatter')
            plt.show()


    def plot_scatters(self, Xs, labels, figsize=(8,6)):
        '''
        Plots scatter plot for data.
        
        Arguments
        Xs List[np.array]: the list of datasets for which scatter plot will be plotted
        labels List[str]: the list of the label of each dataset will be plotted
        figsize : the fig size of the output plot
        
        Returns
        None
                
        '''


        plt.rcParams["figure.figsize"] = figsize
            
        n_dimensions = Xs[0].shape[1]
            
        # 2D data
        if n_dimensions==2:
            for i, X in enumerate(Xs):
                plt.scatter(X[:, 0], X[:, 1], marker = 'o', s = 6, alpha = 0.5, label=labels[i])
                plt.title("Scatter plot for Gaussian Mixture Model")
                plt.legend(fontsize=min(figsize)*2)
                plt.xlabel("$\mu1$")
                plt.ylabel("$\mu2$")
            plt.savefig('2Dscatter')
            
        # 3D data
        elif n_dimensions == 3:
            for i, X in enumerate(Xs):
                fig = plt.figure(min(figsize))
                ax = fig.add_subplot(111, projection = '3d')
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker = 'o', s = 6, alpha = 0.5, label=labels[i])
                plt.title("Scatter plot for Gaussian Mixture Model")
                ax.legend(fontsize=min(figsize)*2)
                ax.set_xlabel("$\mu1$")
                ax.set_ylabel("$\mu2$")
                ax.set_zlabel("$\mu3$")
            plt.savefig('2Dscatter')

        plt.show()




    def plot_histograms(self, X, noise_scale=None):
        '''
        Plots histograms for data.
        Arguments
        X [np.array]: dataset for which histograms will be plotted
        noise_scale [float]: standard deviation of the Poisson noise added to the data, if any
        Returns
        None
        '''

        n_dimensions = X.shape[1]

        fig, axs = plt.subplots(n_dimensions, figsize=(6, 3*self.n_dimensions))

        for i in range(n_dimensions):
            if noise_scale is not None:
                noisy_data = np.random.normal(X[:, i], scale=noise_scale)
                noisy_counts, noisy_edges = np.histogram(noisy_data, bins=30)
                noisy_bin_centers = (noisy_edges[:-1] + noisy_edges[1:]) / 2.0
            else:
                noisy_counts, noisy_bin_centers = None, None

            data_counts, data_edges = np.histogram(X[:, i], bins=30)
            data_bin_centers = (data_edges[:-1] + data_edges[1:]) / 2.0

            axs[i].bar(data_bin_centers, data_counts, alpha=0.5, label='Data', width=(data_bin_centers[1] - data_bin_centers[0]))

            if noisy_data is not None:
                for j in range(len(data_counts)):
                    idx = np.abs(noisy_bin_centers - data_bin_centers[j]).argmin()
                    axs[i].scatter(data_bin_centers[j], noisy_counts[idx], color='red')

            axs[i].set_xlabel('mu{}'.format(i+1))
            axs[i].set_ylabel('Frequency')

            # Fit a normal distribution to the data
            if noise_scale is None:
                mu, std = norm.fit(X[:, i])
                xmin, xmax = axs[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                axs[i].plot(x, p*len(X[:,i])*(data_bin_centers[1] - data_bin_centers[0]), 'k', linewidth=2)

                # Add the best fit line to the legend
                axs[i].legend(['Best fit line', 'Data'], loc='upper right')
            else:
                axs[i].legend(['Noisy data', 'Data'], loc='upper right')

        fig.tight_layout()
        plt.show()


    def pdf(self, X):
        '''
        Calculates the probability density function (pdf) of the Gaussian Mixture Model 
        for a given set of samples in X.
    
    
            p(x_j|θ) is the probability density function at each independent sample with
            θ represents the parameters mean, covariance, and mixture coefficients,
            and x_1, x_2, ... , x_n are the observed samples in dataset X.
        
        
        Arguments
        X [np.array]: dataset for which pdf will be calculated
        Returns
        probs [np.array]: probabilities of the Gaussian Mixture Model for each sample in X
        
        '''

        X = np.asarray(X)

        if len(X.shape)==1:
            prob = 0
            
            for i in range(self.n_components):
                prob += self.weights[i]*multivariate_normal.pdf(X, self.means[i], self.covs[i])
            return prob

        else:
            prob = np.zeros((X.shape[0], self.n_components)) # initialize a numpy array of zeros to store probs of each sample


            for i in range(self.n_components):
                prob[:,i] = self.weights[i]*multivariate_normal.pdf(X, self.means[i], self.covs[i])

            probs = np.sum(prob, axis=1)

            return probs

         
    def calculate_log_likelihoods(self, mean_and_cov):
        '''
        Calculates the log-likelihood of the Gaussian Mixture Model for a given set of samples in X.
        
            The likelihood is calculated as the product of individual pdfs at the observed samples:
            L(θ) = p(x_1, x_2, ..., x_N|θ) = Π_j=1^N p(x_j|θ)

            Since likelihood of each point may be very small, leading to extremely small products, 
            a log likelihood is often used in practice:
            logL(θ) = Σ_j=1^N log(p(x_j|θ)) 

        Arguments
        mean_and_cov [list]: list of tuples containing the mean and covariance for each Gaussian component

        Returns
        log_likelihoods [np.array]: log-likelihood of each sample in dataset X

        '''
        
        log_likelihoods = np.zeros(self.noisy_data.shape[0]) # initialize a numpy array of zeros to store log-likelihoods of each sample

        for i, x in enumerate(self.noisy_data):
            log_likelihood = 0
            for j in range(self.n_components):
                # calculate log-likelihood of each sample by
                # adding product of mixture coefficient and pdf at sample x_j for each Gaussian component k
                log_likelihood += self.weights[j] * multivariate_normal.pdf(x, mean = mean_and_cov[j][0], cov = mean_and_cov[j][1])
            log_likelihoods[i] = np.log(log_likelihood) # store in log_likelihoods numpy array

        return log_likelihoods # return log-likelihood of each sample in dataset X


    def plot_log_likelihood(self, X, probs):
        '''
        Plots the log-likelihood surface of data.
        Arguments
        X [np.array]: dataset for which scatterplot will be plotted
        probs [np.array]: probabilities of the Gaussian Mixture Model for each sample in X
        
        Returns
        None
        '''
        
        if self.n_dimensions == 2:
            fig = plt.figure(figsize = (8, 8), facecolor = "white")
            ax = fig.add_subplot(projection = "3d")
            
            x = X[:, 0]
            y = X[:, 1]
            z = probs
            
            # interpolate the points to make the plot smoother
            xi = np.linspace(min(x), max(x), 500)
            yi = np.linspace(min(y), max(y), 500)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), z, (X, Y), method='cubic')

            # plot the wireframe
            ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
            
            # plot scatter plot
            ax.scatter(x, y, z, color = 'r',  marker = '.', alpha = 0.7)

            ax.set_xlabel("mu1")
            ax.set_ylabel("mu2")
            ax.set_zlabel("Likelihood")
            ax.set_title("Likelihood Plot")
            fig.tight_layout()
            plt.savefig('likelihood')
            fig.show()
