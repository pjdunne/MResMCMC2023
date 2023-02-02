import numpy as np

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

        if (np.random.uniform(0, 1) < np.exp(current_U - proposed_U + current_K - proposed_K)):
            current_theta = theta
            #return (theta) # accept
            theta_accept.append(theta)
        else:
            theta_reject.append(theta)
            #return (current_theta) # reject

    return theta_accept, theta_reject
