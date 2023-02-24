import numpy as np

def auto_correlation(thetas, time_lag):
    
    """
    
    calculating the auto-correlation of the stochastic processes of parameter theta

    Arguments
    ---------
    theta : the inputed value of the parameter theta generated by the stochastic process
    time_lag : the value of the time-lag of the of the auto-correlation

    Returns
    -------
    auto_cor : the auto-correlation value of the inputed parameters

    """

    thetas = np.asarray(thetas)
    n = thetas.shape[0]
    t1 = 0
    assert(t1+time_lag<=n-1)
    auto_cor = 0
    while(t1+time_lag<=n-1):
        auto_cor += thetas[t1]*thetas[t1+n]
        t1 += 1
    return auto_cor/t1