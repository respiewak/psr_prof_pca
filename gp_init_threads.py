##  Author: Renee Spiewak


import george
import numpy as np


def init_thread(kern_len, _data, mjds, errs=None):
    global gp
    global data
    data = _data
    variance = np.var(data)
    kernel = variance * george.kernels.Matern52Kernel(kern_len)
    gp = george.GP(kernel, np.mean(data), fit_mean=True, solver=george.HODLRSolver,
                   white_noise=np.log(np.sqrt(variance)*0.8), fit_white_noise=True)
    if errs is not None:
        gp.compute(mjds, errs)
    else:
        gp.compute(mjds)
        
    return()


def lnprob(p):
    global gp
    global data
    # Trivial uniform prior
    if p[-1] < np.log(1.4e+03) or p[-1] > np.log(1.8e+06):
        return(-np.inf)
            
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return(-np.inf)
        
    # Update the kernel and compute the lnlikelihood
    gp.set_parameter_vector(p)
    return(gp.lnlikelihood(data, quiet=True))
    
