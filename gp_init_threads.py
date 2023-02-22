##  Author: Renee Spiewak


import numpy as np
import celerite as cel


def init_thread(kern_len, _data, mjds, errs=None):
    global gp
    global data
    data = _data
    variance = np.var(data)
    term1 = cel.terms.Matern32Term(np.log(1), np.log(kern_len), bounds=dict(log_sigma=(-15, 15), log_rho=(np.log(57.61), np.log(2467.01))))
    term2 = cel.terms.JitterTerm(np.log(np.sqrt(variance)), bounds=dict(log_sigma=(-15, 15)))
    kernel = term1 + term2
    gp = cel.GP(kernel, np.mean(data), fit_mean=True)
    if errs is not None:
        gp.compute(mjds, errs)
    else:
        gp.compute(mjds)
        
    return()


def lnprob(p):
    global gp
    global data
    # Trivial uniform prior on length scale
    if p[1] < np.log(57.61) or p[1] > np.log(2467.01):
        return(-np.inf)
            
    if np.any((-30 > p) + (p > 30)):
        return(-np.inf)
        
    # Update the kernel and compute the log-likelihood
    gp.set_parameter_vector(p)
    return(gp.log_likelihood(data, quiet=True))
    
