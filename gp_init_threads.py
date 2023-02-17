##  Author: Renee Spiewak


import celerite as cel
import numpy as np


def init_thread(kern_len, _data, mjds, errs=None):
    global gp
    global data
    data = _data
    variance = np.var(data)
    kernel = cel.terms.Matern32Term(np.log(1), np.log(kern_len)) + cel.terms.JitterTerm(np.log(np.sqrt(variance)))
    gp = cel.GP(kernel, np.mean(data), fit_mean=True)
    if errs is not None:
        gp.compute(mjds, errs)
    else:
        gp.compute(mjds)
        
    return()


def lnprob(p):
    global gp
    global data
    # Trivial uniform prior
    #if p[-1] < np.log(3.2e+02) or p[-1] > np.log(1.5e+06):
    #    return(-np.inf)
            
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return(-np.inf)
        
    # Update the kernel and compute the log-likelihood
    gp.set_parameter_vector(p)
    return(gp.log_likelihood(data, quiet=True))
    
