#!/usr/bin/env python
# coding: utf-8

# This notebook is a work in progress, a first attempt to combine the eigenvalue GPs with nu-dot measurements to find correlations. 
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[2]:


import os
import numpy as np
from scipy import stats
from astropy.timeseries import LombScargle
import astropy.units as u
from matplotlib import pyplot as plt
import cmasher as cmr
from all_prof_functions import plot_eig_gp

get_ipython().run_line_magic('aimport', '-os -np -plt -cmr -mpr -corner')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# Set up some plotting stuff
plt.rc('savefig', bbox='tight')
plt.rcParams['text.usetex'] = False
use_bk_bgd = False #### Change this to use white backgrounds for plots  #####
if use_bk_bgd:
    plot_style = 'dark_background'
    # The CMasher package provides lots of lovely colour maps; chroma is a handy sequential cmap
    cmap = cmr.chroma_r
    c1 = cmap(0.0)
    c2 = cmap(0.1)
    c3 = cmap(0.33)
    c4 = cmap(0.55)
    c5 = cmap(0.68)
    c6 = cmap(0.815)

else:
    plot_style = 'default'
    cmap = cmr.chroma
    c1 = cmap(0.0)
    c2 = cmap(0.3)
    c3 = cmap(0.53)
    c4 = cmap(0.65)
    c5 = cmap(0.78)
    c6 = cmap(0.915)


# The following cell initialises some key variables, defining the dataset you're working on and where the data are located. Change the parameters noted below. 
# 

# In[4]:


data_dir = '/home/s86932rs/research/nudot_stuff/' # change this to the absolute/relative path to your data and .npz files
plots_dir = os.path.join(data_dir, 'plots') # change this if necessary
psr = 'B1828-11' # change to your pulsar name
freq = 1400 # change to your frequency band
be = 'afb' # change this to your backend


# In[5]:


be = be.lower()
BE = be.upper()
npz_file = os.path.join(data_dir, psr+'_gps_fin.npz') # contains BE_mjds_pred, BE_res_pred, BE_vars_pred
if not os.path.exists(npz_file):
    raise(RuntimeError("File containing eigenvalue GPs does not exist"))
    
nudot_file = os.path.join(data_dir, psr+"_nudot_gp.txt") # contains columns of MJD, nudot, uncertainty
if not os.path.exists(nudot_file):
    raise(RuntimeError("File containing nu-dot GPs does not exist"))


# Read in the data, selecting only the eigenvalue GPs for the relevant backend. 
# 

# In[6]:


var_dict = {}
with np.load(npz_file, allow_pickle=True) as f_npz:
    for key in f_npz.keys():
        if BE in key:
            var_dict[key] = f_npz[key]
            
if len(var_dict) == 0:
    raise(RuntimeError("No data found for that backend"))
    
eig_mjds = var_dict[BE+'_mjds_pred'] # these MJDs were previously set using the nu-dot MJDs
eig_vals = var_dict[BE+'_res_pred']
eig_errs = var_dict[BE+'_vars_pred']


# In[7]:


nudot_mjds, nudot_vals = np.loadtxt(nudot_file, unpack=True, usecols=(0, 1))
if nudot_vals.mean() > 1e7: # fix any very wrong orders of magnitude
    nudot_vals *= 1e-30
elif nudot_vals.mean() > 1e-7: # or just the appropriate order of magnitude change
    nudot_vals *= 1e-15

if min(nudot_mjds) >= max(eig_mjds) or min(eig_mjds) >= max(nudot_mjds):
    raise(RuntimeError("No overlap between timespans"))
    
nudot_errs = None


# In[8]:


print("The nudot MJDs span {:.2f} to {:.2f}, and the {} profile MJDs span {:.2f} to {:.2f}".format(min(nudot_mjds), max(nudot_mjds), BE, min(eig_mjds), max(eig_mjds)))
eig_lim = np.logical_and(eig_mjds >= min(nudot_mjds)-1, eig_mjds <= max(nudot_mjds)+1)
nudot_lim = np.logical_and(nudot_mjds >= min(eig_mjds)-1, nudot_mjds <= max(eig_mjds)+1)

if len(eig_mjds[eig_lim]) != len(nudot_mjds[nudot_lim]):
    print("The maximum and minimum values of the nudot array and eigs array are {:.3f}, {:.3f}, {:.3f}, and {:.3f}, respectively"
          .format(nudot_mjds[nudot_lim].min(), nudot_mjds[nudot_lim].max(), eig_mjds[eig_lim].min(), eig_mjds[eig_lim].max()))
    print(eig_mjds[eig_lim][-2:])
    with plt.style.context(plot_style):
        #plt.plot(eig_mjds[eig_lim], color=c2)
        #plt.plot(nudot_mjds[nudot_lim], '--', color=c3)
        #gap = eig_mjds[eig_lim][-20] - eig_mjds[eig_lim][20] - 5
        plt.plot(eig_mjds[eig_lim][-20:], color=c2)
        plt.plot(nudot_mjds[nudot_lim][-20:], color=c3)
        plt.show()
    raise(RuntimeError("The trimmed MJD arrays have different lengths: {} and {}".format(len(eig_mjds[eig_lim]), len(nudot_mjds[nudot_lim]))))


# In[9]:


gp_corrs = np.zeros(eig_vals.shape[0])
for eignum in range(eig_vals.shape[0]):
    if eignum == 1:
        suff = 'st'
    elif eignum == 2:
        suff = 'nd'
    elif eignum == 3:
        suff = 'rd'
    else:
        suff = 'th'
    res = stats.spearmanr(eig_vals[eignum,eig_lim], nudot_vals[nudot_lim])
    gp_corrs[eignum] = res.correlation
    print("The correlation value for the {}{} eigenvector is {:.3f}".format(eignum, suff, res.correlation))


# In[ ]:





# In[10]:


# output the correlations in a simple format
out_file = os.path.join(data_dir, '{}_{}_{}_corrs.txt'.format(psr, be, freq))
with open(out_file, 'w') as f:
    f.write('# Component num. | Correlation value\n')
    for eignum, corr in enumerate(gp_corrs):
        f.write('{}\t\t{}\n'.format(eignum, corr))


# In[11]:


corr_lim = np.abs(gp_corrs) > 0.3
err_lim = np.array([np.any(eig_vals[num,:] - eig_vals[num,:].mean() > np.sqrt(eig_errs[num,:])) for num in range(eig_vals.shape[0])])
use_lim = np.logical_and(corr_lim, err_lim)

if len(gp_corrs[use_lim]) == 0:
    print("There are no significant correlated eigenvectors for {} with {} at {} MHz".format(psr, BE, freq))
else:
    nudot_vars = nudot_errs[nudot_lim]**2 if nudot_errs is not None else None
        
    plot_eig_gp(eig_mjds[eig_lim], eig_vals[:,eig_lim][use_lim,:], eig_errs[:,eig_lim][use_lim,:], bk_bgd=use_bk_bgd, show=True,
                nudot_mjds=nudot_mjds[nudot_lim], nudot_vals=nudot_vals[nudot_lim], nudot_vars=nudot_vars, eig_nums=np.arange(len(gp_corrs))[use_lim],
                savename=os.path.join(plots_dir, "{}_{}_{}_nudot_eigs_corr.png".format(psr, be, freq)))


# Do a Lomb-Scargle analysis on the nu-dot data as well.
# 

# In[12]:


min_freq = 0.025/u.year # a period of 40 years
max_freq = 6/u.year # a period of ~60 days
nudot_errs = nudot_errs*u.Hz if nudot_errs is not None else None
LS = LombScargle(nudot_mjds*u.day, nudot_vals*u.Hz, nudot_errs)
freqs, power = LS.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=10)
freq_max_power = freqs[power == np.max(power)][0]
print("The frequency of the maximum power is {:.3f} ({})".format(freq_max_power.value, freq_max_power.unit))
print("That corresponds to a period of {:.2f}".format((1/freq_max_power).to('day')))
with plt.style.context(plot_style):
    plt.clf()
    plt.plot(freqs, power, '-', color=c1)
    plt.ylabel('Periodogram Power')
    plt.xlabel('Frequency ({})'.format(freqs.unit))
    plt.xlim(min_freq.value, max_freq.value)
    plt.savefig(os.path.join(plots_dir, '{}_{}_nudot_LS.png'.format(psr, freq)), bbox_inches='tight')
    plt.show()
    


# In[ ]:




