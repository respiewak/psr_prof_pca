#!/usr/bin/env python
# coding: utf-8


import os, sys
import numpy as np
from matplotlib import pyplot as plt
import argparse as ap
import cmasher as cmr
import corner    
import multiprocessing as mpr
mpr.set_start_method('fork')
from all_prof_functions import (run_each_gp, plot_recon_profs, setup_log)


# Set up some plotting stuff
plt.rc('savefig', bbox='tight')

# The following section contains variables describing the dataset and where the data are located
pars = ap.ArgumentParser(description='The third step')
pars.add_argument('psr_list', nargs='+', help="Pulsar names")
pars.add_argument('-f', '--frq_list', nargs='+', default=1400, type=int,
                  help="Frequency band(s) as they appear in the file names (integers)")
pars.add_argument('-b', '--be_list', nargs='+', default=['afb', 'dfb'],
                  help="Backend names/abbreviations")
pars.add_argument('-d', '--data_dir', default='../profiles/',
                  help="Absolute or relative path to the directory containing profile data")
pars.add_argument('-l', '--log_name', default='prof_gps',
                  help="Name for the log file")
pars.add_argument('-k', '--use_bk_bgd', action='store_true',
                  help='Use a dark background for all plots')
pars.add_argument('-m', '--max_num', default=3, type=int, help='Max. number of eigen components')
pars.add_argument('-w', '--num_walkers', default=200, type=int,
                  help="Number of walkers for MCMC")
pars.add_argument('-cb', '--burn_chain', default=300, type=int,
                  help="Number of steps for burn-in for MCMC")
pars.add_argument('-cp', '--prod_chain', default=3000, type=int,
                  help="Number of steps for production chain for MCMC")
args = vars(pars.parse_args())

use_bk_bgd = args['use_bk_bgd']
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

#data_dir = '/home/s86932rs/research/nudot_stuff/'
data_dir = args['data_dir']
log_name = args['log_name']
if len(log_name.split('.')) == 1:
    log_name += '.log'
    
logger = setup_log(os.path.join(data_dir, log_name))
plots_dir = os.path.join(data_dir, 'plots')
if not os.path.exists(plots_dir):
    logger.warning("Creating directory for the output plots: "+plots_dir)
    os.chdir(data_dir)
    os.mkdir('plots')

#psr = 'B1828-11'
psr_list = args['psr_list']
if type(psr_list) is str:
    psr_list = [psr_list]

#freq = 1400
frq_list = args['frq_list']
if type(frq_list) is int:
    frq_list = [frq_list]

#be = 'afb'
be_list = args['be_list']
if type(be_list) is str:
    be_list = [be_list]
    
BE_list = [A.upper() for A in be_list]
be_list = [A.lower() for A in BE_list]

for psr in psr_list:
    for freq in frq_list:
        var_dict = {}
        be_exists = []
        for BE, be in zip(BE_list, be_list):
            desc = "{}_{}_{}".format(psr, be, freq)
            DESC = "{}_{}_{}".format(psr, BE, freq)
            logger.info('Working on the {} dataset'.format(DESC))
            
            npz_file = os.path.join(data_dir, psr+'_{}_eigs.npz'.format(freq))
            if not os.path.exists(npz_file):
                logger.error("File containing eigenvalues for {} does not exist".format(DESC))
                continue

            nudot_file = os.path.join(data_dir, psr+"_nudot_gp.txt") # contains columns of MJD, nudot, uncertainty

            with np.load(npz_file, allow_pickle=True) as d:
                if BE+'_errs' in d.keys():
                    BE_errs = d[BE+'_errs']
                    BE_mean = d[BE+'_mean']
                    BE_eigval = d[BE+'_values']
                    BE_eigvec = d[BE+'_vectors']
                    BE_mjds = d[BE+'_mjds']
                    BE_rms = d[BE+'_rms']
                else:
                    logger.warning("No data to load for "+DESC)
                    continue
                
            logger.info("The shape of the {} eigval array is {}.".format(BE, BE_eigval.shape))
            if len(BE_errs) == BE_eigval.shape[0] and len(BE_mean) == BE_eigvec.shape[1] and BE_eigval.shape[1] == BE_eigvec.shape[0] \
                and BE_eigval.shape[0] == len(BE_mjds) and BE_eigval.shape[0] == len(BE_rms):
                logger.info("All {} shapes and lengths agree".format(BE))
            else:
                logger.warning("Lengths and shapes for {} do not agree!".format(BE))

            with plt.style.context(plot_style):
                plt.clf()
                _, bins, _ = plt.hist(BE_mjds[1:] - BE_mjds[:-1], bins=50, color=c2)
                plt.xlabel("Separation between observations (MJD)")
                plt.savefig(os.path.join(plots_dir, desc+"_lag_hist.png"), bbox_inches='tight')
                #plt.show()

            # read in the MJDs from the nudot GP file to make alignment (for finding correlations) easier
            if not os.path.exists(nudot_file):
                logger.warning("File containing nu-dot GPs does not exist")
                BE_mjds_pred = None
                BE_mjds_pred = None
            else:
                nudot_mjds = np.loadtxt(nudot_file, unpack=True, usecols=(0,))
                if np.any(nudot_mjds != sorted(nudot_mjds)):
                    logger.info("Sorting nudot MJDs")
                    nudot_mjds = sorted(nudot_mjds)
                    
                avg_sep = np.mean(nudot_mjds[1:] - nudot_mjds[:-1]) # separation between nudot MJDs
                nudot_min = min(nudot_mjds)
                nudot_max = max(nudot_mjds)
                BE_min = min(BE_mjds)
                BE_max = max(BE_mjds)
                if nudot_min > BE_min:
                    num_pre = int(np.ceil((nudot_min - BE_min)/avg_sep)) # number of points needed to fill
                    new_min = nudot_min - avg_sep*num_pre # shift the minimum to use an integer
                    pre_mjds = np.linspace(new_min, nudot_min, num_pre, endpoint=False)
                    nudot_mjds = np.append(pre_mjds, nudot_mjds)
        
                if nudot_max < BE_max:
                    num_post = int(np.ceil((BE_max - nudot_max)/avg_sep)) # number of points needed to fill
                    new_max = nudot_max + avg_sep*num_post # shift the maximum to use an integer
                    post_mjds = np.linspace(nudot_max, new_max, num_post, endpoint=False)
                    nudot_mjds = np.append(nudot_mjds, post_mjds)

                BE_lim = np.logical_and(nudot_mjds <= max(BE_mjds)+0.1, nudot_mjds >= min(BE_mjds)-0.1)
                BE_mjds_pred = nudot_mjds[BE_lim]
                BE_mjds_pred = np.unique(BE_mjds_pred) # this shouldn't be necessary... why is it?
        
                logger.info("The MJDs for the GP prediction span from {:.5f} to {:.5f} with an average separation of {:.3f}"
                            .format(min(nudot_mjds), max(nudot_mjds), avg_sep))

            read_old = False # make this not hard-coded
            gp_file = os.path.join(data_dir, '{}_{}_gps_fin.npz'.format(psr, freq))
            if read_old and os.path.exists(gp_file):
                with np.load(gp_file, allow_pickle=True) as d:
                    if BE+'_mjds_pred' in d.keys():
                        BE_mjds_pred = d[BE+'_mjds_pred']
                        BE_pred_vars = d[BE+'_vars_pred']
                        BE_pred_res = d[BE+'_res_pred']
                    else:
                        logger.info("No {} arrays in npz file".format(DESC))
                        
            if BE+'_pred_res' not in locals():
                logger.info("Running GP for {} data".format(BE))
                lags = BE_mjds[1:] - BE_mjds[:-1]
                pmin = np.percentile(lags, 97.5)
                logger.info("The minimum length scale for {} is {:.2f}".format(BE, pmin))
                mjd_range = BE_mjds[-1] - BE_mjds[0]
                kern_len = max(pmin*2, mjd_range/10)

                BE_pred_res, BE_pred_vars, BE_mjds_pred = run_each_gp(
                    BE_eigval, BE_mjds, BE_errs, kern_len=kern_len, max_num=args['max_num']-1,
                    prior_min=pmin, prior_max=0.5*mjd_range, mjds_pred=BE_mjds_pred,
                    burn_chain=args['burn_chain'], prod_chain=args['prod_chain'], num_walkers=args['num_walkers'],
                    plot_chains=True, plot_corner=True, plot_gps=True, mcmc=True, multi=True,
                    verb=False, bk_bgd=use_bk_bgd, show_plots=False, plot_dir=plots_dir, descrpn=desc,
                    gp_plotname=os.path.join(plots_dir, desc+'_gp_preds.png'), logg=logger)

            plot_recon_profs(BE_mean, BE_eigvec, BE_mjds_pred, BE_pred_res, psr, mjds_real=BE_mjds, bk_bgd=use_bk_bgd,
                             sub_mean=True, savename=os.path.join(plots_dir, desc+'_recon_wfall.png'), show=False)
            logger.info("Plot of reconstructed profile saved to "+os.path.join(plots_dir, desc+'_recon_wfall.png'))

            plt.close('all')
            # We want to save the arrays containing GP predicted values, so the following cells will check if an older file exists (which may contain arrays for different datasets), read that into a separate dictionary, and write both dictionaries to the '.npz' file. 
            var_dict[BE+'_mjds_pred'] = BE_mjds_pred
            var_dict[BE+'_res_pred'] = BE_pred_res
            var_dict[BE+'_vars_pred'] = BE_pred_vars
            
        # after looping over all given backends, write out data for each frequency and pulsar combo
        out_file = os.path.join(data_dir, '{}_{}_gps_fin.npz'.format(psr, freq))
        old_dict = {}
        if os.path.exists(out_file):
            with np.load(out_file, allow_pickle=True) as f:
                for key in f.keys():
                    if key not in var_dict.keys():
                        old_dict[key] = f[key]
                    else:
                        logger.info("Replacing an older value for "+key)
                
        np.savez(out_file, **var_dict, **old_dict)
