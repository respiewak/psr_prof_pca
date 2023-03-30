#!/usr/bin/env python
# coding: utf-8

# This notebook is a work in progress, a first attempt to combine the eigenvalue GPs with nu-dot measurements to find correlations. 

import os
import numpy as np
from scipy import stats
from astropy.timeseries import LombScargle
import astropy.units as u
from matplotlib import pyplot as plt
import argparse as ap
import cmasher as cmr
from all_prof_functions import plot_eig_gp, setup_log, plot_colors_corrs


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
    k_alpha = 0.6

else:
    plot_style = 'default'
    cmap = cmr.chroma
    c1 = cmap(0.0)
    c2 = cmap(0.3)
    c3 = cmap(0.53)
    c4 = cmap(0.65)
    c5 = cmap(0.78)
    c6 = cmap(0.915)
    k_alpha = 0.4

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
    nudot_file = os.path.join(data_dir, psr+"_nudot_gp.txt") # contains columns of MJD, nudot, uncertainty
    if not os.path.exists(nudot_file):
        logger.warning("File containing nu-dot GPs does not exist; skipping pulsar")
        continue
        
    nudot_mjds, nudot_vals = np.loadtxt(nudot_file, unpack=True, usecols=(0, 1))
    if nudot_vals.mean() > 1e7: # fix any very wrong orders of magnitude
        nudot_vals *= 1e-30
    elif nudot_vals.mean() > 1e-7: # or just the appropriate order of magnitude change
        nudot_vals *= 1e-15

    nudot_errs = None

    if np.any(sorted(nudot_mjds) != nudot_mjds):
        logger.info("Nudot MJDs are not sorted")
        sort_inds = np.argsort(nudot_mjds)
        nudot_mjds = nudot_mjds[sort_inds]
        nudot_vals = nudot_vals[sort_inds]
        
        if np.any(sorted(nudot_mjds) != nudot_mjds):
            logger.warning("Sorting nudot MJDs failed!!!!")

    for freq in frq_list:
        var_dict = {}
        be_exists = []
        for BE, be in zip(BE_list, be_list):
            desc = "{}_{}_{}".format(psr, be, freq)
            DESC = "{}_{}_{}".format(psr, BE, freq)
            logger.info('Working on the {} dataset'.format(DESC))

            npz_file = os.path.join(data_dir, '{}_{}_gps_fin.npz'.format(psr, freq)) # contains BE_mjds_pred, BE_res_pred, BE_vars_pred
            if not os.path.exists(npz_file):
                logger.warning("File containing eigenvalue GPs does not exist")
                continue
    
            with np.load(npz_file, allow_pickle=True) as f_npz:
                for key in f_npz.keys():
                    if BE in key:
                        var_dict[key] = f_npz[key]
            
            if BE+"_mjds_pred" not in var_dict.keys():
                logger.warning("No data found for "+BE)
                continue
    
            eig_mjds = var_dict[BE+'_mjds_pred'] # these MJDs were previously set using the nu-dot MJDs
            eig_vals = var_dict[BE+'_res_pred']
            eig_errs = var_dict[BE+'_vars_pred']

            if min(nudot_mjds) >= max(eig_mjds) or min(eig_mjds) >= max(nudot_mjds):
                logger.error("No overlap between timespans for "+DESC)
                continue
    
            logger.info("The nudot MJDs span {:.2f} to {:.2f}, and the {} profile MJDs span {:.2f} to {:.2f}"
                        .format(min(nudot_mjds), max(nudot_mjds), BE, min(eig_mjds), max(eig_mjds)))
            eig_lim = np.logical_and(eig_mjds > min(nudot_mjds)-1, eig_mjds < max(nudot_mjds)+1)
            nudot_lim = np.logical_and(nudot_mjds > min(eig_mjds)-1, nudot_mjds < max(eig_mjds)+1)

            if len(eig_mjds[eig_lim]) != len(nudot_mjds[nudot_lim]):
                logger.info("The max. and min. values of the nudot array and eigs array are {:.3f}, {:.3f}, {:.3f}, and {:.3f}, respectively"
                            .format(nudot_mjds[nudot_lim].min(), nudot_mjds[nudot_lim].max(), eig_mjds[eig_lim].min(), eig_mjds[eig_lim].max()))
                logger.info("The last few values in the eigenvalue MJD array are: {}".format(eig_mjds[eig_lim][-5:]))
                with plt.style.context(plot_style):
                    #plt.plot(eig_mjds[eig_lim], color=c2)
                    #plt.plot(nudot_mjds[nudot_lim], '--', color=c3)
                    #gap = eig_mjds[eig_lim][-20] - eig_mjds[eig_lim][20] - 5
                    plt.plot(eig_mjds[eig_lim][-20:], color=c2)
                    plt.plot(nudot_mjds[nudot_lim][-20:], color=c3)
                    plt.savefig(os.path.join(plots_dir, desc+'_mjds_comp.png'), bbox_inches='tight')
                    #plt.show()
                    
                logger.info("Comparison plot of the nudot and eigenvalue MJDs saved to "+os.path.join(plots_dir, desc+'_mjds_comp.png'))
                logger.error("The trimmed MJD arrays have different lengths: {} and {}".format(len(eig_mjds[eig_lim]), len(nudot_mjds[nudot_lim])))
                continue

            eig_sigs = np.array([np.max((eig_vals[num,:] - eig_vals[num,:].mean())/np.sqrt(eig_errs[num,:])) for num in range(eig_vals.shape[0])])
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
                logger.info("The correlation value for the {}{} eigenvector is {:.3f}".format(eignum, suff, res.correlation))

            # output the correlations in a simple format
            out_file = os.path.join(data_dir, desc+'_corrs.txt')
            with open(out_file, 'w') as f:
                f.write('# Component num. | Correlation value | Significance of Eigenvalue\n')
                for eignum, corr in enumerate(gp_corrs):
                    f.write('{}\t\t{}\t{}\n'.format(eignum, corr, eig_sigs[eignum]))

            corr_lim = np.abs(gp_corrs) > 0.3
            err_lim = eig_sigs > 1
            use_lim = np.logical_and(corr_lim, err_lim)

            if len(gp_corrs[use_lim]) == 0:
                logger.info("There are no significant correlated eigenvectors for {} with {} at {} MHz".format(psr, BE, freq))
            else:
                nudot_vars = nudot_errs[nudot_lim]**2 if nudot_errs is not None else None
                plot_eig_gp(eig_mjds[eig_lim], eig_vals[:,eig_lim][use_lim,:], eig_errs[:,eig_lim][use_lim,:], bk_bgd=use_bk_bgd, show=False,
                            nudot_mjds=nudot_mjds[nudot_lim], nudot_vals=nudot_vals[nudot_lim], nudot_vars=nudot_vars, eig_nums=np.arange(len(gp_corrs))[use_lim],
                            savename=os.path.join(plots_dir, desc+"_nudot_eigs_corr.png"))
                logger.info("Plot of the Nudot GP and correlated Eigenvalue GPs saved to "+os.path.join(plots_dir, desc+"_nudot_eigs_corr.png"))
            
            var_dict[BE+'_gp_corrs'] = gp_corrs
            var_dict[BE+'_corr_lim'] = use_lim
            plt.close('all')
            
        # combine the data from different backends into a single plot
        n_comp_sig = 0
        for BE in BE_list:
            n_comp_sig += len(var_dict[BE+'_gp_corrs'][var_dict[BE+'_corr_lim']]) if BE+"_gp_corrs" in var_dict.keys() else 0
            
        if n_comp_sig == 0:
            continue
            
        plot_colors_corrs(var_dict, nudot_mjds, nudot_vals, nudot_errs, bk_bgd=use_bk_bgd,
                         savename=os.path.join(plots_dir, psr+'_{}_combined_corr.png'.format(freq)), show=False, logg=logger)
        logger.info("Plot of the correlated eigenvalue series for all backends saved to "+os.path.join(plots_dir, psr+'_{}_combined_corr.png'.format(freq)))
        plt.close('all')

    # Do a Lomb-Scargle analysis on the nu-dot data as well.
    min_freq = 0.05/u.year # a period of 20 years
    max_freq = 4/u.year # a period of ~90 days
    nudot_errs = nudot_errs*u.Hz if nudot_errs is not None else None
    LS = LombScargle(nudot_mjds*u.day, nudot_vals*u.Hz, nudot_errs)
    freqs, power = LS.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=10)
    freq_max_power = freqs[power == np.max(power)][0]
    #print("The signal-to-noise ratio (significance) of the peak found is {:.1f}".format(find_eq_width_snr(power.value)[1]))
    logger.info("The frequency of the maximum power is {:.3f} ({}), which corresponds to a period of {:.2f}".format(freq_max_power.value, freq_max_power.unit, (1/freq_max_power).to('day')))
    with plt.style.context(plot_style):
        plt.clf()
        plt.plot(freqs, power, '-', color=c1)
        plt.ylabel('Periodogram Power')
        plt.xlabel('Frequency ({})'.format(freqs.unit))
        plt.xlim(min_freq.value, max_freq.value)
        plt.savefig(os.path.join(plots_dir, psr+'_nudot_LS.png'), bbox_inches='tight')
        #plt.show()
        
    logger.info("Plot of the nudot periodogram saved to "+os.path.join(plots_dir, psr+'_nudot_LS.png'))
        
    plt.close('all')
