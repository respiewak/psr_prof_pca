#!/usr/bin/env python
# coding: utf-8


import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.optimize as op
from scipy.stats import ks_2samp
import cmasher as cmr
sys.path.append('/home/s86932rs/research/psrcelery/')
import psrcelery
from all_prof_functions import (bin_array, get_rms_bline,# check_null_prob, 
                                calc_snr, _find_off_pulse,
                                err_eigval, err_eigval_off, find_dists_outliers, rolling_out_rej,
                                bad_mjds_eigs, setup_log)


# Set up some plotting stuff
plt.rc('savefig', bbox='tight')

# The following section contains variables describing the dataset and where the data are located
pars = ap.ArgumentParser(description='The second step')
pars.add_argument('psr_list', nargs='+', help="Pulsar names")
pars.add_argument('-f', '--frq_list', nargs='+', default=1400, type=int,
                  help="Frequency band(s) as they appear in the file names (integers)")
pars.add_argument('-b', '--be_list', nargs='+', default=['afb', 'dfb'],
                  help="Backend names/abbreviations")
pars.add_argument('-d', '--data_dir', default='../profiles/',
                  help="Absolute or relative path to the directory containing profile data")
pars.add_argument('-l', '--log_name', default='prof_pca',
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

            # read data from `step_1_clean_align`
            # files contain (BE_aligned, BE_mjds_new, BE_tobs, BE_template, BE_null_prob, BE_mjds_null)
            npz_file = os.path.join(data_dir, '{}_{}_arrs.npz'.format(psr, freq))
            with np.load(npz_file, allow_pickle=True) as d:
                if BE not in d.keys() and BE+'_aligned' in d.keys():
                    BE_aligned = d[BE+'_aligned']
                    if BE+'_temp' in d.keys():
                        BE_template = d[BE+'_temp']
                    elif BE+'_template' in d.keys():
                        BE_template = d[BE+'_template']
                    else:
                        raise(RuntimeError("Cannot find template array in data file"))
            
                    if BE+'_mjds_new' in d.keys():
                        BE_mjds_new = d[BE+'_mjds_new']
                    else:
                        BE_mjds_new = d[BE+'_mjds']
            
                    if BE+'_tobs' in d.keys():
                        BE_tobs = d[BE+'_tobs']
                                    
                else:
                    logger.warning("There are no {} data to analyse".format(DESC))
                    continue

            logger.info("The number of {} observations is {}, with {} bins, for a shape of {}.".format(BE, len(BE_mjds_new), len(BE_template), BE_aligned.shape))
            if len(BE_mjds_new) < 100:
                logger.warning("This dataset is too small for final purpose; skipping")
                be_exists.append(False)
                continue
            else:
                be_exists.append(True)

            # this should no longer be necessary
            if np.any(sorted(BE_mjds_new) != BE_mjds_new):
                logger.warning(BE+" MJDs are not sorted!!")
                sort_inds = np.argsort(BE_mjds_new)
                BE_mjds_new = BE_mjds_new[sort_inds]
                BE_aligned_new = BE_aligned[:,sort_inds]
                if BE_aligned_new.shape == BE_aligned.shape:
                    BE_aligned = BE_aligned_new
        
                if np.any(sorted(BE_mjds_new) != BE_mjds_new):
                    logger.warning("Sorting failed!!!!")
        
            # Mike's alignment function
            #plt.plot(BE_aligned[:,100])   
            BE_aligned = psrcelery.data.align_and_scale(BE_aligned.T, BE_template, nharm='auto').T
            #plt.plot(BE_aligned[:,100])   
            #plt.show()

            BE_off, _ = _find_off_pulse(BE_template)
            BE_offrms = np.std(BE_aligned[BE_off,:], axis=0)
            BE_aligned = (BE_aligned.T - BE_template).T/BE_offrms
            #plt.plot(BE_aligned[:,100])   
            #plt.show()
            #plt.imshow(BE_aligned.T, aspect='auto')
            #plt.show()

            # Try to set the phase cuts automatically
            nbin = len(BE_template)
            phase = np.linspace(0, 1, nbin)

            # define on-pulse ranges as fractions
            ip_exist = len(phase[phase > 0.65]) != len(phase[BE_off][phase[BE_off] > 0.65]) # all points near IP are "off-pulse"
            one_bin = 1/nbin
            peak_min = np.max(phase[BE_off][phase[BE_off] < 0.25])-2*one_bin
            peak_max = np.min(phase[BE_off][phase[BE_off] > 0.25])+2*one_bin
            off_min = peak_min - min(peak_min/2, 0.03)
            off_max = min(2*peak_max - peak_min, 0.7)
            if ip_exist:
                ip_min = np.max(phase[BE_off][phase[BE_off] < 0.75])-2*one_bin
                ip_max = np.min(phase[BE_off][phase[BE_off] > 0.75])+2*one_bin

            # plot the templates and define some useful values
            with plt.style.context(plot_style):
                plt.clf()
                fig = plt.figure(num=1)
                fig.set_size_inches(14, 4)

                plt.title("{}, {}, {}, template".format(psr, BE, freq))
                BE_nbin = len(BE_template)
                plt.plot(np.linspace(0, 1, BE_nbin), BE_template, color=c2)

                ylims = plt.ylim()
                plt.vlines([off_min, peak_min, peak_max, off_max], ylims[0], ylims[1], linestyle='dashed', colors='grey')
                if ip_exist:
                    plt.vlines([ip_mean, ip_max], ylims[0], ylims[1], linestyle='dashed', colors='grey')
        
                plt.ylim(ylims)
                plt.xticks(np.linspace(0, 1, 21))
                plt.xlim(0, 1)
                plt.ylabel('Normalised intensity', fontsize=12)
                plt.xlabel('Phase (turns)', fontsize=12)
                plt.savefig(os.path.join(plots_dir, desc+'_template.png'), bbox_inches='tight')
                logger.info('Template plot saved to '+os.path.join(plots_dir, desc+'_template.png'))
                #plt.show()

            # We first need to fit the profiles to get eigenvalues and eigenvectors. The eigenvectors describe the profiles, including bin-wise dependence. The eigenvalues describe the variation between profiles. Once we have the eigenvalues, we can fit GPs, and then find correlations. 
            BE_bins = np.linspace(0, 1, num=BE_aligned.shape[0], endpoint=False)
            BE_mask = np.logical_and(BE_bins > off_min, BE_bins < off_max)
            BE_off = np.logical_or(BE_bins[BE_mask] < peak_min, BE_bins[BE_mask] > peak_max)
            if ip_exist:
                BE_mask = np.logical_or(BE_mask, np.logical_and(BE_bins > ip_min, BE_bins < ip_max))
                BE_off = np.logical_or(BE_bins[BE_mask] < peak_min, np.logical_and(BE_bins[BE_mask] > peak_max, BE_bins[BE_mask] < off_max))
    
            #BE_range = (int(peak_min*BE_nbin), int(peak_max*BE_nbin))
            #BE_range = (0, BE_aligned.shape[0])
            BE_pca = PCA(n_components=30)
            BE_comps_all = BE_pca.fit_transform(BE_aligned[BE_mask,:].T) * BE_offrms.reshape(-1,1)

            logger.info("Check that these are number of profiles by number of components:")
            logger.info(BE+": {}".format(BE_comps_all.shape))
    
            logger.info("Check that these are number of components by number of bins (used):")
            logger.info(BE+": {}".format(BE_pca.components_.shape))

            BE_rms, _ = get_rms_bline(BE_aligned)
            #print(len(BE_rms))
            logger.info("The (max, median, and min) off-pulse rms for {} are ({:.5f}, {:.5f}, {:.5f})".format(BE, BE_rms.min(), np.median(BE_rms), BE_rms.max()))
    
            bad_mjds_be = bad_mjds_eigs(BE_aligned, BE_mjds_new, peak_min, peak_max)

            #print(len(bad_mjds_be))
            #imjd = np.random.randint(len(bad_mjds_be))
            #bad_mjd = bad_mjds_be[imjd]
            #lim = BE_mjds_new == bad_mjd
            #print(bad_mjd)
            #with plt.style.context(plot_style):
            #    plt.plot(BE_bins, BE_aligned[:,imjd], color=c2)
            #    plt.plot(BE_bins[BE_mask], BE_pca.mean_, '-', color=c1)
            #    #plt.show()

            # define axes parameters for following plots
            #w = 0.92
            l1 = 0.1
            b = 0.1
            h = 0.82
            sep = 0.08
            frac = 0.5
            l2 = l1 + frac + sep
            w1 = frac
            w2 = 1 - l2 - sep

            with plt.style.context(plot_style):
                plt.clf()
                fig = plt.figure(num=1)
                fig.set_size_inches(7, 4)
                fig.suptitle("{}, {}, example profiles and mean".format(psr, BE))
                first, second = np.random.randint(BE_comps_all.shape[0], size=2)
                if ip_exist:
                    ax1 = fig.add_axes((l1, b, w1, h))
                    ax2 = fig.add_axes((l2, b, w2, h))
                    mask1 = np.logical_and(BE_mask, BE_bins < 0.5)
                    mask2 = np.logical_and(BE_mask, BE_bins > 0.5)
                    mean_mask1 = np.arange(len(BE_pca.mean_)) < len(BE_bins[mask1])
                    mean_mask2 = np.arange(len(BE_pca.mean_)) >= len(BE_bins[mask1])
                    for ax, mask, mmask in zip([ax1, ax2], [mask1, mask2], [mean_mask1, mean_mask2]):
                        ax.plot(BE_bins[mask], BE_aligned[mask,first], color=c2, label='Obs. No. {}'.format(first)) #two example profiles of the two states
                        ax.plot(BE_bins[mask], BE_aligned[mask,second], color=c3, ls='--', label='Obs. No. {}'.format(second))
                        ax.plot(BE_bins[mask], BE_pca.mean_[mmask], color=c1) #mean plotted, subtracted off before computing
                        ax.set_xlabel('Phase (turns)', fontsize=12)
        
                    ax = ax1

                else:
                    ax = plt.gca()
                    plt.plot(BE_bins[BE_mask], BE_aligned[BE_mask,first], color=c2, label='Obs. No. {}'.format(first)) #two example profiles of the two states
                    plt.plot(BE_bins[BE_mask], BE_aligned[BE_mask,second], color=c3, ls='--', label='Obs. No. {}'.format(second))
                    plt.plot(BE_bins[BE_mask], BE_pca.mean_, color=c1) #mean plotted, subtracted off before computing
                    plt.xlabel('Phase (turns)', fontsize=12)
    
                plt.text(0.78, 0.87, '$\sigma_{{{:d}}} = {:.4f}$\n$\sigma_{{{:d}}} = {:.4f}$'.format(first, BE_rms[first], second, BE_rms[second]), transform=ax.transAxes)
                ax.set_ylabel('Intensity (normalised to peak)', fontsize=12)
                plt.legend(loc=2)
                plt.savefig(os.path.join(plots_dir, desc+'_exmp_profs.png'), bbox_inches='tight')
                logger.info('Example profiles plot saved to '+os.path.join(plots_dir, desc+'_exmp_profs.png'))
                #plt.show()

            # plot the first five principal components
            with plt.style.context(plot_style):
                plt.clf()
                plt.title("{}, {}, mean and first 6 eigenvectors".format(psr, BE))
                plt.plot(0.5*(BE_pca.mean_/BE_pca.mean_.max())+0.4, color='grey', ls='--')
                plt.plot(BE_pca.components_[0,:], color=c1)
                plt.plot(BE_pca.components_[1,:]-0.4, color=c2)
                plt.plot(BE_pca.components_[2,:]-0.8, color=c3)
                plt.plot(BE_pca.components_[3,:]-1.2, color=c4)
                plt.plot(BE_pca.components_[4,:]-1.6, color=c5)
                plt.plot(BE_pca.components_[5,:]-2, color=c6)
                plt.ylabel("Relative Intensity", fontsize=12)
                plt.xlabel("Phase (bins)", fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, '{}_components.png'.format(desc)), bbox_inches='tight')
                logger.info('Plot of eigenvectors saved to '+os.path.join(plots_dir, '{}_components.png'.format(desc)))
                #plt.show()

            # explained ratio, the variance of each component, and can compute a cumulative sum
            with plt.style.context(plot_style):
                plt.clf()
                plt.title("{}, {}, explained variance".format(psr, BE))
                plt.bar(np.arange(1, BE_comps_all.shape[1]+1), BE_pca.explained_variance_ratio_, color='None', edgecolor='k', tick_label=np.arange(1, BE_comps_all.shape[1]+1))
                plt.plot(np.arange(1, BE_comps_all.shape[1]+1), np.cumsum(BE_pca.explained_variance_ratio_[:BE_comps_all.shape[1]+1]), marker='o', color=c2)
                plt.ylim(0, 1)
                plt.savefig(os.path.join(plots_dir, '{}_variance.png'.format(desc)), bbox_inches='tight')
                logger.info('Plot of explained variance saved to '+os.path.join(plots_dir, '{}_variance.png'.format(desc)))
                #plt.show()

            BE_errs_new = err_eigval(BE_aligned[BE_mask,:], BE_pca.components_, BE_off) * BE_offrms.reshape(-1,1)

            plt.clf()
            BE_mjds_out1 = find_dists_outliers(BE_comps_all, BE_mjds_new, psr, BE, 6, savename=os.path.join(plots_dir, desc+"_eigval_dists.png"),
                                               first_out=True, sigma=5, show=False, bk_bgd=use_bk_bgd, logg=logger)
            logger.info("Plot of eigenvalue distributions and outliers saved to "+os.path.join(plots_dir, desc+"_eigval_dists.png"))
            BE_mjds_out2 = rolling_out_rej(BE_comps_all, BE_mjds_new, psr, BE, 6, first_out=True, show=False, bk_bgd=use_bk_bgd, logg=logger)
            #print(BE_mjds_out1)

            with plt.style.context(plot_style):
                plt.clf()
                fig = plt.figure(figsize=(20, 5))
                fig.suptitle("{}, {}, {}, eigenvalues".format(psr, BE, freq))
                ax1 = fig.gca()
                ax1.errorbar(BE_mjds_new, BE_comps_all[:,3], yerr=BE_errs_new[:,3], fmt='o', ecolor=c4, mec=c4, mfc=c4)
                ax1.errorbar(BE_mjds_new, BE_comps_all[:,2], yerr=BE_errs_new[:,2], fmt='v', ecolor=c3, mec=c3, mfc=c3)
                ax1.errorbar(BE_mjds_new, BE_comps_all[:,1], yerr=BE_errs_new[:,1], fmt='s', ecolor=c2, mec=c2, mfc=c2)
                ax1.errorbar(BE_mjds_new, BE_comps_all[:,0], yerr=BE_errs_new[:,0], fmt='*', ecolor=c1, mec=c1, mfc=c1, ms=9)
                ax1.set_xlabel('MJD (days)', fontsize=12)
                ax1.set_ylabel('Value', fontsize=12)
                vlim = ax1.get_ylim()
                ax1.vlines(BE_mjds_out1, vlim[0], vlim[1], ls='--', color=c6, zorder=1)
                ax1.set_ylim(vlim)
                plt.savefig(os.path.join(plots_dir, "{}_eigs_v_mjd.png".format(desc)), bbox_inches='tight')
                logger.info('Plot of eigenvalue timeseries saved to '+os.path.join(plots_dir, "{}_eigs_v_mjd.png".format(desc)))
                #plt.show()

            #lim = BE_mjds_new > 38000
            #imjd = np.argmax(BE_comps_all[:,2][lim])
            #bad_mjd = BE_mjds_new[lim][imjd]
            ##print(bad_mjd)
            #with plt.style.context(plot_style):
            #    plt.clf()
            #    plt.plot(BE_bins, BE_aligned[:,imjd], color=c2)
            #    plt.plot(BE_bins[BE_mask], BE_pca.mean_*BE_aligned[:,imjd].max()/BE_pca.mean_.max(), '--', color=c1)
            #    #plt.show()

            plt.close('all')
            # Save the eigenvectors and eigenvalues
            var_dict[BE+'_errs'] = BE_errs_new
            var_dict[BE+'_mean'] = BE_pca.mean_
            var_dict[BE+'_values'] = BE_comps_all
            var_dict[BE+'_vectors'] = BE_pca.components_
            var_dict[BE+'_mjds'] = BE_mjds_new
            var_dict[BE+'_rms'] = BE_rms

        out_file = os.path.join(data_dir, '{}_{}_eigs.npz'.format(psr, freq))
        if np.any(np.array(be_exists)):
            old_dict = {}
            if os.path.exists(out_file):
                with np.load(out_file, allow_pickle=True) as f:
                    for key in f.keys():
                        if key not in var_dict.keys():
                            old_dict[key] = f[key]
                        else:
                            logger.info("Replacing an older value for "+key)
                
            np.savez(out_file, **var_dict, **old_dict)
        elif os.path.exists(out_file):
            logger.warning("No dataset to save but removing an old npz file:", out_file)
            os.remove(out_file)
