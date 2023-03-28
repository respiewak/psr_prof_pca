#!/usr/bin/env python
# coding: utf-8

# # Test and Tutorial Notebook for Profile Variation Analysis
# _Author: Renee Spiewak_
# 
# This notebook will demonstrate the key steps of the PCA+GP profile variation analysis through use of a simulated dataset. To simply test the functions, hit the fast-forward button to run all cells and scroll through to check for errors. 
# 
# 
# ## Outline
# 
# 1. Producing the dataset - A set of ''observations'' with varying noise (white noise only for now), a phase drift over time, and various types of profile variation are generated automatically. If desired, the ''average profile'' can be generated first and used to produce the ''observations''. 
# 
# 2. Cleaning and aligning the dataset - Following the exact procedure used for real data, the simulation dataset is cleaned and aligned. The second ''Joy Division''-style waterfall plot should be examined to ensure the alignment of all ''observations''. 
# 
# 3. Running the PCA - The aligned profiles are passed through a standard PCA tool to extract eigenvectors and eigenvalues. The plot of eigenvalues (for the most significant eigenvectors) over time should be examined for outliers. (Mis-aligned ''observations'' will result in an extraneous eigenvector.) 
# 
# 4. Running the GPs - The cleaned eigenvalues are passed to a function to determine the GPs using MCMC. For each eigenvalue, a plot of the chain and ''corner'' plot are generated; for this fairly ideal dataset, these should be clean and smooth (e.g., roughly Gaussian distributions in the corner plots). The final best GPs are plotted on top of the eigenvalues for each eigenvector in a summary plot, and the ''reconstructed'' profiles (with the mean subtracted) are shown in a waterfall plot. 



import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col
import cmasher as cmr
from sklearn.decomposition import PCA
import corner
import george
from george import kernels
import emcee
import scipy.optimize as op
from scipy import stats
from astropy.timeseries import LombScargle
import astropy.units as u
import multiprocessing as mpr
mpr.set_start_method('fork')
import argparse as ap
sys.path.append('/home/s86932rs/research/psrcelery/')
import psrcelery
from all_prof_functions import (aligndata, smart_align, calc_snr, plot_joydivision, make_fake_obss,
                                make_fake_profile, add_gauss, do_rem_aln,
                                read_pdv, _find_off_pulse, err_eigval, find_eq_width_snr,
                                get_gp, run_each_gp, plot_eig_gp, plot_recon_profs)
    

pars = ap.ArgumentParser(description='Test suite for the PCA+GP profile variation pipeline')
pars.add_argument('-b', '--bk_bgd', action='store_true',
                  help="Use a dark background for all plots")
pars.add_argument('-d', '--plot_dir', default='./test_plots',
                  help="The directory for outputing plots; created if DNE")

args = vars(pars.parse_args())

# Set up some plotting stuff
plt.rcParams["figure.figsize"] = (6, 10)
plt.rc('savefig', bbox='tight')
use_bk_bgd = args['bk_bgd']
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
    
plot_dir = args['plot_dir']
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# ## Simulating a dataset
# 
# The real data you will use for the profile variation analysis will contain profiles in a 2D numpy array (phase bins vs. observation number), MJDs in a 1D array, and observation lengths in a 1D array (not used in this version of the analysis pipeline). In order to test the analysis functions, these 3 data products must be simulated. 
# 
# 
# ### Generating a fake average profile
# 
# The ''profiles'' are generated based on an input profile or by adding a number of Gaussian components with white noise. For >1 Gaussian components, the centre bin, width, and amplitude are adjusted using normally- or lognormally-distributed random numbers. For >2 Gaussian components: If `no_ip` is set to False, at least one component will be shifted by `nbin/2` bins, simulating an interpulse. If `no_ip` is set to True, no components will be shifted to form an interpulse (although the normal distribution of phase shifts allows for significant separation of components). 
# 
# If you later see that the alignment of profiles is failing, I recommend generating a profile with fewer components, or where the components significantly overlap (i.e., the separation of centre bins is less than the widths). 


# The function to generate an ''average'' profile
#get_ipython().run_line_magic('pinfo', 'make_fake_profile')

# generate a normal noisy profile but save the parameters to make a noiseless version later
prof, cens, wids, hits = make_fake_profile(4, 0.01, no_ip=True, not_wide=True)

with plt.style.context(plot_style):
    plt.clf()
    fig = plt.figure(num=1)
    fig.set_size_inches(7, 5.5)
    plt.plot(prof)
    plt.xlim(0, len(prof))
    _ = plt.xticks([i*50 for i in range(1+int(np.floor(len(prof)/50)))])
    plt.xlabel('Phase bins')
    plt.ylabel('Amplitude')
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'in_avg_prof.png'))

# this cell just demonstrates the method used to find the width of the profile
width, snr = find_eq_width_snr(prof, verb=False, plot_style=plot_style)
print("The equivalent width is {:d} bins, and the S/N is {:.3f}".format(width, snr))

# ### Simulating a realistic set of observations
# 
# The following function attempts to simulate a dataset that is realistic enough to test the PCA and GP parts of this analysis. For each ''observation'':
# 0. An array of sorted MJD values is generated to represent observation epochs such that `max(mjds) - min(mjds) > nobs*2`, with each successive value separated by at least 0.9 days but no more than `nobs/20`. 
# 
# 1. A known ''average profile'' can be given (as a numpy array of floats), or a profile will be generated as above with the given number of components and phase bins. WIP: If an ''average profile'' is provided rather than generated inside the function, the shape is treated as single Gaussian for the determination of Gaussian parameters for the next step. 
# 
# 2. If shape variations is desired: Using the ''average profile'' as a starting point, between 1 and 3 ''eigenvectors'' are produced using a single Gaussian component each, with parameters based on those of the ''average profile'' with random variations. The ''eigenvalues'' are generated for each ''observation'' based on the chosen (or randomly selected) pattern:
#     1. Quasi-periodic variations (`qp`) - an offset sinusoid with a timescale between `mjd_range/30` and `mjd_range/10` (i.e., so that between 10 and 30 periods are ''observed'') is used as the base, upon which random variations in both time and amplitude are applied. 
#     
#     2. Bimodal variations (standard; `bi`) - a boxcar-like function with a timescale between `mjd_range/30` and `mjd_range/5` is used as the base, with the amplitude alternating between 0 and 1 (normal distributions around those values). 
#     
#     3. Bimodal variations (quick; `bi_quick`) - like the above but with a shorter timescale, chosen from a lognormal distribution with a median value of twice the mean lag between MJDs
#     
#     4. Random variations (`random`) - with no timescale set, the amplitude of the eigenvalues are simply drawn from a offset lognormal distribution with a median value around 1 (minimum value less than 0, maximum greater than 10)
#     
# 3. An array of ''observation lengths'' is generated using lognormal distribution with a median value of 900s (increased to 3600s for 30\% of ''observations''). The `tobs` values are used to vary the white noise level of the ''observation'', multiplying by `sqrt(tobs_mean/tobs)`. 
# 
# 4. If nulling is desired: The timescales given (in days) are used to determine whether the ''pulsar'' switches between states, completely independently from any shape variation (i.e., if the ''pulsar'' is ''off'', the output is simply the noise as normal; when ''on'', the full profile with any shape variation is returned). 
# 
# 5. If desired, a random, small phase shift can be added for successive ''observations'', representing an increasingly incorrent phase solution. This is useful for testing the alignment function. 
# 

# using a newly-generated average profile with nulling, and no shape variation or phase misalignment
fakedata, fake_mjds, fake_tobs = make_fake_obss(avg_shape=3, nobs=1000, nbin=512, shape_change=False,
                                                null_time=10, on_time=20, # these values are in days
                                                verb=True, plot_style=plot_style, no_misalign=True,
                                                show_plot=False)

# using a previously generated profile (noiseless) with with quasi-periodic shape variation and phase misalignment, and no nulling
prof_noiseless = np.zeros(512)
for icomp in range(len(cens)):
    prof_noiseless = add_gauss(prof_noiseless, cens[icomp], wids[icomp], hits[icomp])

fakedata, fake_mjds, fake_tobs = make_fake_obss(avg_shape=prof_noiseless, nobs=1000, shape_change='qp', null_time=None,
                                                verb=True, plot_style=plot_style, no_misalign=False,# strong=True,
                                                show_plot=False, save_plot=os.path.join(plot_dir, 'input_eigvecs.png'))
print("Using the dataset with shape variations")

# plot this dataset
plot_joydivision(fakedata, 'test', savename=os.path.join(plot_dir, 'joydiv_raw.png'), bk_bgd=use_bk_bgd, show=False)

# plot the actual average profile
with plt.style.context(plot_style):
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(7, 5.5)
    plt.plot(np.mean(fakedata, axis=1))
    plt.ylabel('Amplitude')
    plt.xlabel('Phase bins')
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'raw_avg_prof.png'))

# ## Cleaning and aligning profiles
# 
# The cleaning of profiles is done in 3 stages. These, plus the alignment function and normalisation, are performed in a pipeline function `do_rem_aln()`. 
# 
# The cleaning stages:
# 1. Cleaning based on statistics, using `rem_base_outs()` - 3 checks are done on the data, using the off-pulse rms, pulse width, and S/N. For each, a threshold is applied in terms of standard deviations. 
#     1. For the off-pulse rms, the cut is `all_rms < rms_med + all_rms.std()*threshold` where `rms_med` is the median off-pulse rms. Note that alignment of the profiles is not guaranteed at this stage, so the rms in several phase windows is compared and the minimum used for this cut. 
#     
#     1. For pulse width, the cut is `abs(all_wid - wid_men) < wid_thrsh*wid_std`  where `wid_men` is the mean width and `wid_thrsh` is `3.5*threshold` for a broad distribution of pulse widths and `2*threshold` otherwise. 
#     
#     2. Optional: The cut on S/N is `all_snr > max(snr_med - all_snr[good_snr].std()*threshold, 5)`, where `snr_med` is the median S/N (after trimming failed S/N calculations with the `good_snr` mask), and a standard minimum threshold of S/N \> 5 is applied. 
#     
# 2. A list of known ''bad'' observations can be given using the `bad_mjds` keyword (a list of MJDs); observations with these MJDs are removed after stats-based cleaning. 
# 
# 3. After the profiles are aligned as described below, a more accurate determination of the off-pulse rms is made, with `rem_extra_noisy()`. For each profile, an iterative function to find the on- and off-pulse regions is run, and the `True` (off-pulse) values are converted to 1s (0s for `False`). These arrays are summed to form an array of length `nbin` containing values from 0 to `nobs`. Bins with values \> `0.6*nobs` are considered the off-pulse region, making a new mask to apply to the profiles. The rms of this region for all profiles is then calculated, and the cut from step 1.A. applied with these values. 
# 
# The profiles are aligned using a correlation between a template (the brightest individual profile) and each individual profile (`xcorr = np.correlate(template, obs, "full")`). After this process, a new template is generated (the average profile) and the process repeated, twice. (WIP: if a better template could be provided, some alignment issues with multi-peak profiles could be fixed.) 
# 
# Finally, the profiles are normalised by the sum of on-pulse bins (using the aforementioned iterative function to find the off-pulse region).
#

# run the fake data through the whole cleaning function
fake_aligned, fake_template, fake_mjds_new, _ = do_rem_aln(fakedata, fake_mjds, fake_tobs, thrsh=1.75,
                                                           cut_snr=False, bad_mjds=None, wide=True, quiet=False)
plot_joydivision(fake_aligned, 'test_aligned', savename=os.path.join(plot_dir, 'joydiv_clean.png'),
                 bk_bgd=use_bk_bgd, show=False)

# ## Running the Principal Component Analysis
# 
# ### Caveats
# 
# PCA is sensitive to misalignment of profiles. Any misaligned profiles will result in one or more extraneous eigenvectors, skewing the analysis and producing outlier eigenvalues, which could cause the GP to fail. You should always check the waterfall plots for misaligned profiles, and then check the eigenvalue plots for significant outliers. The MJDs for any outliers can be given to the `do_rem_aln` function for removal.
# 
# Unless your pulse profile is very wide, it is unwise to include the entire rotation in the PCA as it will look for patterns in the off-pulse which are not relevant to the analysis. Find the rough on-pulse region and make a bin-wise mask (array of booleans). If the profile has an interpulse, that region should be included as well, cutting out any off-pulse region between the MP and IP.

# refine the alignment to better than a bin using an FFT technique
fake_aligned = psrcelery.data.align_and_scale(fake_aligned.T, fake_template, nharm='auto').T

fake_off, _ = _find_off_pulse(fake_template)
fake_offrms = np.std(fake_aligned[fake_off,:], axis=0)
fake_aligned = (fake_aligned.T - fake_template).T/fake_offrms
with plt.style.context(plot_style):
    plt.imshow(fake_aligned.T, aspect='auto')
    plt.ylabel('Observation num.')
    plt.xlabel('Phase Bin')
    plt.title('Waterfall after subtraction of mean and normalisation by off-pulse rms')
    plt.savefig(os.path.join(plot_dir, 'subd_normd_wfall.png'), bbox_inches='tight')
    #plt.show()

lim, _ = _find_off_pulse(fake_template)
test_nbin = len(fake_template)
phase = np.linspace(0, 1, test_nbin)

# define on-pulse ranges as fractions
one_bin = 1/test_nbin
shift_centre = 0
if len(phase[lim][phase[lim] < 0.25]) == 0:
    print("Need to roll the array to centre the peak at phase=0.5")
    fake_template = np.roll(fake_template, test_nbin//4)
    fake_aligned = np.roll(fake_aligned, test_nbin//4, axis=0)
    lim, _ = _find_off_pulse(fake_template)
    shift_centre = 0.25

peak_min = np.max(phase[lim][phase[lim] < 0.25+shift_centre])-2*one_bin
peak_max = np.min(phase[lim][phase[lim] > 0.25+shift_centre])+2*one_bin
off_min = peak_min - min(peak_min/2, 0.05)
off_max = min(2*peak_max - peak_min, 0.7+shift_centre)

# plot the template to find the on-pulse region
with plt.style.context(plot_style):
    fig = plt.figure(num=1)
    fig.set_size_inches(14, 4)
    ax = fig.gca()
    plt.title("test template(s)")
    plt.plot(phase, fake_template, color=c2)
    ylims = plt.ylim()
    plt.vlines([off_min, peak_min, peak_max, off_max], ylims[0], ylims[1], linestyle='dashed', colors='grey')
    plt.ylim(ylims)
    plt.xticks(np.linspace(0, 1, 21))
    plt.xlim(0, 1)
    plt.ylabel('Normalised intensity', fontsize=12)
    plt.xlabel('Phase (turns)', fontsize=12)
    plt.text(0.75, 0.87, 'The cuts are: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(off_min, peak_min, peak_max, off_max), transform=ax.transAxes)
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'template.png'))

test_bins = np.linspace(0, 1, num=fake_aligned.shape[0], endpoint=False)
test_mask = np.logical_and(test_bins > off_min, test_bins < off_max)
test_off = np.logical_or(test_bins[test_mask] < peak_min, test_bins[test_mask] > peak_max)

# With the on- and off-pulse regions defined, you can now use the PCA class from scikit-learn to find the 30 most significant eigenvectors and associated eigenvalues. (You can change the maximum number of components retained, but 30 should work well for most cases.)

# The test_pca object contains many useful attributes, such as the eigenvectors (test_pca.components_) and mean profile (test_pca.mean_), and the eigenvalues are returned to a separate variable by the .fit_transform() function. The shape of the test_comps_all numpy array is (N_obs, N_comp), and the shape of test_pca.components_ is (N_comp, N_bin), where N_comp in this case is 30, and N_bin is the number of bins remaining in the masked part of the profiles.

# test the PCA stuff
test_pca = PCA(n_components=30)
test_comps_all = test_pca.fit_transform(fake_aligned[test_mask,:].T) * fake_offrms.reshape(-1,1)

# plot the first six principal components
with plt.style.context(plot_style):
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(6, 5.5)
    plt.title("test, mean and first 6 eigenvectors")
    plt.plot(0.5*(test_pca.mean_/np.max(test_pca.mean_))+0.4, '--', color='grey')
    plt.plot(test_pca.components_[0,:], color=c1)
    plt.plot(test_pca.components_[1,:]-0.4, color=c2)
    plt.plot(test_pca.components_[2,:]-0.8, color=c3)
    plt.plot(test_pca.components_[3,:]-1.2, color=c4)
    plt.plot(test_pca.components_[4,:]-1.6, color=c5)
    plt.plot(test_pca.components_[5,:]-2, color=c6)

    plt.xlabel('Phase bins')
    plt.ylabel('Arbitrary amplitude')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'test_eigvecs.png'))

# The uncertainties on the eigenvalues are very useful for identifying outliers, and for accurate GP results. The function to calculate these (using error propagation) requires the full (masked) dataset, the components found by PCA, and the portion of the masked region that is the off-pulse region (as seen above, some off-pulse is included for this reason).

new_errs = err_eigval(fake_aligned[test_mask,:], test_pca.components_, test_off) * fake_offrms.reshape(-1,1)

with plt.style.context(plot_style):
    plt.clf()
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle("test, eigenvalues")
    ax1 = fig.gca()
    ax1.errorbar(fake_mjds_new, test_comps_all[:,3], yerr=new_errs[:,3], fmt='o', ecolor=c4, mec=c4, mfc=c4)
    ax1.errorbar(fake_mjds_new, test_comps_all[:,2], yerr=new_errs[:,2], fmt='v', ecolor=c3, mec=c3, mfc=c3)
    ax1.errorbar(fake_mjds_new, test_comps_all[:,1], yerr=new_errs[:,1], fmt='s', ecolor=c2, mec=c2, mfc=c2)
    ax1.errorbar(fake_mjds_new, test_comps_all[:,0], yerr=new_errs[:,0], fmt='*', ecolor=c1, mec=c1, mfc=c1, ms=9)
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'eigvals_vs_mjd.png'))

# ## Running the Gaussian Process with Markov Chain Monte Carlo
# The final step of the analysis (at this time) is to run GPs on the eigenvalues for the most significant components identified by PCA. This is done with the celerite package, using MCMC (implemented by emcee) to optimise the parameters.
#
# The kernel used in this analysis is the Matern-3/2, which only has the amplitude and length scale as independent variables. (There are also a constant and white noise term which will be optimised by the MCMC.) The bounds on the length scale can be derived from the input data: the minimum bound is set to the 99.5th percentile of the separation of observations (so that the GP will not latch onto inter-observation variations), and the maximum is set to half the observed timespan.
#
# The calculations inside celerite are performed in log-space, so the chains and distributions plotted for each component must be interpreted as such.
#
# Once the GPs have been run for your selected number of significant components (typically between 2 and 5), a plot will be made showing, for each component, the input eigenvalues over time overlaid with the predictions using the GP median values with uncertainties. This plot is valuable for determining if the GP has succeeded in modeling the eigenvalues for each component.
#

# plot the distribution of lags to ensure the minimum bound on the kernel length is appropriate
with plt.style.context(plot_style):
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(6, 5)
    lags = fake_mjds_new[1:]-fake_mjds_new[:-1]
    _ = plt.hist(lags, bins=100)
    plt.ylabel('Count')
    plt.xlabel('Lag (days)')
    print("The mean lag between observation epochs is {:.2f}".format(np.mean(lags)))
    #plt.show()
    plt.savefig(os.path.join(plot_dir, 'lag_hist.png'))
    
pmin = np.percentile(lags, 99.5)
mjd_range = fake_mjds_new[-1] - fake_mjds_new[0]

pred_res, pred_vars, mjds_pred = run_each_gp(test_comps_all, fake_mjds_new, new_errs, kern_len=50, max_num=4,
                                             prior_min=pmin**2, prior_max=(0.5*mjd_range)**2, long=True, plot_chains=True,
                                             plot_corner=True, plot_gps=True, mcmc=True, multi=True, verb=True,
                                             bk_bgd=use_bk_bgd, plot_dir=plot_dir, show_plots=False,
                                             gp_plotname=os.path.join(plot_dir, 'gp_eigs_res.png'))

# Finally, once the GPs are completed and you have the predicted (smoothed) eigenvalues over time, you can reconstruct the profile variations by simply summing the products of the eigenvectors with their eigenvalues over time. While this plot does not show uncertainties on the variations, it is a visually appealing way to examine the profile variations.

plot_recon_profs(test_pca.mean_, test_pca.components_, mjds_pred, pred_res, 'test', mjds_real=fake_mjds_new,
                 sub_mean=True, savename=os.path.join(plot_dir, 'recon_from_gps.png'), bk_bgd=use_bk_bgd)

# The last step of the real analysis is to look at the statistics, whether or not there is any significant correlation between the eigenvalue timeseries and the nu-dot timeseries. Since we simply need to show the use of these functions for this tutorial, we will replace the nu-dot timeseries with random numbers, or values derived from the eigenvalue timeseries themselves (to ensure correlation). 

# make your fake nu-dot timeseries
nudot_mjds = mjds_pred
#nudot_vals = np.random.normal(size=len(nudot_mjds)) # Gaussian noise, should not show correlation
nudot_vals = np.random.normal(0, pred_res[0,:].std()/3, size=len(nudot_mjds))+pred_res[0,:] # not random, based on most significant eigenvalue timeseries
nudot_lim = np.array([True for A in nudot_mjds])

# pred_res, pred_vars, mjds_pred
eig_vals = pred_res
eig_mjds = mjds_pred
eig_lim = np.array([True for A in mjds_pred])

# use the Spearman Rank Correlation test on the eigenvalue timeseries
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

# Finally, do a LombScargle analysis to check for periodicity in the nu-dot timeseries. As with the correlation test, this should find nothing of significance using the Gaussian nu-dot timeseries, but it should find some periodicity for a nu-dot timeseries based on the most significant eigenvalue (since we have simulated a dataset with quasi-periodic variation). The significance of the value identified is estimated by the signal-to-noise ratio of the array of ''power'' from the LombScargle, but this is a very rough indicator. 

min_freq = 0.025/u.year # a period of 40 years
max_freq = 4/u.year # a period of ~90 days
nudot_errs = None # we could include uncertainties on nu-dot if we had them
LS = LombScargle(nudot_mjds*u.day, nudot_vals*u.Hz, nudot_errs)
freqs, power = LS.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=10)
freq_max_power = freqs[power == np.max(power)][0]
print("The signal-to-noise ratio (significance) of the peak found is {:.1f}".format(find_eq_width_snr(power.value)[1]))
print("The frequency of the maximum power is {:.4f} ({})".format(freq_max_power.value, freq_max_power.unit))
print("That corresponds to a period of {:.1f}".format((1/freq_max_power).to('day')))
with plt.style.context(plot_style):
    plt.clf()
    plt.plot(freqs, power, '-', color=c1)
    plt.ylabel('Periodogram Power')
    plt.xlabel('Frequency ({})'.format(freqs.unit))
    plt.xlim(min_freq.value, max_freq.value)
    plt.savefig(os.path.join(plots_dir, 'test_nudot_LS.png'), bbox_inches='tight')
    #plt.show()
