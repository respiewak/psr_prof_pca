##  Author: Renee Spiewak


import os, sys
import numpy as np
import importlib as implib
from matplotlib import pyplot as plt
from matplotlib import colors as col
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import george
import celerite as cel
from george import kernels
import emcee
import corner
import scipy.optimize as op
import multiprocessing as mpr
#import gp_init_threads as th_init
import cmasher as cmr
import logging
sys.path.append('/home/s86932rs/research/psrcelery/')
import psrcelery

## fix these lines to use the nulling analysis functions
if os.path.exists('../nulling2'):
    sys.path.append('../nulling2/')
    import null2_mcmc as nm
else:
    try:
        import nulling_mcmc as nm
    except ImportError:
        print("Could not import nulling package; skipping all nulling steps")


def _gauss(x, a, b, c):
    return(a*np.exp(-0.5*((x-b)/c)**2))


def _gauss_2(x, a1, b1, c1, a2, b2, c2):
    return(_gauss(x, a1, b1, c2) + _gauss(x, a2, b2, c2))


def _gauss_3(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return(_gauss(x, a1, b1, c2) + _gauss(x, a2, b2, c2) + _gauss(x, a3, b3, c3))


# Paul's alignment function
def aligndata(baselineremoved, brightest):
    nbins = baselineremoved.shape[0]
    nprofiles = baselineremoved.shape[1]
    template = baselineremoved[:,brightest]
    # rotate template to put peak at 1/4
    peakbin = np.argmax(template)
    fixedlag = int(nbins/4)-peakbin
    aligned = np.zeros((nbins, nprofiles))
    aligned_temp = np.zeros((nbins, nprofiles))
    aligned_temp2 = np.zeros((nbins, nprofiles))     
    newtemplate = np.roll(template, fixedlag)
    template = newtemplate
    for i in range(nprofiles):
        xcorr = np.correlate(template,baselineremoved[:,i], "full")
        lag = np.argmax(xcorr)
        aligned[:,i] = np.roll(baselineremoved[:,i], lag)
        
    template = np.median(aligned,1)
    # repeat with better template now and shift peak to 1/4 of the profile
    peakbin = np.argmax(template)
    fixedlag = int(nbins/4)-peakbin
    double = np.zeros(2*nbins)
    for i in range(nprofiles):
        double[0:nbins] = baselineremoved[:,i]
        double[nbins:2*nbins] = baselineremoved[:,i]
        xcorr = np.correlate(template, double, 'full')
        lag = np.argmax(xcorr) + fixedlag
        lag_temp = np.argmax(xcorr) + fixedlag - 1
        aligned[:,i] = np.roll(baselineremoved[:,i], lag)
        aligned_temp[:,i] = np.roll(baselineremoved[:,i], lag_temp)
        newtemplate = np.median(aligned, 1)
        
    return(np.array(aligned), np.array(newtemplate))


# Also Paul's alignment function
def smart_align(template, observations):
    aligned_obs = np.zeros((observations.shape[1], observations.shape[0]))
    aligned_obs_norm = np.zeros((observations.shape[1], observations.shape[0]))
    
    keep_lag = np.zeros(observations.shape[1])

    for n in range(observations.shape[1]):
        #print('Aligning observation',n+1,'of',observations.shape[1])
        max_val = max(1e-6, np.max(observations[:,n]))
        if np.isnan(max_val):
            max_val = 1e-6
            
        obs = observations[:,n]/max_val
        bins = template.shape[0]
        first_try = 0
        no_scale_incr = 100
        #template_noise_list = []
        #obs_noise_list = []
        list_of_means = []
        list_of_stds = []
        list_list_means = []
        list_list_stds = []
        list_list_no_points = []
        min_arg_list = []
        min_val_list = []
        std_min_val_list = []
        mean_times_points = []

        # Correlate to find rough alignment and then start with a fractional offset before rolling the observations past each other
        # make sure observations don't span the edge
        # rotate template to put peak at 1/4 
        peak_bin = np.argmax(template)
        initial_shift = int(bins/4)-peak_bin
        template = np.roll(template, initial_shift)

        xcorr = np.correlate(template, obs, "full")
        lag = np.argmax(xcorr) - len(template) - 1
        keep_lag[n] = lag

        obs = np.roll(obs, lag)
        aligned_obs[n,:] = obs
        max_val = max(1e-6, np.max(obs))
        if np.isnan(max_val):
            max_val = 1e-6
            
        aligned_obs_norm[n,:] = obs/max_val
        continue
        print("Error, didn't skip this part of the loop")
        break
        # obs = np.roll(obs, -int(bins/7.0))

        # Break the template into 8 parts and find the rms of each. Then find the smallest. Do with the observation too
        #for z in range(8):
        #    template_noise_list.append(np.std(template[z*int(bins/8.0):(z+1)*int(bins/8.0)]))
        #    obs_noise_list.append(np.std(obs[z*int(bins/8.0):(z+1)*int(bins/8.0)]))

        # Find the approximate peaks of template and observation so give an idea of the range over which to scale the observations
        temp_peak = np.mean(np.sort(template)[-10:])
        obs_peak = np.mean(np.sort(obs)[-10:])
        rough_scale = temp_peak / obs_peak
        rough_scale_upp = rough_scale * 1.1
        rough_scale_low = rough_scale * 0.9
        scale_incr = (rough_scale_upp - rough_scale_low)/no_scale_incr

        # Keep a copy of the observation in its original state
        obs_original = obs[:]

        # Phase shift over ~30% of the bins
        for roll in range(int(bins/3.5)):
            #if (roll+1)%100 == 0:
            #    print( 'Bin',roll+1,'out of',int(bins/3.5)
            closest_to_one = 1e10
            bins_with_signal = []
            bins_with_signal_test = []
            list_mean_each_scale_shift = []
            list_std_each_scale_shift = []
            list_points_each_scale_shift = []
            
            # No shift needed for first try
            if roll != 0:
                obs = np.roll(obs, 1)
                
            # If the level is too low in either template or observation, don't include the bin in further analysis
            for r in range(bins):
                #print( r,obs[r],obs_peak,np.min(obs_noise_list),template[r],temp_peak,np.min(template_noise_list) 
                if obs[r] > obs_peak/3. and template[r] > temp_peak/3.:
                    bins_with_signal.append(r)
                    bins_with_signal_test.append(1)
                else:
                    bins_with_signal_test.append(0)

            # For each roll, only proceed if there are more than 10 bins that have signal in them
            if len(bins_with_signal) >= 10.0:
                # Loop over each of the 100 scale attempts to find which is the best fit
                for s in range(no_scale_incr):
                    diff = []
                    escape = 0
                    first_try = 0
                    scaled_obs = obs*(rough_scale_low+s*scale_incr)
                    # Loop over all the bins with signal and find the absolute difference between template and observation
                    for each in bins_with_signal:
                        diff.append(abs(scaled_obs[each] - template[each]))
                        # Save this difference list before outliers are removed
                        
                    orig_diff = diff[:]
                    # Remove outliers (over 2 sigma) and re-evaluate the mean. If mean doesn't change much, exit the loop. Record the last set of data that had outliers removed
                    while escape == 0:
                        #diff_mean = np.mean(diff)
                        #diff_std = np.std(diff)
                        #outlier_list = []
                        #for y in range(len(diff)):
                        #    if abs(diff[y]-diff_mean) > 2*diff_std:
                        #        outlier_list.append(y)

                        #latest_diff_list = diff[:]
                        #for index in sorted(outlier_list, reverse=True):
                        #    del diff[index]
                            
                        #latest_diff_list = diff[abs(diff - diff_mean) <= 2*diff_std]

                        if np.mean(diff) == 0 or (np.mean(diff)/diff_mean < 1.001 and np.mean(diff)/diff_mean > 0.999 and first_try == 1):
                            escape = 1
                        else:
                            diff_mean = np.mean(diff)
                            diff_std = np.std(diff)
                            diff = diff[abs(diff - diff_mean) <= 2*diff_std]

                        first_try = 1
                        
                    # In lists - For any phase, record the mean and std and number of data points after all outliers removed at each scale attempt
                    list_mean_each_scale_shift.append(abs(np.mean(diff)))
                    list_std_each_scale_shift.append(np.std(diff))
                    list_points_each_scale_shift.append(len(diff))

                # Make a list containing the above lists. 1 per phase shift
                list_list_means.append(list_mean_each_scale_shift)
                list_list_stds.append(list_std_each_scale_shift)
                list_list_no_points.append(list_points_each_scale_shift)

            else:
                # If the number of bins with signal is not high enough, just put 1s into the list of lists. We will find minimum later, and 1 is >>
                list_list_means.append([1]*no_scale_incr)
                list_list_stds.append([1]*no_scale_incr)
                list_list_no_points.append([1]*no_scale_incr)

        # Calculate the mean / number of points. This should be minimised to find the best fit
        for h in range(len(list_list_means)):
            for y in range(len(list_list_means[0])):
                mean_times_points.append(list_list_means[h][y]/list_list_no_points[h][y])

        min_arg_final = np.argmin(mean_times_points)
        the_scale = min_arg_final%no_scale_incr
        the_roll = min_arg_final/no_scale_incr
        min_val_final = np.min(mean_times_points)
        std_min_val_final = list_list_stds[int(min_arg_final/no_scale_incr)][int(min_arg_final%no_scale_incr)]

        # Return the aligned and scaled observations
        aligned_obs_norm[n,:] = np.roll(obs_original*(rough_scale_low+the_scale*scale_incr), int(the_roll))
        #obs = observations[:,n]/np.max(observations[:,n])

        aligned_obs[n,:] = np.roll(obs_original*np.max(observations[:,n]), int(the_roll))                         

    aligned_obs = np.transpose(aligned_obs)
    aligned_obs_norm = np.transpose(aligned_obs_norm)
    
    return(aligned_obs_norm, aligned_obs, keep_lag)


# Paul's baseline removal function
def removebaseline(data, outliers, logg=None):
    # chop profile into 8 parts and check the part with the lowest rms.
    # Remove the mean of that from everything. Remove outliers based on rms.
    outlierindex = []
    inlierindex = []
    nbins = data.shape[0]
    nprofiles = data.shape[1]
    # initialize output array
    baselineremoved = data
    smallestrms = np.zeros(nprofiles)
    smallestmean = np.zeros(nprofiles)
    peak = np.zeros(nprofiles)
    for i in range(nprofiles):
        mask = np.array([np.isfinite(A) for A in data[:,i]])
        if not np.any(mask):
            outlierindex.append(i)
            continue
            
        rms = np.zeros(8)
        mean = np.zeros(8)
        section = int(nbins/8)
        for j in range(8):
            rms[j] = np.std(data[j*section:(j+1)*section,i])
            mean[j] = np.mean(data[j*section:(j+1)*section,i])
            
        smallestrms[i] = np.min(rms)
        peak[i] = np.max(data[:,i]) # remove low snr not just the std of noise
        baseindex = np.argmin(rms)
        baseline = mean[baseindex]
        smallestmean[i] = baseline
        baselineremoved[:,i] = data[:,i] - baseline
        
    baselineremoved = np.nan_to_num(baselineremoved) # my modification
    medianrms = np.median(smallestrms)
    medianpeak = np.median(peak)
    for i in range(nprofiles):
        if smallestrms[i]/np.max(data[:,i]) > outliers * medianrms/medianpeak and i not in outlierindex:
            outlierindex.append(i)
        else:
            inlierindex.append(i)

    ou = np.array(sorted(outlierindex))
    inl = np.array(inlierindex)
    
    if len(ou) > 0:
        removedprofiles = np.delete(baselineremoved,inl,1)
        baselineoutlierremoved = np.delete(baselineremoved, ou, 1)
        rmsremoved = np.delete(smallestrms, ou)
        if logg:
            logg.info('Number of noisy profiles identified (old function): {}'.format(ou.shape[0]))
        else:
            print('Number of noisy profiles identified (old function): ', ou.shape[0])
    else:
        baselineoutlierremoved = np.copy(baselineremoved)
        removedprofiles = np.array([])
        rmsremoved = np.copy(smallestrms)
        
    return(baselineoutlierremoved, removedprofiles, rmsremoved, ou, inl)


# calculate the S/N of profiles in an array
# for a 1D array of values `val`, define `lim` to exclude the highest values in `val` (iteratively), so the width of the pulse and S/N are:
#width = float(max(len(val) - len(val[lim]), 1))
#snr = np.abs(np.array([(n - val[lim].mean()) for n in val]).sum())/(sig*np.sqrt(width))
def calc_snr_old(data, peak_bin=None):
    """
    If the input data array is 2D, this function assumes the profiles are aligned (not strictly required)
    
    Input:
        data - a 1- or 2-D array where `nbin` is the first dimension (if 2-D, shape is (nbin, nobs))
        peak_bin - integer representing the index of the average maximum value (bin-wise)
        
    Output:
        if the input array is 1D, a float will be returned
        if the input array is 2D, a 1D array of length `nobs` will be returned
    
    """
    
    import decimal
    decimal.getcontext().prec = 80
    d = decimal.Decimal
    
    if peak_bin is None:
        if len(data.shape) > 1:
            summed = np.sum(data, axis=1)
            peak_bin = np.where(summed == summed.max())[0][0]
            
        else:
            peak_bin = np.where(data == data.max())[0][0]
            
    if len(data.shape) > 1:
        result = np.zeros(data.shape[1])
        for iobs, profile in enumerate(data.T):
            off_p, width = _find_off_pulse(profile, profile != profile[peak_bin])
            #snr = np.abs(np.array([(n - profile[off_p].mean()) for n in profile]).sum())/(profile[off_p].std()*np.sqrt(width))
            width = max(1, width)
            #snr = np.abs(np.sum(profile - profile[off_p].mean()))/(profile[off_p].std()*np.sqrt(width))
            off_mean = profile[off_p].mean()
            term1 = d(np.abs(np.sum(profile - off_mean)))
            term2 = d(profile[off_p].std()*np.sqrt(width))
            if term1 == 0:
                result[iobs] = 0
            elif term2 == 0:
                result[iobs] = 1e6
            else:
                try:
                    snr = float(term1/term2)
                except decimal.InvalidOperation:
                    c = decimal.Context()
                    mag1 = c.logb(term1)
                    mag2 = c.logb(term2)
                    if mag2 >= mag1:
                        snr = 1
                    else:
                        mag_fin = mag1 - mag2
                        snr = float(((term1/10**mag1)/(term2/10**mag2))*10**mag_fin)

                result[iobs] = snr
            
    else:
        off_p, width = _find_off_pulse(profile, profile != profile[peak_bin])
        result = np.abs(np.sum(profile - profile[off_p].mean()))/(profile[off_p].std()*np.sqrt(width))
        
    return(result)


def _check_lim(lim, prof):
    if np.all(lim) or not np.any(lim):
        i = np.argmax(prof)
        lim = np.array([True for a in prof])
        lim[i-1:i+2] = False

    return(lim)


def _find_off_pulse(profile, lim=None):
    num = np.arange(len(profile))
    if lim is None:
        lim = profile < (profile.max() - profile.min())/2 + profile.min()
        
    lim = _check_lim(lim, profile)
    i = 0
    while i < 3:
        lim2 = profile < profile[lim].mean() + 2.5*profile[lim].std()
        lim2 = _check_lim(lim2, profile)
        lim = profile < profile[lim2].mean() + 2.5*profile[lim2].std()
        lim = _check_lim(lim, profile)
        if profile[lim].std() == profile[lim2].std():
            break
            
        i += 1
        
    width = float(max(len(profile) - len(profile[lim]), 1))
    if width > 1:
        high_val = num[profile == profile.max()][0]
        sig = profile[lim].std()
        moff = profile[lim].mean()
        for i, V in enumerate(profile):
            if i == high_val or (lim[i-1] == lim[i] and lim[i] == lim[(i+1)%len(lim)]):
                continue
                
            list_be = [profile[A] for A in [i-1, i]]
            list_af = [profile[A] for A in [i, (i+1)%len(lim)]]
            av_be = np.mean(list_be)
            av_af = np.mean(list_af)
            
            if av_be < V < av_af or av_be > V > av_af: # monotonic
                if av_be > moff + 2*sig or av_af > moff + 2*sig:
                    # if neighbouring point is signal, include point i
                    lim[i] = False
            elif lim[i] and (V < av_be and V < av_af):
                lim[i] = False # include point in pulse if in a local min
            elif not lim[i] and (V > av_be and V > av_af):
                lim[i] = True # exclude if a local max
            
            if profile[i] < moff + sig:
                # make sure no low points pass through
                lim[i] = True
                
    width = float(max(len(profile) - len(profile[lim]), 1))
    lim = _check_lim(lim, profile)

    return(lim, width)


def rem_extra_noisy(data, mjds, threshold=2):
    """
    Look at aligned profiles to find bins most frequently identified as 'off-pulse', compare rms values, and remove
    As compared with `get_rms_bline`, this function finds the most accurate off-pulse rms, not the minimum
    
    Input:
        data - a 2D array with shape (nbin, nobs) representing observed profiles
        mjds - a 1D array with length `nobs` representing the epochs of observations
        threshold - a float representing the number of standard deviations to determine outliers (rms and S/N)
        
    Output:
        a 2D array with shape (nbin, nobs_new) representing good profiles with baselines removed
        a 1D array with length `nobs_new` representing the epochs of good observations

    """
    
    mask = np.zeros(data.shape[0])
    for prof in data.T:
        lim, _ = _find_off_pulse(prof)
        for num, val in enumerate(lim):
            mask[num] += 1 if val else 0
            
    # mask now contains counts of which bins are included as 'off-pulse'
    new_lim = mask > data.shape[1]*0.6 # bins marked as off-pulse in >60% of observations
    # subtract baseline based on new 'off-pulse' region
    data -= np.mean(data[new_lim,:], axis=0)
    max_val = np.nan_to_num(np.max(data, axis=0))
    for i, val in enumerate(max_val):
        if val <= 0:
            max_val[i] = 1e-4
            
    data /= max_val
    
    new_rms = np.sqrt(np.mean(data[new_lim,:]**2, axis=0))
    rms_avg = new_rms.mean()
    lim1 = new_rms < rms_avg + new_rms.std()*threshold
    good_data = (data.T[lim1]).T
    good_mjds = mjds[lim1]
    
    return(good_data, good_mjds)


# new baseline/outlier removal function
def rem_base_outs(data, mjds, tobs=None, threshold=1.5, wide=False, logg=None, cut_snr=False, quiet=False):
    """
    The basic cleaning function
    
    Input:
        data - a 2D array with shape (nbin, nobs) representing observed profiles
        mjds - a 1D array with length `nobs` representing the epochs of observations
        threshold - a float representing the number of standard deviations to determine outliers (rms and S/N)
        wide - a boolean indicating if the distribution of widths is assumed to be broad
        logg - a `logging` object
        cut_snr - a boolean flag indicating whether to cut based on S/N with a strict threshold
        quiet - a boolean indicating whether to not print info (still print warnings and errors)
        
    Output:
        a 2D array with shape (nbin, nobs_new) representing good profiles with baselines removed
        a 2D array with shape (nbin, nobs_bad) representing *bad* profiles with baselines removed
        a 1D array with length `nobs_new` representing the epochs of good observations
        a 1D array with length `nobs_new` representing the off-pulse rms of good observations
        a 1D array with length `nobs_bad` representing the indices of *bad* observations
        a 1D array with length `nobs_new` representing the indices of good observations
    
    """
    
    data = np.nan_to_num(data, 1e-6)
    phase = np.linspace(0, 1, data.shape[0])
    
    # find the width of the peaks and the off-pulse rms and baseline
    all_wid = []
    all_rms = []
    all_bln = []
    for prof in data.T:
        lim, width = _find_off_pulse(prof)
        all_wid.append(width)
        all_rms.append(np.sqrt(np.mean(prof[lim]**2)))
        all_bln.append(np.mean(prof[lim]))
        
    all_wid = np.array(all_wid)
    all_rms = np.array(all_rms)
    all_bln = np.array(all_bln)
    
    data_subd = data - all_bln    
    rms_avg = all_rms.mean()
    rms_med = np.median(all_rms)
    lim1 = all_rms < rms_med + all_rms.std()*threshold
    if len(all_rms[lim1]) < data.shape[1]*0.5:
        if logg:
            logg.warning("Cutting by rms will remove more than 50% of the observations")
        else:
            print("Warning: Cutting by rms will remove more than 50% of the observations")
        
    if cut_snr:
        all_snr = calc_snr(data_subd)
        all_snr = np.nan_to_num(all_snr, 1e-6)
        #plt.hist(all_snr, bins=30)
        good_snr = all_snr > 1e-6
        snr_med = np.median(all_snr[good_snr])
        lim2 = all_snr > max(snr_med - all_snr[good_snr].std()*threshold, 5)
        if len(all_snr[lim2]) < data.shape[1]*0.5:
            if logg:
                logg.warning("Cutting by S/N will remove more than 50% of the observations")
            else:
                print("Warning: Cutting by S/N will remove more than 50% of the observations")
    else:
        lim2 = np.array([True for a in all_rms])
     
    # also check the pulse widths
    wid_men = all_wid.mean()
    # beware of bimodal/wide distributions
    if wide:
        wid_thrsh = 3.5*threshold
    else:
        wid_thrsh = 2*threshold
        
    wid_std = all_wid.std()
    lim3 = np.logical_and(all_wid > wid_men - wid_thrsh*wid_std, all_wid < wid_men + wid_thrsh*wid_std)
    if len(all_wid[lim3]) < data.shape[1]*0.5:
        if logg:
            logg.warning("Cutting by width will remove more than 50% of the observations")
        else:
            print("Warning: Cutting by width will remove more than 50% of the observations")
     
    lim_good = np.logical_and(np.logical_and(lim1, lim2), lim3)
    lim_bad = np.logical_not(lim_good)
    good_data = (data_subd.T[lim_good]).T
    bad_data = (data_subd.T[lim_bad]).T
    good_mjds = mjds[lim_good]
    rms_kept = all_rms[lim_good]
    inds = np.arange(data.shape[1])
    out_inds = inds[lim_bad]
    in_inds = inds[lim_good]
    
    if logg:
        logg.info('Number of noisy profiles removed: {}'.format(len(out_inds)))
    elif not quiet:
        print('Number of noisy profiles removed: {}'.format(len(out_inds)))
    
    return(good_data, bad_data, good_mjds, rms_kept, out_inds, in_inds)


def get_rms_bline(profs, centre=0.7, width=0.2, test=False):
    """
    Input:
        profs - a 2D array with shape (nbin, nobs) representing observed profiles (assume aligned)
        centre - the phase centre for the 'off-pulse' region
        width - the total width (fraction) of the 'off-pulse' region
        test - boolean indicating whether to test different centre and width values to minimise the rms
               if True, the test is run over `centre` values in [centre - 0.4, centre + 0.4] and `width` in [0.2, 0.7]
        
    Output:
        a 1D array of length `nobs` of measured off-pulse rms
        a 1D array of length `nobs` of measured off-pulse baseline
    
    """
    
    if test:
        cen_arr = np.linspace(centre - 0.4, centre + 0.4, 16)
        wid_arr = np.linspace(0.2, 0.7, 10)
        res_arr = np.empty((profs.shape[1], 16, 10))
        bln_arr = np.empty((profs.shape[1], 16, 10))
        for cen_ind, centre in enumerate(cen_arr):
            for wid_ind, width in enumerate(wid_arr):
                phase = np.linspace(0, 1, profs.shape[0])
                lim = np.logical_and(phase > centre - width/2, phase < centre + width/2)
                res_arr[:, cen_ind, wid_ind] = np.sqrt(np.mean(profs[lim,:]**2, axis=0))
                bln_arr[:, cen_ind, wid_ind] = np.mean(profs[lim,:], axis=0)
                
        print(res_arr.shape)
        res_arr = np.reshape(res_arr, (profs.shape[1], 160))
        bln_arr = np.reshape(bln_arr, (profs.shape[1], 160))
        lim2 = (res_arr.T == np.min(res_arr, axis=1)).T
        result = res_arr[lim2]
        res_bln = bln_arr[lim2]

    else:
        phase = np.linspace(0, 1, profs.shape[0])
        lim = np.logical_and(phase > centre - width/2, phase < centre + width/2)
        if len(phase[lim]) == 0:
            raise(RuntimeError("Slice based on centre and width has length of 0"))
            
        result = np.sqrt(np.mean(profs[lim,:]**2, axis=0))
        res_bln = np.mean(profs[lim,:], axis=0)
    
    if len(res_bln) == 0 or result.shape != res_bln.shape:
        raise(RuntimeError("Error making array of baseline values"))
        
    return(result, res_bln)


# New function to replace `findbrightestprofile`
def find_bright(data):
    snr = calc_snr(data)
    return(np.argmax(snr))


# Paul's function to determine the brightest profile
def findbrightestprofile(data, rmsdata):
    snr = np.zeros(rmsdata.shape[0])
    for i in range(data.shape[1]):
        snr[i] = np.max(data[:,i])/rmsdata[i]
        
    brightestindex = np.argmax(snr)
    return(brightestindex)


def read_pdv(filename, nbin=1024, logg=None):
    output = None
    mjd_out = None
    tobs_out = None
    with open(filename, 'r') as f:
        i_bin = 0
        j_file = 0
        f_lines = f.readlines()
        n_files = len(f_lines)/(nbin+2.) # 2 header lines per file
        if n_files % 1 != 0:
            if logg:
                logg.info("Incorrect number of bins given, or input data have inconsistent "\
                          "numbers of bins")
            else:
                print("Incorrect number of bins given, or input data have inconsistent "\
                      "numbers of bins")
        else:
            output = np.empty((nbin, int(n_files)))
            mjd_out = np.empty(int(n_files))
            tobs_out = np.empty(int(n_files))
            
        for line in f_lines:
            line_vals = line.split()
            if i_bin == 0 and j_file == 0 and line_vals[0] == 'File:':
                # get nbin from header line
                for num, val in enumerate(line_vals):
                    if val == 'Nbin:' and len(line_vals) > num+1 and nbin != int(line_vals[num+1]):
                        nbin = int(line_vals[num+1])
                        n_files = len(f_lines)/(nbin+2.)
                        if n_files % 1 != 0:
                            if logg:
                                logg.info("Incorrect number of bins given, or input data have "\
                                          "inconsistent numbers of bins")
                            else:
                                print("Incorrect number of bins given, or input data have "\
                                      "inconsistent numbers of bins")
                        else:
                            output = np.empty((nbin, int(n_files)))
                            mjd_out = np.empty(int(n_files))
                            tobs_out = np.empty(int(n_files))
                                
                        break
                        
                if output is None:
                    if logg:
                        logg.error("Number of bins not identified for {} so no output array "\
                                   "was made".format(filename))
                    else:
                        print("ERROR: Number of bins not identified for {} so no output array "\
                              "was made".format(filename))
                        
                    break
            else:
                if len(line_vals) < 5:
                    output[i_bin, j_file] = float(line_vals[3])
                    i_bin += 1
                elif line_vals[0] == 'File:':
                    i_bin = 0
                    j_file += 1
                elif line_vals[0] == 'MJD(mid):':
                    mjd_out[j_file] = float(line_vals[1][:13])
                    tobs_out[j_file] = round(float(line_vals[3]), 0)

    return(output, mjd_out, tobs_out)


def read_bad_mjd_file(filename):
    """
    Read the provided ascii file and return a dictionary containing MJDs for each
    <pulsar_BE_freq> keyword

    MJDs contained in the file are not required to be ordered or unique (including
    for any given <pulsar_BE_freq> keyword)

    """

    n = 0
    out_dict = {}
    mjd_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            sline = line.strip()
            if n == 0:
                if sline[0] == '#':
                    continue

                if len(sline.split()) == 1:
                    kwd = sline
                else:
                    raise(ValueError("Cannot interpret multi-word header as keyword"))
                
                n = 1
            elif sline[0] == '#':
                if len(np.unique(list(sline))) > 1:
                    # skip lines starting with # but containing other characters
                    continue
                
                # reached the end of the MJDs for that pulsar_BE_freq
                out_dict[kwd] = mjd_list
                mjd_list = []
                n = 0
            else:
                try:
                    mjd_list.append(round(float(sline), 7))
                except TypeError:
                    raise(TypeError("Found a non-float: {}".format(sline)))

    # ensure the last set is properly recorded
    if len(mjd_list) > 0:
        out_dict[kwd] = mjd_list
        
    return(out_dict)


# find outliers by first modeling the distribution of eigenvalues (each component separately) as 3 gaussians (bimodal distribution plus)
def find_dists_outliers(eigvals, mjds, psr, be, ncomp=5, savename=None, show=True, bk_bgd=False,
                        first_out=False, nbins=80, return_lim=False, sigma=3, logg=None, verb=False):
    """
    Find outliers based on the eigenvalues
    Number of points per eigenvector must be greater than 100 (150 for 3-gaussian fit)
    
    Input:
        eigvals : 2D array with shape (nobs, components) representing the eigenvalues for all eigenfunctions
        mjds : 1D array of floats of length `nobs`
        psr : str indicating the pulsar, used only for printing
        be : str indicating the backend, used only for printing on the plot
        ncomp : int representing the maximum number of components to check (defaults to the total number of components)
        savename : str or NoneType, the filename for saving the plot to disk (if `None`, do not save the plot)
        show : boolean, whether to show the plot
        bk_bgd : boolean, whether to plot on a black background
        first_out : boolean, whether to check the components but only return the outliers from the first (most significant) eigenvalues
        nbins : int, the number of bins to use in the histograms
        return_lim : boolean, whether to return just the boolean mask to be applied to `mjds` such that `mjds[mask] == bad_mjds`
        sigma : float or int, the number of standard deviations to use as a cut for all outliers
        logg : logging object
        verb : boolean
        
    Output:
        a 1D array of:
            (Default) MJDs for the outliers (with length < `nobs`)
            OR booleans to be used as a mask, of length `nobs`
    
    """
    
    if len(mjds) < 100:
        if logg is not None:
            logg.warning("Dataset too small")
        else:
            print("Dataset too small")
            
        if return_lim:
            return(np.array([False for A in mjds]))
        else:
            return(np.array([]))
    
    if bk_bgd:
        #plt.style.use('dark_background')
        style = 'dark_background'
        cmap = cmr.chroma_r
        lc = 'w'
        k_alpha = 0.8
    else:
        #plt.style.use('default')
        style = 'default'
        cmap = cmr.chroma
        lc = 'k'
        k_alpha = 0.2
    
    if ncomp is None or ncomp > eigvals.shape[1]:
        ncomp = eigvals.shape[1]        
        
    #c1 = cmap(0.0)
    c2 = cmap(0.3)
    #c3 = cmap(0.53)
    #c4 = cmap(0.65)
    #c5 = cmap(0.9)
    
    with plt.style.context(style):
        plt.clf()
        if savename or show:
            fig = plt.figure(num=1)
            fig.set_size_inches(8, 4)
            fig.suptitle('{}, {}, eigenvalue distributions'.format(psr, be), fontsize=13)
            fig.set_size_inches(6, 8)
            
        # define axes parameters
        w = 0.92 # width
        l = 0.05 # left
        b = 0.05 # bottom
        sep = 0.05 # separation
        # h*n+sep*(n-1) = 0.87
        h = (0.87 - (sep*(ncomp-1)))/ncomp # height
    
        lim_out = np.array([False for a in mjds])

        for icomp, vals in zip(range(ncomp), np.flip(eigvals[:,:ncomp], axis=1).T):
            if savename or show:
                if icomp == 0:
                    ax = fig.add_axes((l, b, w, h))
                    ax.set_xlabel('Eigenvalue', fontsize=12)
                else:
                    ax = fig.add_axes((l, b + icomp*h + icomp*sep, w, h))

            #print(vals[:10])
            if savename or show:
                dist, bins, _ = ax.hist(vals, bins=nbins)
            else:
                dist, bins = np.histogram(vals, bins=nbins)
    
            bin_size = bins[1] - bins[0]
            mids = np.array((bins[1:] + bins[:-1])/2)
            dist = np.array(dist)
    
            guess_1 = np.argmax(dist)
            lim1 = np.array([True for a in dist])
            lim1[guess_1-1:guess_1+2] = False
            guess_2 = np.argmax(dist[lim1])
            lim2 = lim1
            lim2[guess_2-1:guess_2+2] = False
            guess_3 = np.argmax(dist[lim2])
    
            p0 = (dist[guess_1], mids[guess_1], bin_size*2, dist[guess_2], mids[guess_2], bin_size*2, dist[guess_3], mids[guess_3], bin_size*2)
            bounds = ([0, bins[0], 0, 0, bins[0], 0, 0, bins[0], 0],
                      [np.max(dist)*2, bins[-1], bins[-1]-bins[0], np.max(dist)*2, bins[-1], bins[-1]-bins[0], np.max(dist)*2, bins[-1], bins[-1]-bins[0]])
    
            extra_bins = np.linspace(bins[0]-bin_size, bins[-1]+bin_size, 200)
            proceed3 = False
            proceed2 = False
            try:
                if len(mjds) < 150:
                    raise(RuntimeError("Dataset too small for 3 gaussians"))
                    
                popt, pcov = op.curve_fit(_gauss_3, mids, dist, p0, bounds=bounds)
                proceed3 = True
            except RuntimeError:
                if logg is not None:
                    logg.info("Could not fit 3 gaussians to the data, trying 2 gaussians")
                elif verb:
                    print("Could not fit 3 gaussians to the data, trying 2 gaussians")
                p0 = (dist[guess_1], mids[guess_1], bin_size*2, dist[guess_2], mids[guess_2], bin_size*2)
                bounds = ([0, bins[0], 0, 0, bins[0], 0],
                          [np.max(dist)*2, bins[-1], bins[-1]-bins[0], np.max(dist)*2, bins[-1], bins[-1]-bins[0]])
                try:
                    popt, pcov = op.curve_fit(_gauss_2, mids, dist, p0, bounds=bounds)
                    proceed2 = True
                except RuntimeError:
                    if logg is not None:
                        logg.info("Failed to fit the data with 2 gaussians; proceeding with only rough outlier excision")
                    else:
                        print("Failed to fit the data with 2 gaussians; proceeding with only rough outlier excision")
            
            if proceed3 or proceed2:
                if proceed3:
                    fit_fun = _gauss_3(extra_bins, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
                else:
                    fit_fun = _gauss_2(extra_bins, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        
                fill_fun = []
                while len(fill_fun) < 2000:
                    x = np.random.random()*(bins[-1]-bins[0]) + bins[0]
                    if proceed3:
                        y = _gauss_3(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
                    else:
                        y = _gauss_2(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
                    
                    y_samp = np.random.random()*np.max(dist)*1.2
                    if y_samp <= y:
                        fill_fun.append(x)
                        
                fill_fun = np.array(fill_fun)
                if int(round(2*sigma, 0)) == 2:
                    perc_val = 16
                elif int(round(2*sigma, 0)) == 3:
                    perc_val = 6.6
                elif int(round(2*sigma, 0)) == 4:
                    perc_val = 2.3
                elif int(round(2*sigma, 0)) == 5:
                    perc_val = 0.65
                elif int(round(2*sigma, 0)) == 6:
                    perc_val = 0.14
                elif int(round(2*sigma, 0)) == 7:
                    perc_val = 0.023
                elif int(round(2*sigma, 0)) == 8:
                    perc_val = 0.003
                else:
                    perc_val = 0.001
                
                low_lim = np.percentile(fill_fun, perc_val)
                up_lim = np.percentile(fill_fun, 100-perc_val)
                        
                if savename or show:
                    y_uplim = max(np.max(fit_fun), np.max(dist))*1.1
                    ax.plot(extra_bins, fit_fun, '-', color=c2)
                
            else:
                y_uplim = np.max(dist)*1.1
                low_lim = dist.mean() - sigma*dist.std()
                up_lim = dist.mean() + sigma*dist.std()
        
            #low_lim = min(popt[1]-2.5*popt[2], popt[4]-2.5*popt[5], popt[7]-2.5*popt[8])
            #up_lim = max(popt[1]+2.5*popt[2], popt[4]+2.5*popt[5], popt[7]+2.5*popt[8])
    
            if savename or show:
                ax.vlines([low_lim, up_lim], 0, y_uplim, ls='--')
                ax.set_xlim(bins[0]-2*bin_size, bins[-1]+2*bin_size)
                ax.set_ylim(0, y_uplim)
                ax.text(0.72, 0.95, "{0:.1f}-$\sigma$ lo-lim: {1:.2f}; N={3:d}\n{0:.1f}-$\sigma$ up-limit: {2:.2f}; N={4:d}"
                        .format(sigma, low_lim, up_lim, len(vals[vals < low_lim]), len(vals[vals > up_lim])),
                       fontsize=10, transform=ax.transAxes, verticalalignment='top')
                ax.text(0.025, 0.95, "$i_{{comp}}$ = {}".format(ncomp-icomp-1), fontsize=10, transform=ax.transAxes, verticalalignment='top')
        
            lim_out = np.logical_or(lim_out, np.logical_or(vals < low_lim, vals > up_lim))
            if first_out: # last array analysed is first component, will be the only outliers returned
                lim_out = np.logical_or(vals < low_lim, vals > up_lim)
    
        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
        
        if show:
            plt.show()
        
    if return_lim:
        return(lim_out)
    else:
        return(mjds[lim_out])


def rolling_out_rej(eigvals, mjds, psr, be, ncomp=5, savename=None, show=True, bk_bgd=False, first_out=False, nbins=40, logg=None, verb=False):
    """
    A wrapper for `find_dists_outliers` which checks if the dataset is large enough to be split
    into 8 sections to be checked for outliers (with overlap)
    
    """
    
    nobs = len(mjds)
    if nobs >= nbins*10:
        mjd_bins = np.linspace(mjds.min(), mjds.max(), 9)
        mjd_mins = mjd_bins[:-2]
        mjd_maxs = mjd_bins[2:]
        short_list = np.array([])
        count_fails = 0
        for mi, ma in zip(mjd_mins, mjd_maxs):
            lim = np.logical_and(mjds >= mi, mjds <= ma)
            evs = eigvals[lim,:]
            mds = mjds[lim]
            if len(mds) >= nbins*2 and len(mjds) >= 100:
                mjds_out = find_dists_outliers(evs, mds, psr, be, ncomp, None, False, bk_bgd, first_out, nbins, logg=logg)
                plt.close('all')
                short_list = np.unique(np.append(short_list, mjds_out))
            else:
                count_fails += 1
            
        if count_fails > 3:
            if logg is not None:
                logg.warning("At least half of the sections had too few observations for the calculation")
            else:
                print("Warning: at least half of the sections had too few observations for the calculation")
            
    elif nobs >= nbins*3:
        if logg is not None:
            logg.info("Cannot split dataset into sections; analysing all together")
        elif verb:
            print("Cannot split dataset into sections; analysing all together")
            
        short_list = find_dists_outliers(eigvals, mjds, psr, be, ncomp, savename, show, bk_bgd, first_out, nbins, logg=logg)
    else:
        if logg is not None:
            logg.info("Cannot reject outliers using this method; too few observations")
        elif verb:
            print("Cannot reject outliers using this method; too few observations")
            
        short_list = mjds
        
    return(short_list)


def bad_mjds_eigs(data, mjds, peak_min, peak_max, ip=False, ip_min=None, ip_max=None):
    nbins = data.shape[0]
    bins = np.linspace(0, 1, num=nbins, endpoint=False)
    mask = np.logical_or(bins < peak_min, bins > peak_max)
    if ip:
        mask = np.logical_and(mask, np.logical_or(bins < ip_min, bins > ip_max))

    pca = PCA(n_components=30)
    compsall = pca.fit_transform(data[mask,:].T)
    
    # get the uncertainties
    eig_errs = err_eigval_off(data[mask,:], pca.components_)
    
    bad_mjds = np.array([False for a in mjds])
    for icomp in np.arange(compsall.shape[1]):
        lim = np.abs(compsall[:,icomp]) > 3*eig_errs[:,icomp]
        if len(mjds[lim]) > 0.8*len(mjds):
            print("Too many bad MJDs in eigenvector number {}".format(icomp))
        else:
            bad_mjds = np.logical_or(bad_mjds, lim)
            
        if len(mjds[lim]) < 0.01*len(mjds):
            break
            
    return(mjds[bad_mjds])


def plot_joydivision(data, psrname, bk_bgd=True, color=True, show=True, short=False, savename='quick.png', roll=0):
    """
    Do some nice plotting of the observations
    
    """
    
    max_val = np.max(data)
    
    plt.clf()
    if bk_bgd:
        #plt.style.use('dark_background')
        style = 'dark_background'
        col_map = cmr.chroma
        bw_line = 'w-'
    else:
        #plt.style.use('default')
        style = 'default'
        col_map = cmr.chroma_r
        bw_line = 'k-'
        
    with plt.style.context(style):
        fig = plt.figure()
        fig.set_size_inches(5, 15)
        if data.shape[1] > 300:
            fig.set_size_inches(5, 25)
        
        if short:
            fig.set_size_inches(7, 10)
        
        plt.title("'Joy Division' stack of {} pulses for {}".format(data.shape[1], psrname), fontsize=12)
        ax = fig.gca()

        if color:
            # do manual clipping
            data = np.clip(data, max_val*1e-5, max_val)
            data = np.nan_to_num(data, nan=max_val*1e-5)
            map_norm = col.LogNorm(max_val*3e-4, vmax=max_val, clip=True)
        
        for n_file, prof in enumerate(np.flip(data.transpose(), 0)):
            m_file = data.shape[1] - n_file
            if color:
                prof_cols = np.array([max(max_val*3e-4, A) for A in np.roll(prof, roll)])
                p = ax.scatter(np.arange(len(prof)), (np.roll(2*prof, roll)/max(prof))+m_file, s=2, c=prof_cols, cmap=col_map, norm=map_norm)
            else:
                ax.plot(np.arange(len(prof)), (np.roll(prof, roll)/max(prof))+m_file, bw_line)
    
        if color:
            fig.colorbar(p, extend='min', fraction=0.05)

        plt.xlim(0, len(prof))
        plt.ylim(0, n_file + 4)
        plt.xlabel('Phase Bin', fontsize=12)
        plt.ylabel('Observation Number', fontsize=12)
    
        #ax.text(0.8, 0.99, 'N={}'.format(data.shape[1]), fontsize=14, transform=ax.transAxes)
    
        if savename is not None and type(savename) == str:
            plt.savefig(savename, bbox_inches='tight')

        if show:
            fig.set_size_inches(15, 45)
            if data.shape[1] > 300:
                fig.set_size_inches(15, 75)
            
            plt.show()
        
        
def setup_log(filename):
    logger = logging.getLogger(filename.split('.')[0])
    logger.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                  '- %(message)s')

    # Create file logging only if logging file path is specified
    if filename:
        # create file handler which logs everything
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Check if file already exists, differentiate between runs
        if os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write(20*"#"+"\n")

    # Create console handler with log level based on verbosity
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    return(logger)


def do_rem_aln(data_arr, mjds_arr, tobs_arr, thrsh=1.5, bad_mjds=None, wide=False, logg=None, cut_snr=False, quiet=False):
    """
    A pipeline-ish function to remove the baseline, remove bad profiles (including rms outliers and bad MJDs), align the peaks, and make a template.
    
    Input: 
      data_arr : a 2D numpy array of floats with shape (nbin, nprof) representing the input data
      mjds_arr : a 1D numpy array of floats with shape (nprof,) representing MJDs
      tobs_arr : a 1D numpy array of floats with shape (nprof,) representing observation lengths in seconds
      thrsh : float, representing the threshold for trimming outliers based on off-pulse rms
      bad_mjds : list (or numpy array) of floats, representing MJDs to be removed
      wide : bool, whether to assume the distribution of pulse widths is broad (e.g., from shape variation)
      logg : a `logging` object (e.g., from `setup_log()`)
      cut_snr : bool, whether to cut on S/N or leave possible nulls in the dataset
      quiet : bool, whether to not print anything beyond warnings and errors
      
    Output:
      2D numpy array of data, properly aligned with baseline removed
           shape is (nbin, good_profs) where good_profs <= nprof - len(bad_mjds)
      1D numpy array with shape (nbin,) representing the profile template
      1D numpy array with shape (good_profs,) representing the remaining MJDs
      1D numpy array with shape (good_profs,) representing the remaining observation lengths in seconds
    
    """
    
    # get counts of how many observations go in and go out
    nobs_in = data_arr.shape[1]
    
    # use new function to remove baseline and bad profiles
    a_wo_bl, a_rp, a_mjds_new, a_rms, a_out, a_in = rem_base_outs(
        data_arr, mjds_arr, thrsh, wide=wide, logg=logg, cut_snr=cut_snr, quiet=quiet)
    
    if bad_mjds is not None:
        if type(bad_mjds) is float:
            bad_mjds = np.array([bad_mjds])

        lim = np.array([M not in bad_mjds for M in a_mjds_new])
        a_wo_bl_new = a_wo_bl[:,lim]
        a_mjds_new_new = a_mjds_new[lim]
        
        if np.all(a_wo_bl_new == a_wo_bl) or np.all(a_mjds_new_new == a_mjds_new):
            raise(RuntimeError("Problem lies with applying the mask"))
        else:
            logg.info("{} MJDs were removed from the dataset, leaving {} epochs".format(len(a_mjds_new) - len(a_mjds_new_new), len(a_mjds_new_new)))
            a_wo_bl = a_wo_bl_new
            a_mjds_new = a_mjds_new_new
            
    if bad_mjds is not None and np.any(np.array([M in bad_mjds for M in a_mjds_new])):
        raise(RuntimeError("Bad MJDs remain in the array"))
                
    if a_wo_bl.shape[1] < 20:
        if logg is not None:
            logg.warning("Cuts have removed too many observations for the remaining to be useful")
        else:
            print("WARNING: Cuts have removed too many observations for the remaining to be useful")
            
        raise(RuntimeError("Not enough data"))
            
    # find the brightest observation to use as a template for alignment
    a_brightest = find_bright(a_wo_bl)
    a_aln, a_temp = aligndata(a_wo_bl, a_brightest)
    
    # one final step of removing outliers
    a_aln_new, a_mjds_new = rem_extra_noisy(a_aln, a_mjds_new, threshold=thrsh)
    a_tobs_new = tobs_arr[np.array([M in a_mjds_new for M in mjds_arr])]
    
    # re-align using a new template
    a_temp = np.mean(a_aln_new, axis=1)
    _, a_aln, lags = smart_align(a_temp, a_aln_new)
            
    #a_norm = a_aln_new/np.max(a_aln_new, axis=0)
    a_original = np.copy(a_temp)

    # normalise by the sum of on-pulse bins
    lim_off, _ = _find_off_pulse(a_original)
    lim_on = np.logical_not(lim_off)
    a_temp_norm = a_original/np.sum(a_original[lim_on])
    
    a_aln_norm = np.zeros(a_aln.shape)
        
    sum_val = np.sum(a_aln[lim_on,:], axis=0)
    sum_val = np.nan_to_num(sum_val)
    if len(sum_val.shape) > 1:
        raise(TypeError("Array of sums has more than 1 dimension??"))
        
    min_pos = np.min(sum_val[sum_val >= 0])
    for i, val in enumerate(sum_val):
        if val <= 0:
            sum_val[i] = min_pos/2
            
    a_aln_norm = a_aln/sum_val
    try:
        a_aln_norm = psrcelery.data.align_and_scale(a_aln_norm.T, a_temp_norm, nharm='auto').T
    except ValueError:
        logg.warning("FFT alignment failed for this dataset")
    
    if logg:
        logg.info("In total, removed {} observations, aligned remaining obs., and normalised according to on-pulse sum".format(nobs_in - a_aln_norm.shape[1]))
    elif not quiet:
        print("In total, removed {} observations, aligned remaining obs., and normalised according to on-pulse sum".format(nobs_in - a_aln_norm.shape[1]))
    
    return(a_aln_norm, a_temp_norm, a_mjds_new, a_tobs_new)


def calc_snr(data, verb=False):
    """
    A simple function to loop over a 2D dataset using `find_eq_width_snr` to get the S/N
    for each profile in the dataset
    
    Input:
        data - 2D array of floats, shape of (nbin, nobs)
        verb - bool, not currently implemented
        
    Output:
        1D array of floats, length of `nobs`
    
    """
    
    snr_ar = np.zeros(data.shape[1])
    for iobs, prof in enumerate(data.T):
        _, snr_ar[iobs] = find_eq_width_snr(prof)
        
    return(snr_ar)


def find_eq_width_snr(prof, verb=False, plot_style='dark_background'):
    """
    Find the width of a given pulse profile using a tophat convolution
    
    Input:
        prof - 1D array of floats, length of `nbin`, representing a single
            profile
        verb - bool, whether to verbosely print/plot diagnostic info
        plot_style - str, a valid matplotlib 'style', e.g., 'default'
        
    Output:
        float, the best width
        float, the S/N derived from the convolution method
    
    """
    
    try:
        plt.style.use(plot_style)
    except OSError:
        print("Couldn't load selected style; using the default instead")
        plot_style = 'default'
        
    if not (verb is True or verb is False):
        raise(TypeError("The type of `verb` is {}".format(type(verb))))
    
    if not np.any(prof - prof.mean()):
        if verb:
            print("Cannot find width of flat profile")
            
        return(1, 0)
    
    nbin = len(prof)
    tophat = np.zeros(nbin)
    lim_init, wid_guess = _find_off_pulse(prof)        
    if verb:
        print("My initial guess (overestimate) is", int(wid_guess))
        
    bin_cen = nbin/4
    width = int(0.5*nbin)-2
    max_val = []
    wid_val = []
    while width > 1 and width > wid_guess/10:
        for i in range(nbin):
            if i >= bin_cen - width/2 and i < bin_cen + width/2:
                tophat[i] = 1
            else:
                tophat[i] = 0
        
        res = np.convolve(prof, tophat)
        max_val.append(res.max())
        wid_val.append(width)
        
        if width > wid_guess*0.6 and width < wid_guess*2:
            step = 1
        else:
            step = max(int(width/5), 1)
        width -= step
        
    max_val = np.array(max_val)
    wid_val = np.array(wid_val)
    
    imax = np.argmax(max_val/np.sqrt(wid_val))
    best_wid = wid_val[imax]
        
    if verb:
        if 'dark' in plot_style:
            best_colour = 'w'
        else:
            best_colour = 'k'
            
        with plt.style.context(plot_style):
            plt.clf()
            fig = plt.figure(num=1)
            fig.set_size_inches(7, 5.5)
            plt.plot(wid_val, max_val/np.sqrt(wid_val))
            print("The number of trials was", len(wid_val))
            xlims = plt.xlim()
            plt.plot(best_wid, (max_val/np.sqrt(wid_val))[imax], '*', color=best_colour)
            plt.xlim(xlims)
            plt.xlabel('Trial width (bins)')
            plt.ylabel('Max. convolution value / sqrt(width)')
            plt.show()
            
    rms_off = np.sqrt(np.mean(prof[lim_init]**2))
    best_snr = (max_val/np.sqrt(wid_val))[imax]/rms_off
            
    return(best_wid, best_snr)


def sim_mjds(nobs=1000, min_lag=0.9, max_lag_frac=0.02):
    """
    Make fake MJDs for an 'observing programme' with a minimum cadence
    Ensure no separations between consecutive dates exceed a fraction of the number of observations
    
    Input:
        nobs - int, the length of the output array, the number of 'observations'
        min_lag - float, the minimum acceptable separation between MJDs, in days
        max_lag_frac - float, the maximum acceptable separation as a fraction of `nobs`
        
    Output:
        a 1D numpy array of floats representing the dates of the 'observations'
    
    """
    
    # make fake MJD values
    mjd_min = 49350 + np.random.randint(100)/np.random.randint(1, 100)
    fake_mjds = np.array(sorted(mjd_min + np.random.randint(nobs*np.random.lognormal(sigma=0.6), size=nobs)\
                                + np.random.normal(size=nobs)))
    
    # require the average lag to exceed twice the minimum acceptable cadence
    while np.max(fake_mjds)-np.min(fake_mjds) < nobs*2*min_lag:
        fake_mjds += np.random.lognormal(np.log(2*min_lag), 0.4, size=nobs)
        
    fake_mjds = np.array(sorted(fake_mjds))
    fake_mjds -= fake_mjds.min() - mjd_min
        
    # check the epoch separations to ensure a realistic distribution
    lags = fake_mjds[1:] - fake_mjds[:-1]
    n = 0
    while lags.max() > nobs*max_lag_frac or lags.min() < min_lag:
        last_mjd = mjd_min
        lag_inc = np.zeros(len(fake_mjds))
        now_inc = 0
        for imjd, MJD in enumerate(fake_mjds):
            if imjd == 0:
                continue
                
            lag = lags[imjd-1]
            if lag < min_lag:
                now_inc += min_lag*2
            elif lag > nobs*max_lag_frac:
                now_inc -= int(lag - nobs*max_lag_frac*0.95)
                
            lag_inc[imjd] = now_inc
            
        fake_mjds += lag_inc
        fake_mjds = np.array(sorted(fake_mjds))
        lags = fake_mjds[1:] - fake_mjds[:-1]
        n += 1
        
    return(fake_mjds)


def make_fake_obss(avg_shape=3, nobs=1000, nbin=512, low_noise=True, shape_change=False,
                   null_time=22, on_time=6, verb=True, plot_style='dark_background',
                   no_misalign=False, strong=False, save_plot=None, show_plot=False):
    """
    Make a 2D array representing fake observations with some profile shape and nulling parameters
    
    Input:
        avg_shape - int or numpy.array of floats (with length `nbin`)
        nobs - int, the number of observations to simulate
        nbin - int, the number of phase bins for each observation
        low_noise - bool, indicating whether the S/N of each observation is high
        shape_change - bool or str, use `False` to skip shape variations; `True` to select from
            ['qp', 'bi', 'bi_quick', 'random']; or specify a desired mode from the same list
        null_time - int or NoneType, indicating the mean number of days per null
            Use `None` or `0` to indicate no nulling behaviour
        on_time - int, the mean number of days where pulsar is on
            Ignore this parameter if no nulling behaviour
        verb - bool, whether to print/plot diagnostic information
        plot_style - str, a valid matplotlib 'style' (e.g., 'dark_background')
        no_misalign - bool, whether to prevent a phase shift from being added
        strong - bool, whether the shape variations should be painfully obvious
        save_plot - str or NoneType
        show_plot - bool
            
    Output:
        a 2D numpy.array of shape (nbin, nobs) representing the observations
        a 1D numpy.array of length (nobs) representing the observation epochs (MJDs)
        a 1D numpy.array of length (nobs) representing the integration times per observation
    
    """
    
    fake_mjds = sim_mjds(nobs)
    mjd_range = np.max(fake_mjds) - np.min(fake_mjds)
    
    # set the mean noise level (off-pulse rms)
    noise_rms = 0.02 if low_noise else 0.15
    
    # if no "average" profile is given, generate one with the desired number of components but no IP
    if type(avg_shape) is int:
        avg_prof, comps_cens, comps_wids, comps_hgts = make_fake_profile(avg_shape, None, nbin, no_ip=True)
    else: # if a profile is given, find the rough parameters assuming it is a single Gaussian
        avg_prof = avg_shape
        # assume the template has only 1 component
        comps_cens = np.array([np.argmax(avg_prof)])
        width, _ = find_eq_width_snr(avg_prof)
        comps_wids = np.array([width])/4
        comps_hgts = np.array([np.max(avg_prof)])
        
    # if profile variation is desired, determine the method and parameters
    if shape_change:
        if strong:
            neigs = 4
            min_eigs = 2
            eig_height = 0.6
        else:
            neigs = 3
            min_eigs = 1
            eig_height = 0.3
            
        # generate up to 3 "eigenvectors"
        eigs = np.array([np.zeros(nbin) for A in range(neigs+min_eigs)])
        used_prof_comps = []
        # run this loop min_eigs to neigs+min_eigs times for different eigenvectors
        for icomp_eig in np.arange(np.random.randint(neigs)+min_eigs):
            # randomly select one of the profile components to vary
            which_prof_comp = np.random.randint(len(comps_cens))
            used_prof_comps.append(which_prof_comp)
            
            # use the profile component parameters as a starting point
            width = np.random.lognormal(np.log(comps_wids[which_prof_comp]/4),
                                        comps_wids[which_prof_comp]/10)\
                    + comps_wids[which_prof_comp]/5
            
            # prevent the width from being too large which would create unreasonable profiles
            while width > nbin/3:
                width = np.random.lognormal(np.log(comps_wids[which_prof_comp]/4),
                                            comps_wids[which_prof_comp]/10)\
                        + comps_wids[which_prof_comp]/5
                
            centre = comps_cens[which_prof_comp]\
                     + int(np.random.normal(0, comps_wids[which_prof_comp]/2))
            eigs[icomp_eig] = add_gauss(eigs[icomp_eig], centre, width,
                                        height=eig_height*comps_hgts[which_prof_comp])
            
        used_prof_comps = np.array(used_prof_comps)
        
        # figure out which mode, describing the variations over time, to use
        mode_list = np.array(['qp', 'bi', 'bi_quick', 'random'])
        if shape_change is True:
            change_mode = mode_list[np.random.randint(3)]
        elif shape_change in mode_list:
            change_mode = shape_change
            
        if change_mode == 'qp': # quasi-periodic; the best for testing the GP stuff
            tau = np.random.randint(mjd_range/20, mjd_range/8) # random timescale between one-twentieth and one-eighth the timespan
            amp = np.array([np.random.normal(0.8, 0.2) for A in range(len(used_prof_comps))])
            offset = np.array([A/2 for A in amp])
            mode = None
            
            # define the function that will return "eigenvalues" as a function of MJD
            def eig_val_func(mjd, tau, amp, offset, nobs, mode=None):
                # randomly alter the timescale for each "observation"
                tau = np.random.normal(tau, 3*tau/nobs)
                
                # use a modified sinusoid with a constant offset (to prevent too much negative "emission")
                eig = amp*np.sin((mjd-fake_mjds.min())*2*np.pi/tau)\
                      + offset + np.random.normal(0, amp/5)
                
                return(eig, tau, None)
            
        elif 'bi' in change_mode: # bi-modal, shifting between two distinct states
            if 'quick' in change_mode: # timescale is much shorter
                tau = int(np.random.lognormal(0.5)+2*np.mean(fake_mjds[1:] - fake_mjds[:-1]))
            else:
                tau = np.random.randint(mjd_range/30, mjd_range/5) # random timescale between one-thirtieth and one-fifth the timespan
                
            amp = np.array([np.random.normal(1, 0.2) for A in range(len(used_prof_comps))])
            offset = np.array([None for A in range(len(used_prof_comps))])
            mode = 0

            # define the function that will return "eigenvalues" as a function of MJD
            def eig_val_func(mjd, tau, amp, offset, nobs, mode):
                # determine whether to change the state from the previous observation
                switch_mode = 1 if np.random.randint(tau) == 0 else 0
                if switch_mode == 1:
                    mode = 0 if mode == 1 else 1
                    
                # the resulting eigenvalue will be a value close to 0 or 1
                eig = np.random.normal(amp*mode, amp/10)
                return(eig, tau, mode)
            
        else: # randomly change shape with the given eigenvectors
            tau = None
            amp = np.array([None for A in range(len(used_prof_comps))])
            offset = np.array([0.4*comps_hgts[A] for A in used_prof_comps])
            mode = None

            # define the function that will return "eigenvalues" as a function of MJD
            def eig_val_func(mjd, tau, amp, offset, nobs, mode):
                eig = np.random.lognormal(sigma=0.7) - offset
                return(eig, tau, None)
            
        if verb:
            print("The mode is {} with a timescale of {} days, amplitudes of {}, and offsets of {}".format(change_mode, tau, amp, offset))
            try:
                plt.style.use(plot_style)
            except OSError:
                print("Cannot find given plot style; using default instead")
                plot_style = 'default'
                
            with plt.style.context(plot_style):
                plt.clf()
                fig = plt.figure(num=0)
                fig.set_size_inches(7, 5.5)
                fig.suptitle('Eigenvectors')
                for icomp_eig in range(len(eigs)):
                    plt.plot(eigs[icomp_eig])
                    
                plt.xlabel('Phase bins')
                plt.ylabel('Amplitude')
                
                if save_plot:
                    plt.savefig(save_plot, bbox_inches='tight')
        
                if show_plot:
                    plt.show()
                    
    # set some initial parameters for the "observations"
    data = np.zeros((nbin, nobs))
    now_on = True
    first_on = fake_mjds[0]
    last_on = fake_mjds[0]
    first_off = None
    last_off = None
    fake_tobs = np.zeros(len(fake_mjds))
    mean_tobs = 900
    misalign = 0
    for iobs in np.arange(nobs):
        # generate an "observation length" for each iteration
        if np.random.randint(10) >= 7:
            tobs = np.random.lognormal(np.log(mean_tobs*4), 0.3)
        else:
            tobs = np.random.lognormal(np.log(mean_tobs), 0.4)
            
        fake_tobs[iobs] = tobs
        
        # vary the noise for each obs and reduce the value based on the `tobs`
        off_rms = noise_rms*0.75*(np.random.lognormal(sigma=1.3)+0.9)*np.sqrt(mean_tobs/tobs)
        
        # for the first "observation", do not add nulling or profile varation, just white noise
        if iobs == 0:
            data[:,iobs] = avg_prof + np.random.normal(scale=off_rms, size=nbin)
            continue
        
        # this is the nulling section
        if null_time is not None and null_time > 0:
            # determine whether the pulsar is "on" based on timescales
            if last_on == fake_mjds[iobs-1]: # pulsar was on in last obs
                this_time = last_on - first_on
                
                # random number saying how long this 'on' phase should last
                check = np.random.normal(on_time, scale=on_time/3)
                if this_time < check: # keep the pulsar "on"
                    now_on = True
                    last_on = fake_mjds[iobs]
                else: # switching to null
                    now_on = False
                    first_off = fake_mjds[iobs]
                    last_off = fake_mjds[iobs]
                    
            else: # pulsar was off in last obs
                this_time = last_off - first_off
                check = np.random.normal(null_time, scale=null_time/3)
                if this_time < check: # pulsar is still in a null
                    now_on = False
                    last_off = fake_mjds[iobs]
                else:
                    now_on = True
                    first_on = fake_mjds[iobs]
                    last_on = fake_mjds[iobs]
                    
        if not no_misalign:
            # add a phase misalignment, increasing (or remaining constant) with each obs
            misalign += int(np.floor(np.random.lognormal(sigma=nbin/600))) if iobs != 0 else 0
                    
        if now_on:
            data[:,iobs] = np.roll(avg_prof + np.roll(np.random.normal(scale=off_rms, size=nbin), np.random.randint(nbin)), misalign)
            
            # add in the profile variation using the generated eigenvectors
            if shape_change:
                for icomp in range(len(used_prof_comps)):
                    # this is the function we defined for each mode above
                    eigval, tau, mode = eig_val_func(fake_mjds[iobs], tau, amp[icomp], offset[icomp], nobs, mode)
                    data[:,iobs] += np.roll(eigs[icomp]*eigval, misalign)
        else: # null, just make this one noise
            data[:,iobs] = np.roll(np.random.normal(scale=off_rms, size=nbin), np.random.randint(nbin))
    
    return(data, fake_mjds, fake_tobs)


def make_fake_profile(ncomp=2, noise_sigma=0.02, nbin=512, no_ip=None,
                      not_wide=False):
    """
    Produce a fake profile with the given number of Gaussian components
    If ncomp > 2, profile may (10% chance) have a main pulse and interpulse
    Can also guarantee presence (or absence) of an interpulse
    
    Input:
        ncomp - int, the number of distinct Gaussian components to compose
            the profile
        noise_sigma - float, the ''scale'' of the normal distribution for
            generating white noise on top of the components, relative to
            a profile height of roughly 1;
            `None` indicates a noiseless profile, just the Gaussian components
        nbin - int, the number of phase bins for the profile
        no_ip - bool or None, whether to prohibit or guarantee the presence
            of an interpulse (True/False) or leave it to chance (None)
        not_wide - bool, whether to prevent the components from being too wide
            
    Output:
        a 1D numpy array of length `nbin` representing the profile
        a 1D numpy array of length `ncomp` representing the component centres
        a 1D numpy array of length `ncomp` representing the component widths
        a 1D numpy array of length `ncomp` representing the component heights
    
    """
    
    prof = np.zeros(nbin)
    # put the centre of the main component at roughly phase 0.5
    centre = nbin/2
    width = 0.01*nbin
    height = 1
    
    cen_arr = np.zeros(ncomp)
    wid_arr = np.zeros(ncomp)
    hgt_arr = np.zeros(ncomp)
    
    # loop over the components to add random variations to the centre, width, and height
    for icomp in np.arange(ncomp)+1:
        centre += int(np.random.normal(scale=width*2))
        
        wid_sig = 0.4 if not_wide else 0.8
        width *= np.random.lognormal(sigma=wid_sig)
        wid_lim = nbin/5 if not_wide else nbin/3
        while width > wid_lim:
            width = 0.01*nbin*np.random.lognormal(sigma=wid_sig)
            
        height = (1/icomp) + np.random.randint(5)/10
        if icomp > 2:
            randint = 0 if np.random.randint(10) > 0 else 1
            if no_ip is True:
                randint = 0
            elif no_ip is False and icomp == ncomp:
                randint = 1
                
            centre += int(randint*0.5*nbin)
            centre = centre % nbin
        
        # this function generates the Gaussian component and adds to the existing profile
        prof = add_gauss(prof, centre, width, height)
        cen_arr[icomp-1] = centre
        wid_arr[icomp-1] = width
        hgt_arr[icomp-1] = height
        
    # add white noise to the profile
    if noise_sigma is not None:
        prof += np.random.normal(scale=noise_sigma, size=nbin)
    
    # return the profile with the arrays of component parameters
    return(prof, cen_arr, wid_arr, hgt_arr)


def add_gauss(prof, centre, width, height=1):
    """
    Generate a single Gaussian component with given parameters
    Simply apply the function:
        f(x) = A*exp(-0.5*((x-centre)/width)**2)
    to an array of x values
    
    """
       
    bins = np.arange(len(prof))
    new_comp = height*np.exp(-0.5 * ((bins - centre)/width)**2)
    
    return(prof+new_comp)


# function to bin the eigenvalues by averaging over given lengths of time, with some overlap
def bin_array(data, mjd_array, err_array, block=100, overlap=0.5):
    """
    Block length in days, the distance between midpoints
    The shape of the `data` array (X, Y) must match the length of the `mjd_array` in one dimension, eg, X
    The `mjd_array` must be the same length as the `err_array`
    The overlap is the fractional increase to the block length, overlapping adjacent bins (half for first and last bins)
    
    Return:
        an array of averaged values with shape (Z, Y) where Z < X(==len(mjd_array)) (or (X, Z) where Z < Y(==len(mjd_array)))
        an array with the same shape as the above, representing the standard error
        an array of MJDs representing the midpoint of each block/interval, with length == Z
        
    """
    
    if len(mjd_array) not in data.shape or len(mjd_array) != len(err_array):
        raise(ValueError("Mismatch in input array shapes/lengths!"))
        
    if overlap < 0:
        raise(ValueError("Overlap must be a positive float"))
        
    # check data array shape, and, if necessary, transpose the data array for ease later
    avg_axis = 0 if len(mjd_array) == data.shape[0] else 1
    keep_axis = 0 if avg_axis == 1 else 1
    if keep_axis == 0:
        data_new = data.T
    else:
        data_new = data
    
    # make arrays of time ranges to iterate over
    min_mjd = mjd_array.min()-0.5
    max_mjd = mjd_array.max()+0.5
    ints_min = np.arange(min_mjd, max_mjd, block)
    if max_mjd - ints_min[-1] < block/5: # adjust for very short interval at end
        min_mjd -= block/4
        ints_min = np.arange(min_mjd, max_mjd, block)
    
    #ints_med = ints_min + block/2
    ints_max = np.array([ints_min[i+1] for i in range(len(ints_min)-1)]+[max_mjd]) + block*overlap/2
    ints_min -= block*overlap/2
        
    #print(ints_min, ints_max)
    
    result = np.zeros((len(ints_min), data.shape[keep_axis]))
    errors = np.zeros((len(ints_min), data.shape[keep_axis]))
    ints_mean = np.zeros(len(ints_min))
    for M1, M2, count in zip(ints_min, ints_max, np.arange(len(ints_min))):
        lim = np.logical_and(mjd_array > M1, mjd_array < M2)
        if len(mjd_array[lim]) == 0:
            result[count,:] = np.nan
            errors[count,:] = np.nan
            ints_mean[count] = np.nan
        else:
            avg = np.average(data_new[lim,:], axis=0, weights=1/err_array[lim])
            err = np.std(data_new[lim,:], axis=0)/np.sqrt(len(data_new[lim, 0]))
            result[count, :] = avg
            errors[count, :] = err
            ints_mean[count] = np.average(mjd_array[lim], weights=1/err_array[lim])
        
    if keep_axis == 0:
        result = result.T
        errors = errors.T
        
    if result.shape[keep_axis] != data.shape[keep_axis] or result.shape[avg_axis] >= data.shape[avg_axis]:
        raise(ValueError("Resulting array has improper shape"))

    return(result, errors, ints_mean)


def err_eigval(profs, eigfuns, off_lims, verb=False):
    """
    Find the uncertainties on eigenvalues based on the off-pulse rms
    
    Input:
        profs - 2D array of floats, shape of (nbin, nobs), representing
            profiles from which eigenvectors and -values are derived
        eigfuns - 2D array of floats, shape of (ncomp, nbin), representing
            eigenvectors
        off_lims - 1D array of booleans, length of nbin, representing a
            mask to select only off-pulse bins
        verb - boolean, whether to print verbose testing information
        
    Output:
        2D array of floats, shape of (nobs, ncomp), representing uncertainties
    
    """
    
    nobs = profs.shape[1]
    ncomp = eigfuns.shape[0]
    errs_out = np.zeros((nobs, ncomp))
    
    count_nans = 0
    for icomp in range(ncomp):
        for iobs in range(nobs):
            off_rms = np.sqrt(np.mean(profs[off_lims,iobs]**2))
            val = off_rms**2*np.sum(np.abs(eigfuns[icomp,off_lims]))
            if val < 0:
                count_nans += 1
                val = np.random.lognormal(-2, 0.2)**2
                
            errs_out[iobs,icomp] = np.sqrt(val)
            
    if verb:
        print("The number of NaNs was", count_nans)
        
    return(errs_out)


def err_eigval_off(profs, eigfuns):
    """
    Input:
        profs - 2D array of floats, shape of (nbin, nobs), representing
            profiles
        eigfuns - 2D array of floats, shape of (ncomp, nbin), representing
            eigenvectors
            
    Output:
        2D array of floats, shape of (nobs, ncomp), representing uncertainties
    
    """
    
    nbin = profs.shape[0]
    nobs = profs.shape[1]
    ncomp = eigfuns.shape[0]
    if nbin < 30:
        raise(RuntimeError("Data too short for this method"))
        
    nbin_try = int(nbin/5)
    ntry = 0
    bins = np.arange(nbin)
    bins_tried = []
    errs_all_out = np.zeros((nobs, ncomp, 20))
    while ntry < 20:
        bin_start = np.random.randint(nbin-nbin_try)
        if len(bins_tried) > 0:
            while bin_start in bins_tried:
                bin_start = np.random.randint(nbin-nbin_try)
        
        bins_tried.append(bin_start)
        errs_all_out[:,:,ntry] = err_eigval(profs, eigfuns, np.logical_and(bins > bin_start, bins < bin_start+nbin_try), verb=False)
        
        ntry += 1
        
    errs_mean = np.mean(errs_all_out, axis=2)
    return(errs_mean)


if 'nm' in globals():
    # a better function using David's nulling_mcmc code
    # github.com/dlakaplan/nulling-pulsars
    def check_null_prob(data, peak_bin=100, ip=False, on_min=None, onf_range=None, off_min=None):
        """
        Get a decent probability of an observation being a null using Bayesian statistics
        Uses the intensity of the on-pulse region (near `peak_bin`) compared with that of the off-pulse region
        The off-pulse region is assumed to be 1/2 of a rotation away from the `peak_bin` value if `ip` is False,
            1/4 of a rotation if `ip` is True
    
        Input:
            data - a 2D array with shape (nbin, nobs) representing observed profiles (assume aligned)
            peak_bin - an int representing the bin with the highest signal
            ip - a boolean indicating whether there is an interpulse
            on_min - an int (or NoneType) representing the bin starting the on-pulse region
            onf_range - an int (or NoneType) representing the length of the on- and off-pulse regions
            off_min - an int (or NoneType) representing the bin starting the off-pulse region
            
        Output:
            a 1D array with length `nobs`, representing the probabilities
    
        """
    
        if on_min is None:
            on_min = int(peak_bin - data.shape[0]/8)
            onf_range = int(data.shape[0]/4)
            if ip:
                off_min = int(peak_bin + data.shape[0]/8)+1
            else:
                off_min = int(peak_bin + 3*data.shape[0]/8)
        
        on_ints = np.sum(data[on_min:on_min+onf_range,:], axis=0)
        off_ints = np.sum(data[off_min:off_min+onf_range,:], axis=0)
    
        NP = nm.NullingPulsar(on_ints, off_ints)
        means_fit, _, stds_fit, _, weights_fit, _, _, _ = NP.fit_mcmc()
        null_prob = NP.null_probabilities(means_fit, stds_fit, weights_fit, NP.on)
        return(null_prob)


# function stolen and modified from Aris' psrshape/PulseShape.py
def get_gp(data, mjds, kern_len, errs=None, prior_min=200, prior_max=2000,
           bchain=300, pchain=2000, nwalkers=100, mcmc=False, multi=False, verb=False):
    """
    Use george to model a Gaussian Process, with parameters optimised linearly or with emcee
    Input:
        data - a 1D numpy array representing the measured data to be fit
        mjds - a 1D numpy array of the same length as `data`, representing observation epochs
        kern_len - a number representing the length scale of the GP kernel
        errs - a 1D numpy array of the same length as `data`, representing the uncertainties on the measured data
        prior_min - a float representing the minimum bound of the length scale (linear in days)
        prior_max - a float representing the maximum bound of the length scale (linear in days)
        long_chain - boolean
        bchain - int
        pchain - int
        nwalkers - int
        mcmc - a boolean indicating whether to use MCMC
        multi - a boolean indicating whether to use multiprocessing
        verb - bool, whether to be verbose with diagnostic information

    Output:
        a george.GP object with the best parameters after maximising the log likelihood
        a 2D array of samples from MCMC (None object if `mcmc` is False)
    
    """
    
    import gp_init_threads as th_init
    
    # Define functions for fitting using globals (gp and data)
    # Define the objective function (negative log-likelihood in this case)
    def get_prob_funs(p_min, p_max):
        def neg_log_like(p):
            l = p[-1]
            if l < np.log(p_min) or l > np.log(p_max):
                return(1e25)
        
            gp.set_parameter_vector(p)
            log_like = gp.log_likelihood(data, quiet=True)
            result = -log_like if np.isfinite(log_like) else 1e25
            return(result)
    
        def logprob(p):
            # Trivial uniform prior
            if p[-1] < np.log(p_min) or p[-1] > np.log(p_max):
                return(-np.inf)
            
            if np.any((-100 > p[1:]) + (p[1:] > 100)):
                return(-np.inf)
        
            # Update the kernel and compute the lnlikelihood
            gp.set_parameter_vector(p)
            return(gp.log_likelihood(data, quiet=True))
        
        return(neg_log_like, logprob)
    
    # And the gradient of the objective function
    def grad_neg_log_like(p):
        gp.set_parameter_vector(p)
        return(-gp.grad_log_likelihood(data, quiet=True))
    
    def do_sampling(gp, pool, func, nwalkers=100, burn_chain=300, prod_chain=2000,
                    use_errs=False, verb=False):
        ndim = len(gp)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, pool=pool)
    
        # Initialize the walkers
        p0 = gp.get_parameter_vector() + np.random.randn(nwalkers, ndim)
        if verb:
            print("Running burn-in")
            
        s = sampler.run_mcmc(p0, burn_chain, progress=verb)
        samp0 = s[0]
        #lp = s[1]
        
        #samp0 = samp0[np.argmax(lp)] + 0.1*np.random.randn(nwalkers, ndim)
        #sampler.reset()
        #if verb:
        #    print("Running second burn-in")
            
        #s = sampler.run_mcmc(p0, burn_chain, progress=verb)
        samp0 = s[0]
        sampler.reset()

        if verb:
            print("Running production chain")
            
        sampler.run_mcmc(samp0, prod_chain, progress=verb)
        return(sampler, ndim)
        
    variance = np.var(data)
    kernel = variance * kernels.Matern52Kernel(kern_len)#, metric_bounds=dict(log_M_0_0=(np.log(prior_min), np.log(prior_max))))
    if 'gp' in locals() or 'gp' in globals():
        del gp
        
    neg_log_like, logprob = get_prob_funs(prior_min, prior_max)
    gp = george.GP(kernel, np.mean(data), fit_mean=True, solver=george.HODLRSolver,
                   white_noise=np.log(np.sqrt(variance)), fit_white_noise=True)
        
    use_errs = False
    if errs is not None:
        use_errs = True
        #gp = george.GP(kernel, np.mean(data), fit_mean=True, solver=george.HODLRSolver)
        gp.compute(mjds, errs)
        #print(gp.get_parameter_vector())
    else:
        gp.compute(mjds)
        #raise(RuntimeError("Break"))
        
    if verb:
        print("The initial parameter vector is", gp.get_parameter_vector())
    
    if not mcmc: # no multiprocessing here...
        # Run the optimization routine
        p0 = gp.get_parameter_vector()
            
        results = op.minimize(neg_log_like, p0, jac=grad_neg_log_like, method="L-BFGS-B")
        # Update the kernel
        gp.set_parameter_vector(results.x)
        flat_samples = None
        
    else:
        # Set up the sampler
        if multi:
            proc_count = mpr.cpu_count() - 1
            with mpr.Pool(proc_count, initializer=th_init.init_thread, initargs=(kern_len, data, mjds, errs)) as pool:
                sampler, ndim = do_sampling(gp, pool, th_init.lnprob, nwalkers, bchain, pchain, use_errs, verb)
                
        else:
            sampler, ndim = do_sampling(gp, None, logprob, nwalkers, bchain, pchain, use_errs, verb)
        
        # find the burn-in time
        try:
            tau_a = sampler.get_autocorr_time(quiet=True)
            if verb:
                print("The array of autocorrelation times is", tau_a)
                
            lim = np.array([np.isfinite(A) for A in tau_a])
            if not np.any(lim):
                print("Invalid autocorrelation times; using default of 50")
                tau = 50
            else:
                tau = np.mean(tau_a[lim])
                
        except emcee.autocorr.AutocorrError as e:
            print(str(e))
            return(None, None)
        
        flat_samples = sampler.get_chain(discard=max(300, int(np.ceil(tau*2.5))), thin=int(np.floor(tau/2)), flat=True)
        # get the median values for each parameter
        p_final = np.zeros(ndim)
        for i in range(ndim):
            p_final[i] = np.percentile(flat_samples[:, i], 50)
        
        gp.set_parameter_vector(p_final)
    
    del data
    return(gp, flat_samples)


def get_gp_cel(data, mjds, kern_len, errs=None, prior_min=300, prior_max=5000,
               bchain=300, pchain=2000, nwalkers=100, multi=False, verb=False, logg=None):
    """
    Use celerite to model a Gaussian Process that is fit using emcee
    
    Input:
        data - a 1D numpy array representing the measured data to be fit
        mjds - a 1D numpy array of the same length as `data`, representing observation epochs
        kern_len - a number representing the length scale of the GP kernel
        errs - a 1D numpy array of the same length as `data`, representing the uncertainties on the measured data
        prior_min - a float representing the minimum bound of the length scale (linear in days)
        prior_max - a float representing the maximum bound of the length scale (linear in days)
        bchain - int
        pchain - int
        nwalkers - int
        multi - a boolean indicating whether to use multiprocessing
        verb - bool, whether to be verbose with diagnostic information
        logg - logging object

    Output:
        a celerite.GP object with the best parameters after maximising the log likelihood
        a 2D array of samples from MCMC (None object if `mcmc` is False)
    
    """
    
    if 'gp_init_threads' in sys.modules:
        del sys.modules['gp_init_threads']
        if 'th_init' in locals() or 'th_init' in globals():
            del th_init
    
    try:
        implib.invalidate_caches()
        th_init = implib.reload(th_init)
    except NameError:
        import gp_init_threads as th_init
    
    # Define functions for fitting using globals (gp and data)
    # Define the objective function (negative log-likelihood in this case)
    def get_prob_funs(p_min, p_max):
        def neg_log_like(p):
            l = p[1]
            if l < np.log(p_min) or l > np.log(p_max):
                return(1e25)
        
            gp.set_parameter_vector(p)
            log_like = gp.log_likelihood(data, quiet=True)
            result = -log_like if np.isfinite(log_like) else 1e25
            return(result)
    
        def logprob(p):
            # Trivial uniform prior
            if p[1] < np.log(p_min) or p[1] > np.log(p_max):
                return(-np.inf)
            
            if np.any((-30 > p) + (p > 30)):
                return(-np.inf)
        
            # Update the kernel and compute the log-likelihood
            gp.set_parameter_vector(p)
            return(gp.log_likelihood(data, quiet=True))
        
        return(neg_log_like, logprob)
    
    # And the gradient of the objective function
    def grad_neg_log_like(p):
        gp.set_parameter_vector(p)
        return(-gp.grad_log_likelihood(data, quiet=True))
    
    def do_sampling(gp, pool, func, nwalkers=100, burn_chain=300, prod_chain=2000, use_errs=False, verb=False, logg=None):
        ndim = len(gp)            
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, pool=pool)
    
        # Initialize the walkers
        p0 = gp.get_parameter_vector() + 0.5*np.random.randn(nwalkers, ndim)
        if logg is not None:
            logg.info("Running burn-in")
        elif verb:
            print("Running burn-in")
            
        s = sampler.run_mcmc(p0, burn_chain, progress=verb)
        samp0 = s[0]
        lp = s[1]
        
        samp0 = samp0[np.argmax(lp)] + 0.1*np.random.randn(nwalkers, ndim)
        sampler.reset()
        if logg is not None:
            logg.info("Running second burn-in")
        elif verb:
            print("Running second burn-in")
            
        s = sampler.run_mcmc(p0, burn_chain, progress=verb)
        samp0 = s[0]
        sampler.reset()

        if logg is not None:
            logg.info("Running production chain")
        elif verb:
            print("Running production chain")
            
        sampler.run_mcmc(samp0, prod_chain, progress=verb)
        return(sampler, ndim)
        
    variance = np.var(data)
    term1 = cel.terms.Matern32Term(np.log(1), np.log(kern_len),
                                   bounds=dict(log_sigma=(-15, 15), log_rho=(np.log(prior_min), np.log(prior_max))))
    term2 = cel.terms.JitterTerm(np.log(np.sqrt(variance)), bounds=dict(log_sigma=(-15, 15)))
    kernel = term1 + term2
    if 'gp' in locals() or 'gp' in globals():
        del gp
        
    neg_log_like, logprob = get_prob_funs(prior_min, prior_max)
    gp = cel.GP(kernel, np.mean(data), fit_mean=True)
        
    use_errs = False
    if errs is not None:
        use_errs = True
        gp.compute(mjds, errs)
    else:
        gp.compute(mjds)
        
    if logg is not None:
        logg.info("The initial parameter vector is {}".format(gp.get_parameter_vector()))
    elif verb:
        print("The initial parameter vector is", gp.get_parameter_vector())
    
    # Set up the sampler
    if multi:
        proc_count = mpr.cpu_count() - 1
        with mpr.Pool(proc_count, initializer=th_init.init_thread, initargs=(kern_len, data, mjds, errs)) as pool:
            sampler, ndim = do_sampling(gp, pool, th_init.lnprob, nwalkers, bchain, pchain, use_errs, verb, logg)
                
    else:
        sampler, ndim = do_sampling(gp, None, logprob, nwalkers, bchain, pchain, use_errs, verb, logg)
        
    # find the burn-in time
    try:
        tau_a = sampler.get_autocorr_time(quiet=True)
        if logg is not None:
            logg.info("The array of autocorrelation times is {}".format(tau_a))
        elif verb:
            print("The array of autocorrelation times is", tau_a)
                
        lim = np.array([np.isfinite(A) for A in tau_a])
        if not np.any(lim):
            if logg is not None:
                logg.warning("Invalid autocorrelation times; using default of 50")
            else:
                print("Invalid autocorrelation times; using default of 50")
                
            tau = 50
        else:
            tau = np.mean(tau_a[lim])
                
    except emcee.autocorr.AutocorrError as e:
        print(str(e))
        return(None, None)
        
    flat_samples = sampler.get_chain(discard=max(bchain, int(np.ceil(tau*2.5))), thin=int(np.floor(tau/2)), flat=True)
    # get the median values for each parameter
    p_final = np.zeros(ndim)
    for i in range(ndim):
        p_final[i] = np.percentile(flat_samples[:, i], 50)
        
    gp.set_parameter_vector(p_final)
    
    del data
    return(gp, flat_samples)


def run_each_gp(data, mjds_in, errs=None, kern_len=300, max_num=4, prior_min=200, prior_max=2000,
                burn_chain=300, prod_chain=5000, num_walkers=100,
                plot_gps=True, mcmc=True, multi=True, plot_chains=False, plot_corner=False,
                gp_plotname=None, bk_bgd=False, verb=True, logg=None, plot_dir=None, show_plots=True,
                mjds_pred=None, descrpn=''):
    """
    Input:
        data - 2D array of floats, shape of (nobs, ncomp)
        mjds_in - 1D array of floats, length of nobs
        errs - 2D array of floats, shape of (nobs, ncomp)
        kern_len - float
        max_num - int
        prior_min - float
        prior_max - float
        burn_chain - int
        prod_chain - int
        num_walkers - int
        plot_gps - bool
        mcmc - bool
        multi - bool
        plot_chains - bool
        plot_corner - bool
        gp_plotname - str or NoneType, file name to use for the plot of resulting GPs
        bk_bgd - bool, whether to use a dark background for the plots
        verb - bool, whether to be verbose with diagnostic information
        logg - logging object
        plot_dir - str or NoneType
        show_plots - bool
        mjds_pred - 1D array of floats or NoneType
        descrpn - str, a description string to append to all output file names (plots)
        
    Output:
        2D array of floats, shape of (max_num+1, nobs) or (ncomp, nobs) if max_num is `None`
        2D array of floats, same shape as previous variable
        1D array of floats, length of nobs
    
    """
    
    write_gp_init_threads(prior_min, prior_max, use_george=not(mcmc))
    
    if descrpn != '' and descrpn[-1] != '_':
        descrpn += '_'
    
    # subtract the min. of the `mjds`
    mjds_off = int(np.floor(mjds_in.min()))
    if logg is not None:
        logg.info("Subtracting {} from MJDs (will return true MJD values)".format(mjds_off))
    elif verb:
         print("Subtracting {} from MJDs (will return true MJD values)".format(mjds_off))
        
    mjds = mjds_in - mjds_off
    if mjds_pred is None:
        mjds_pred = np.arange(np.ceil(mjds.max()+1))
    elif mjds_pred.min() > 30000:
        mjds_pred -= mjds_off
        
    # initialise arrays
    if max_num is None:
        pred_res = np.zeros((data.shape[1], len(mjds_pred)))
        pred_vars = np.zeros((data.shape[1], len(mjds_pred)))
    else:
        pred_res = np.zeros((max_num+1, len(mjds_pred)))
        pred_vars = np.zeros((max_num+1, len(mjds_pred)))
        
    # loop over eigenvectors
    for eignum, eigval in enumerate(data.T):
        val_errs = None
        if errs is not None:
            val_errs = errs[:,eignum]
            
        # the heavy lifting
        if not mcmc: #keeping this for now
            gp, flat_samps = get_gp(eigval, mjds, kern_len, val_errs, prior_min, prior_max,
                                    bchain=burn_chain, pchain=prod_chain,
                                    nwalkers=num_walkers, mcmc=mcmc, multi=multi, verb=verb, logg=logg)
        else:
            gp, flat_samps = get_gp_cel(eigval, mjds, kern_len, val_errs, prior_min, prior_max,
                                        bchain=burn_chain, pchain=prod_chain,
                                        nwalkers=num_walkers, multi=multi, verb=verb, logg=logg)
        
        #print(gp.get_parameter_dict())
        if plot_gps or plot_chains or plot_corner:
            if bk_bgd:
                plot_style = 'dark_background'
                best_colour = cmr.chroma(0.85)
            else:
                plot_style = 'default'
                best_colour = 'k'
        
        # helpful diagnostic plots
        if plot_chains and flat_samps is not None:
            with plt.style.context(plot_style):
                plt.clf()
                fig = plt.figure(num=eignum+1)
                fig.set_size_inches(10, 4)
                ax = fig.gca()
                ax.plot(flat_samps[:,1], '-', color=best_colour)
                ax.set_xlabel('steps')
                ax.set_ylabel('log_rho')
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                ax.hlines([np.log(prior_min), np.log(prior_max)], xlims[0], xlims[1], linestyles='dashed')
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                if plot_dir is not None:
                    plt.savefig(os.path.join(plot_dir, descrpn+'gp_chain_{}.png'.format(eignum)), bbox_inches='tight')
                    if logg is not None:
                        logg.info("GP chain {} plot saved to {}".format(eignum, os.path.join(plot_dir, descrpn+'gp_chain_{}.png'.format(eignum))))
            
                if show_plots:
                    plt.show()
                    
        if plot_corner and flat_samps is not None:
            with plt.style.context(plot_style):
                plt.clf()
                gp_names = gp.get_parameter_names()
                gp_bounds = gp.get_parameter_bounds()
                if logg is not None:
                    logg.info("The bounds returned by the attribute are {}".format(gp_bounds))
                elif verb:
                    print("The bounds returned by the attribute are", gp_bounds)
                    
                #gp_bounds = [(-11.5, -9.5), (-10, -8), (4, 12)]
                #gp_bounds[1] = (np.log(prior_min), np.log(prior_max))
                gp_bounds = [(-15, 15), (np.log(prior_min), np.log(prior_max)), (-15, 15), (-0.05, 0.05)]
                for q_num, val in enumerate(gp_bounds):
                    if q_num == 1:
                        suff = 'st'
                    elif q_num == 2:
                        suff = 'nd'
                    elif q_num == 3:
                        suff = 'rd'
                    else:
                        suff = 'th'
                        
                    flats = flat_samps[:,q_num]
                    if round(min(flats), 2) < round(val[0]-0.005, 2):
                        if logg is not None:
                            logg.warning("The {:d}{:s} lower bound was breached, reaching a min. of {:.2f}".format(q_num, suff, min(flats)))
                        else:
                            print("The {:d}{:s} lower bound was breached, reaching a min. of {:.2f}".format(q_num, suff, min(flats)))

                    diff_min = abs(min(flats) - val[0])
                    min_val = min(max(val[0], min(flats) - 0.1*diff_min), np.median(flats))
                        
                    if round(max(flats), 2) > round(val[1]+0.005, 2):
                        if logg is not None:
                            logg.warning("The {:d}{:s} upper bound was breached, reaching a max. of {:.2f}".format(q_num, suff, max(flats)))
                        else:
                            print("The {:d}{:s} upper bound was breached, reaching a max. of {:.2f}".format(q_num, suff, max(flats)))
                    
                    diff_max = abs(max(flats) - val[1])
                    max_val = max(min(val[1], max(flats) + 0.1*diff_max), np.median(flats))
                        
                    gp_bounds[q_num] = (min_val, max_val)
                    
                corner.corner(flat_samps, labels=gp_names, range=gp_bounds, 
                              color=best_colour)
                if plot_dir is not None:
                    plt.savefig(os.path.join(plot_dir, descrpn+'gp_corner_{}.png'.format(eignum)), bbox_inches='tight')
                    if logg is not None:
                        logg.info("GP corner plot num. {} saved to {}".format(eignum, os.path.join(plot_dir, descrpn+'gp_corner_{}.png'.format(eignum))))

                if show_plots:
                    plt.show()

        if mjds_pred.min() < mjds.min()+100-mjds_off:
            mjds_pred += mjds_off
        elif mjds_pred.max() > mjds.max()-100+mjds_off:
            mjds_pred -= mjds_off
            
        pred, pred_var = gp.predict(eigval, mjds_pred, return_var=True)
        pred_res[eignum,:] = pred
        pred_vars[eignum,:] = pred_var
        
        del gp
        
        # stop the loop after reaching the desired number of eigenvectors
        if eignum == max_num:
            break
            
    # plot the results
    if plot_gps:
        if max_num is None:
            max_num = data.shape[1]
            
        plot_eig_gp(mjds_pred, pred_res[:max_num+1,:], pred_vars[:max_num+1,:],
                    mjds_off, mjds, data[:,:max_num+1], errs, savename=gp_plotname,
                    bk_bgd=bk_bgd, show=show_plots)
        if logg is not None:
            logg.info("Plot of GPs for all components saved to "+gp_plotname)
                
    if mjds_pred.min() < 20000:
        mjds_pred += mjds_off
        
    return(pred_res, pred_vars, mjds_pred)


def plot_eig_gp(mjds_pred, pred_res, pred_var, mjd_offset=None,
                mjds=None, data=None, data_errs=None,
                savename=None, bk_bgd=False, show=False,
                nudot_mjds=None, nudot_vals=None, nudot_vars=None, eig_nums=None):
    """
    Make a set of panels (numbering `ncomp_short`) of eigenvalues per component and fitted GPs
    Each panel contains the actual eigenvalues (with uncertainties) for each observation and
    the median GP with an error region
    
    Input:
       mjds_pred - 1D array of floats, length of nobs_long, representing a uniform set of
           MJDs spanning the same range as `mjds` (nobs_long >= nobs)
       pred_res - 2D array of floats, shape of (ncomp_short, nobs_long), representing the median
           GPs per eigenvector analysed (ncomp_short <= ncomp)
       pred_var - 2D array of floats, shape of (ncomp_short, nobs_long), representing the
           uncertainties on the GPs per eigenvector analysed
       mjds - 1D array of floats, length of nobs, representing the real observation MJDs
       data - 2D array of floats, shape of (nobs, ncomp), representing the real eigenvalues
       data_errs - 2D array of floats, shape of (nobs, ncomp), representing the uncertainties
           on the real eigenvalues
        mjd_offset - float or NoneType, the zero-point that was subtracted from the real MJDs
       savename - str or NoneType, the file name for saving the plot (or `None` to not save it)
       bk_bgd - bool, whether to use a dark background for the plot
       nudot_mjds - 
       nudot_vals - 
       nudot_vars - 
       eig_nums - 
    
    """
    
    #plt.rcParams['text.usetex'] = True
    
    if bk_bgd:
        style = 'dark_background'
        cmap = cmr.chroma_r
        lc = 'w'
        k_alpha = 0.6
    else:
        style = 'default'
        cmap = cmr.chroma
        lc = 'k'
        k_alpha = 0.4
        
    if mjd_offset is not None:
        xlab = "MJD - {} (day)".format(mjd_offset)
    else:
        xlab = "MJD (day)"
        
    c3 = cmap(0.53)
    
    with plt.style.context(style):
        plt.clf()
        fig = plt.figure(num=1)
        fig.set_size_inches(6, 4)
        title_text = "GP results for {} eigenvalues{}"
        if nudot_vals is not None:
            adj = "correlated"
            mod = " and $\dot \\nu$"
        else:
            adj = "relevant"
            mod = ""
            
        fig.suptitle(title_text.format(adj, mod), fontsize=14)
        if data is None:
            plot_data = False
        else:
            plot_data = True

        if data is None:
            data = np.zeros(pred_res.T.shape)
            mjds = np.zeros(len(mjds_pred))

        if data_errs is None:
            d_errs = np.zeros(data.shape)
        elif data_errs is not None:
            d_errs = data_errs

        num_panels = 1 if len(pred_res.shape) == 1 else pred_res.shape[0]
        if nudot_vals is not None:
            num_panels += 1
            
        if len(pred_res.shape) > 1:
            fig.set_size_inches(6, 8)
            # plot multiple eigenvalues in separate panels
            # define axes parameters
            w = 0.92 # width
            l = 0.05 # left
            b = 0.05 # bottom
            sep = 0.05 # separation
            # h*n+sep*(n-1) = 0.87
            #frac = 0.5
            #h1 = (0.87 - sep)/(1 + frac)
            #h2 = h1*frac
            h = (0.87 - (sep*(num_panels-1)))/num_panels # height
            #b1 = b2 + h + sep
            axes_list = np.array([None for a in range(num_panels)])
        
            num_ar = np.arange(num_panels) if nudot_vals is None else np.arange(num_panels-1)
            for pan_num, preds, predv, eigv, eiger in zip(num_ar, np.flip(pred_res, axis=0), np.flip(pred_var, axis=0),
                                                          np.flip(data, axis=1).T, np.flip(d_errs, axis=1).T):
                if pan_num == 0:
                    ax = fig.add_axes((l, b, w, h))
                    ax.set_xlabel(xlab, fontsize=12)
                else:
                    ax1 = axes_list[0]
                    ax = fig.add_axes((l, b + pan_num*h + pan_num*sep, w, h), sharex=ax1)
                
                ax.set_ylabel('Eigenvalue', fontsize=12)
                axes_list[pan_num] = ax
                ax.fill_between(mjds_pred, preds - np.sqrt(predv), preds + np.sqrt(predv),
                                 color=lc, alpha=k_alpha, zorder=10)
                ax.plot(mjds_pred, preds, lc, lw=1.5, zorder=20)
                if plot_data:
                    if data_errs is not None:
                        ax.errorbar(mjds, eigv, yerr=eiger, fmt='k.', mfc=c3, mec=c3, ecolor=c3, ms=8, zorder=1)
                    else:
                        ax.plot(mjds, eigv, 'k.', color=c3, ms=8, zorder=1)
                        
                if eig_nums is not None:
                    ylims = ax.get_ylim()
                    mod_val = (ylims[1] - ylims[0])/10
                    ax.set_ylim(ylims[0], ylims[1]+mod_val)
                    ax.text(0.035, 0.88, "$N_{{eig}}$ = {}".format(np.flip(eig_nums)[pan_num]), transform=ax.transAxes)
                   
            if nudot_vals is not None:
                pan_num += 1
                ax1 = axes_list[0]
                ax = fig.add_axes((l, b + pan_num*h + pan_num*sep, w, h), sharex=ax1)
                ax.set_ylabel('$\dot \\nu$ (Hz/s)', fontsize=12)
                if nudot_vars is not None:
                    ax.fill_between(nudot_mjds, nudot_vals - np.sqrt(nudot_vars), nudot_vals + np.sqrt(nudot_vars),
                                    color=lc, alpha=k_alpha, zorder=10)
                    
                ax.plot(nudot_mjds, nudot_vals, lc, lw=1.5, zorder=20)
                        
        elif num_panels > 1:
            # make two panels but only one is eig GP
            pass
        
        else:
            plt.fill_between(mjds_pred, pred_res - np.sqrt(pred_var), pred_res + np.sqrt(pred_var),
                             color=lc, alpha=k_alpha)
            plt.plot(mjds_pred, pred_res, lc, lw=1.5)
            if plot_data:
                if data_errs is not None:
                    plt.errorbar(mjds, data, yerr=d_errs, fmt='k.', mfc=c3, mec=c3, ecolor=c3, ms=8, zorder=1)
                else:
                    plt.plot(mjds, data, 'k.', color=c3, ms=8, zorder=1)
        
            plt.ylabel('Eigenvalue', fontsize=12)
            plt.xlabel(xlab, fontsize=12)
    
        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
        
        if show:
            plt.show()


def plot_recon_profs(mean_prof, eigvecs, mjds_pred, pred_reses, psrname, mjds_real=None, sub_mean=True,
                     bk_bgd=False, savename=None, show=True):
    """
    A function to reconstruct profiles from eigenvectors and predicted eigenvalues and make a waterfall-type plot

    Note that the first dimensions of `eigvecs` and `pred_reses` need not be equal,
    as long as `nvec_1` >= `nvec` and the first `nvec` elements are matched between the arrays

    Input:
        mean_prof - 1D array representing the mean profile (length of nbin)
        eigvecs - 2D array of eigenvectors with shape (nvec_1, nbin)
        mjds_pred - 1D array of MJDs
        pred_reses - 2D array of predicted eigenvalues with shape (nvec, len(mjds_pred))
        psrname - str, the name of the source being plotted
        mjds_real - a 1D array representing the real MJDs of observations used to produce model
        sub_mean - bool, whether to subtract the mean profile from the total, 
            i.e., to just plot the summed eigenvectors
        bk_bgd - bool, whether to use a dark background for the plot
        savename - str or NoneType, name of the file to save the plot
            Use `None` to not save the plot to disk (just show the plot)
        show - bool, whether to show the plot instead of just saving it
    
    """
    
    if bk_bgd:
        style = 'dark_background'
        cmap = cmr.chroma
        lc = 'w'
        if sub_mean:
            cmap = cmr.iceburn
    else:
        style = 'default'
        cmap = cmr.chroma_r
        lc = 'k'
        if sub_mean:
            cmap = cmr.fusion_r
            
    with plt.style.context(style):
        plt.clf()
        fig = plt.figure(num=1)
        fig.set_size_inches(5, 9)
        if sub_mean:
            subbed = " (minus the mean)"
        else:
            subbed = ""
        
        plt.title("Reconstructed profiles{} for {}".format(subbed, psrname), fontsize=13)
    
        nvec = pred_reses.shape[0]
        nbin = eigvecs.shape[1]
        
        prof_data = np.zeros((len(mjds_pred), nbin))
        for iday in range(len(mjds_pred)):
            if not sub_mean:
                prof_data[iday,:] = mean_prof
            
            for ivec in range(nvec):
                prof_data[iday,:] += (eigvecs[ivec,:]*pred_reses[ivec,iday])
    
        ymin = mjds_pred.min()
        ymax = mjds_pred.max()
        vmin = np.min(prof_data)
        vmax = np.max(prof_data)
        extend = None
        if sub_mean:
            vmax = np.percentile(abs(prof_data), 99.875)
            vmin = -vmax
            if vmin > np.min(prof_data) and vmax < np.max(prof_data):
                extend = 'both'
            elif vmin > np.min(prof_data):
                extend = 'min'
            elif vmax < np.max(prof_data):
                extend = 'max'
    
        extent = (0, nbin, ymin, ymax)
        p = plt.imshow(prof_data, vmin=vmin, vmax=vmax, cmap=cmap,
                       origin='lower', extent=extent, aspect='auto',
                       interpolation='nearest')
        fig.colorbar(p, extend=extend, fraction=0.05)
        plt.ylabel('MJD (day)', fontsize=11)
        plt.xlabel('Phase bin', fontsize=11)
        
        if mjds_real is not None:
            plt.hlines(mjds_real, xmin=5*nbin/250, xmax=25*nbin/250, linestyle='solid', colors='grey',
                       linewidth=0.5)

        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
        
        if show:
            plt.show()



def write_gp_init_threads(prior_min, prior_max, use_george=False):
    std_text = """##  Author: Renee Spiewak


import numpy as np
import {}


def init_thread(kern_len, _data, mjds, errs=None):
    global gp
    global data
    data = _data
    variance = np.var(data)
    {}
    {}
    kernel = {}
    gp = {}
    if errs is not None:
        gp.compute(mjds, errs)
    else:
        gp.compute(mjds)
        
    return()


def lnprob(p):
    global gp
    global data
    # Trivial uniform prior on length scale
    if p[1] < np.log({:.2f}) or p[1] > np.log({:.2f}):
        return(-np.inf)
            
    if np.any((-30 > p) + (p > 30)):
        return(-np.inf)
        
    # Update the kernel and compute the log-likelihood
    gp.set_parameter_vector(p)
    return(gp.log_likelihood(data, quiet=True))
    
"""
    if use_george:
        imp_mod = "george"
        kernel = 'variance * george.kernels.Matern52Kernel(kern_len)'
        gp = "george.GP(kernel, np.mean(data), fit_mean=True, solver=george.HODLRSolver, "\
             +"white_noise=np.log(np.sqrt(variance)*0.8), fit_white_noise=True)"
        term1 = ""
        term2 = ""
    else:
        imp_mod = "celerite as cel"
        logr = "log_rho=(np.log({:.2f}), np.log({:.2f}))".format(prior_min, prior_max)
        term1 = "term1 = cel.terms.Matern32Term(np.log(1), np.log(kern_len), bounds=dict(log_sigma=(-15, 15), {}))".format(logr) 
        term2 = "term2 = cel.terms.JitterTerm(np.log(np.sqrt(variance)), bounds=dict(log_sigma=(-15, 15)))"
        kernel = "term1 + term2"
        gp = "cel.GP(kernel, np.mean(data), fit_mean=True)"
    
    with open('gp_init_threads.py', 'w') as f:
        f.write(std_text.format(imp_mod, term1, term2, kernel, gp, prior_min, prior_max))
        
    