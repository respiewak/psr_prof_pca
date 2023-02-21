#!/usr/bin/env python
# coding: utf-8

##  Author: Renee Spiewak


import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col
import cmasher as cmr
import argparse as ap
import scipy.optimize as op
from all_prof_functions import (do_rem_aln, aligndata, smart_align, removebaseline, calc_snr,
                                _find_off_pulse, rem_extra_noisy, rem_base_outs,
                                get_rms_bline, _gauss_2, find_bright, findbrightestprofile,
                                read_pdv, plot_joydivision, setup_log, read_bad_mjd_file)

try:
    from all_prof_functions import check_null_prob
except OSError:
    print("Cannot load nulling module; skipping that part of the analysis")

plt.rcParams["figure.figsize"] = (6, 10)


pars = ap.ArgumentParser(description='The first step in analysing pulsar profile variation: '\
                         'cleaning and aligning the profiles provided in pdv files.')
pars.add_argument('psr_list', nargs='+', help="Pulsar names as they appear in the pdv file names")
pars.add_argument('-f', '--frq_list', nargs='+', default=1400, type=int,
                  help="Frequency band(s) as they appear in the pdv file names (integers)")
pars.add_argument('-b', '--be_list', nargs='+', default=['afb', 'dfb'],
                  help="Backend names/abbreviations as they appear in the pdv file names")
pars.add_argument('-s', '--do_snrs', action='store_true',
                  help="Whether to analyse S/N values and determine probability of nulling; "\
                  "requires David Kaplan's nulling-pulsars code (WIP)")
pars.add_argument('-d', '--data_dir', default='../profiles/',
                  help="Absolute or relative path to the directory containing profile data")
pars.add_argument('-m', '--bad_mjd_file', default='bad_mjds_jbo.txt',
                  help="An ascii file listing MJDs of observations to exclude based on "\
                  "pulsar name, backend, and frequency band")
pars.add_argument('-l', '--log_name', default='prof_analysis',
                  help="Name for the log file")
args = vars(pars.parse_args())

data_dir = args['data_dir']
log_name = args['log_name']
if len(log_name.split('.')) == 1:
    log_name += '.log'
    
logger = setup_log(os.path.join(data_dir, log_name))

psr_list = args['psr_list']
if type(psr_list) is str:
    psr_list = [psr_list]
    
frq_list = args['frq_list']
if type(frq_list) is int:
    frq_list = [frq_list]

be_list = args['be_list']
BE_list = [A.upper() for A in be_list]
be_list = [A.lower() for A in BE_list]
    
plots_dir = os.path.join(data_dir, 'plots')
if not os.path.exists(plots_dir):
    logger.warning("Creating directory for the output plots: "+plots_dir)
    os.chdir(data_dir)
    os.mkdir('plots')
    
do_snrs = args['do_snrs']

bad_mjd_file = args['bad_mjd_file']
bms_dict = read_bad_mjd_file(bad_mjd_file)

for psr in psr_list:
    for freq in frq_list:
        var_dict = {}
        for BE, be in zip(BE_list, be_list):
            desc = "{}_{}_{}".format(psr, be, freq)
            DESC = "{}_{}_{}".format(psr, BE, freq)
            
            # find and read the pdv file
            pdv_file = os.path.join(data_dir, desc+'.pdv')
            if not os.path.exists(pdv_file):
                logger.warning("No data found for "+DESC)
                continue
                
            if os.path.exists(pdv_file.split('.pdv')[0]+'_new.pdv'):
                pdv_file = pdv_file.split('.pdv')[0]+'_new.pdv'

            logger.info("Reading data from "+pdv_file)
            raw_data, raw_mjds, raw_tobs = read_pdv(pdv_file, logg=logger)
            if raw_data is None:
                logger.info('Moving to next dataset')
                continue
                
            logger.info("The shape of the {} data is {}".format(DESC, raw_data.shape))
            if raw_data.shape[1] < 20:
                logger.warning("Dataset for {} is too small, skipping".format(DESC))
                var_dict[BE+'_aligned'] = None
                var_dict[BE+'_template'] = None
                var_dict[BE+'_mjds_new'] = None
                var_dict[BE] = True
                continue

            plot_joydivision(raw_data, psr, show=False,
                             savename=os.path.join(plots_dir, '{}_bk.png'.format(desc)))
            logger.info("Saved the joy division plot of raw data in "\
                        +os.path.join(plots_dir, '{}_bk.png'.format(desc)))

            if not DESC in bms_dict:
                bms_dict[DESC] = None

            logger.info("Cleaning data without removing low S/N observations")
            try:
                var_dict[BE+'_aligned'], var_dict[BE+'_template'], var_dict[BE+'_mjds_null'],\
                    var_dict[BE+'_tobs'] = do_rem_aln(raw_data, raw_mjds, raw_tobs,
                                                      bad_mjds=bms_dict[DESC], thrsh=1.25, logg=logger)
            except RuntimeError:
                logger.error('Proceeding to next dataset')
                var_dict[BE+'_aligned'] = None
                var_dict[BE+'_template'] = None
                var_dict[BE+'_mjds_new'] = None
                var_dict[BE] = True
                continue

            plt.close('all')

            if do_snrs and 'check_null_prob' in globals():
                proceed = True
                logger.info("Doing S/N and nulling stuff")
                snrs = calc_snr(var_dict[BE+'_aligned'])
    
                dist, bins, _ = plt.hist(snrs, bins=50)
                mids = 0.5*(bins[:-1] + bins[1:])
                #cut = 100
                #bounds = ([0, bins[0], 0, 0, bins[0], 0],
                #          [np.max(dist)*2, bins[-1], bins[-1]-bins[0], np.max(dist)*2, bins[-1], 2*(bins[-1]-bins[0])])
                #if bounds[1][0] < 0 or bins[-1] < bins[0]:
                #    logger.error('S/N distribution not appropriate, skipping')
                #    proceed = False
                
                if proceed:
                    try:
                        popt, pcov = op.curve_fit(_gauss_2, mids, dist)#, bounds=bounds)
                        std_err = np.sqrt(popt[2]**2 + popt[5]**2)
                        if popt[1] > popt[4] - std_err: 
                            logger.warning('Too much overlap between gaussians, skipping')
                            proceed = False
                        else:
                            proceed = True
                        
                    except RuntimeError:
                        logger.warning('Failed to fit gaussians to S/N distribution, skipping')
                        proceed = False
            
                if proceed:
                    cut = mids[np.argmin(dist[np.logical_and(mids > popt[1], mids < popt[4])])]
                    lims = snrs < cut

                    with plt.style.context('default'):
                        plt.clf()
                        prof1 = np.sum(var_dict[BE+'_aligned'][:,lims], axis=1)
                        prof2 = np.sum(var_dict[BE+'_aligned'][:,np.logical_not(lims)], axis=1)
                        plt.plot(prof1/np.max(prof1), label="S/N < {}".format(cut))
                        plt.plot(prof2/np.max(prof2), label="S/N > {}".format(cut))
                        plt.title('{}, {}, {} MHz, summed profiles (after trimming)'
                                  .format(psr, BE, freq))
                        plt.ylabel('Normalised intensity')
                        plt.xlabel('Phase bins')
                        plt.xlim(int(len(prof1)/8), 3*int(len(prof1)/8))
                        plt.legend()
                        plt.savefig(os.path.join(plots_dir, desc+'_on_off_profs.png'),
                                    bbox_inches='tight')

                var_dict[BE+"_null_prob"] = check_null_prob(var_dict[BE+'_aligned'], peak_bin=100,
                                                            ip=False, on_min=None, onf_range=None,
                                                            off_min=None)
                with plt.style.context('default'):
                    plt.clf()
                    plt.plot(var_dict[BE+"_mjds_null"], var_dict[BE+"_null_prob"])
                    plt.ylabel('Probability of Null')
                    plt.xlabel('MJD (days)')
                    plt.ylim(-0.01, 1.01)
                    plt.title("{}, {}, {} MHz, Nulling probabilities".format(BE, psr, freq))
                    plt.savefig(os.path.join(plots_dir, desc+'_null_probs.png'), bbox_inches='tight')
                
                plt.close('all')
            else:
                var_dict[BE+"_nulling"] = False
                var_dict[BE+"_null_prob"] = None
                var_dict[BE+"_mjds_null"] = None
                
            logger.info("Cleaning data *and* removing low S/N observations")
            try:
                var_dict[BE+'_aligned'], var_dict[BE+'_template'], var_dict[BE+'_mjds_new'],\
                    var_dict[BE+'_tobs'] = do_rem_aln(raw_data, raw_mjds, raw_tobs,
                                                      bad_mjds=bms_dict[DESC], thrsh=1.25,
                                                      logg=logger, cut_snr=True)
            except RuntimeError:
                logger.error('Proceeding to next dataset')
                var_dict[BE+'_aligned'] = None
                var_dict[BE+'_template'] = None
                var_dict[BE+'_mjds_new'] = None
                var_dict[BE] = True
                continue
                
            var_dict[BE+'_aligned'] = np.nan_to_num(var_dict[BE+'_aligned'])

            plot_joydivision(var_dict[BE+'_aligned'], psr, show=False,
                             savename=os.path.join(plots_dir, '{}_aligned_bk.png'.format(desc)))
            logger.info("Saved the joy division plot of cleaned data in "\
                        +os.path.join(plots_dir, '{}_aligned_bk.png'.format(desc)))
            plt.close('all')

        npz_file = os.path.join(data_dir, '{}_{}_arrs.npz'.format(psr, freq))
        if 'AFB' in var_dict.keys() and 'DFB' in var_dict.keys():
            logger.warning("No data to save to a .npz file, skipping")
            if os.path.exists(npz_file):
                os.remove(npz_file)
        else:
            out_file = os.path.join(data_dir, '{}_gps_fin.npz'.format(psr))
            old_dict = {}
            if os.path.exists(out_file):
                with np.load(out_file) as f:
                    for key in f.keys():
                        if key not in var_dict.keys():
                            old_dict[key] = f[key]
                        else:
                            print("Replacing an older value for "+key)
                
            np.savez(out_file, **var_dict, **old_dict)
    
 
