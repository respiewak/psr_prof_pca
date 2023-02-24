# Using PCA and GPs to analyse pulsar profile variation
* * *

## Dependencies
* * *

These scripts require several python packages, those included in standard installations and otherwise. Conda is recommended for managing the environment, and an example package list for conda is given in `package-list.txt`.

* logging
* numpy
* matplotlib
* scikit-learn
* scipy
* multiprocessing
* george
* celerite
* emcee
* corner
* cmasher

In addition, scripts from David Kaplan's [pulse nulling code](github.com/dlakaplan/nulling-pulsars) are required for the optional nulling analysis.

Also, the `jupyter` package is required to utilise the notebooks. Executable versions of all notebooks are included, but the interactive nature of notebooks is better for optimising the later parts of the analysis. 


## Testing
* * *

In addition to the scripts and notebooks outlining the main parts of the analysis, a notebook for the creation of test ''profiles'' and subsequent analysis is included. This is a good place to start, to test your setup and to understand the analysis steps. No real data are required to run this notebook. 

To start, run `jupyter lab testing_prof_analysis.ipynb` (or `jupyter notebook testing_prof_analysis.ipynb`) and follow the steps therein. Alternatively, if you are familiar with the analysis and merely need to test the functions, run `python testing_prof_analysis.py` using the optional `-b` and `-d <path>` flags to use a dark background for plots and to set the output directory (default is `./test_plots`). 


## Running your analysis
* * *

### Prepare your data
* * *

These scripts require the input profiles to be described by ascii files as output by PSRCHIVE's `pdv` tool. I recommend dividing your data according to pulsar, backend, and frequency band and preparing data files based on those combinations. The relevant `pdv` command is:
>`$ pdv -A <psr_files> >> <psr_be_freq.pdv>`


Note that your archives should first be scrunched in time, frequency, and polarisation.

Also note that the expected name convention is strict in this version of the code. The `*.pdv` files should follow the `psr_be_freq.pdv` format, where `psr` is the pulsar name, `be` is the backend abbreviation (lower case), and `freq` is the frequency band (integer; can be arbitrary). For example, my data for B0059+65 with the AFB backend at 1400 MHz are contained in `B0059+65_afb_1400.pdv`. 


If there is any doubt about the consistency of the number of phase bins in individual `*.pdv` files, a simple script to determine the most common number of phase bins and remove inconsistent observations is included. There is no harm in running this script in any case. Run this command in the directory containing your data: 
> `$ python check_nbins.py <pdv_files>`


For every `psr_be_freq.pdv` file containing inconsistent bin counts, a new file will be produced with the name `psr_be_freq_new.pdv`; the input file is not altered. In the next part of the analysis, files with the `_new` tag will be prioritised.


### Cleaning and aligning profiles
* * *

The first steps of the analysis are contained in `first_clean_align.py`, which can be run from the terminal as follows:
> `$ python first_clean_align.py -f 400 1400 -b AFB B0059+65`


The arguments for this script are:
* Mandatory argument: Any number of pulsar names as they appear in the pdv file names (this is a greedy argument, so put it/them first before any flags or last after a non-argument flag or single-argument flag like `-l`)
* Frequency/ies: `-f|--frq_list`, Frequency band(s) as they appear in the pdv file names (integers); this argument is greedy
* Backend(s): `-b|--be_list`, Backend names/abbreviations as they appear in the pdv file names; this argument is greedy
* Nulling analysis: `-s|--do_snrs`, A switch to determine whether to analyse S/N values and determine probability of nulling; requires David Kaplan's nulling-pulsars code (WIP)
* Data directory: `-d|--data_dir`, Absolute or relative path to the directory containing profile data
* Bad data to remove:`-m|--bad_mjd_file`, Absolute or relative path to an ascii file listing MJDs of observations to exclude based on pulsar name, backend, and frequency band
* Log file: `-l|--log_name`, Path to the log file, or short name to use; the default value is 'prof_analysis' in the data directory


For that example, the code would look for two pdv files: `B0059+65_AFB_400.pdv` and `B0059+65_AFB_1400.pdv`.


The code is designed to loop over backends and frequency bands for each pulsar given, but any ''missing'' dataset will be skipped with only a warning printed to the log. Any datasets with fewer than 20 observations (before or after cleaning) will be skipped. 


As the input data are expected to be time-, frequency-, and polarisation-scrunched, it is recommended to perform cleaning and RFI removal prior to producing the pdv files. This script will remove significant outliers in off-pulse rms and pulse width (very generous threshold on width). 


This analysis does not assume perfect alignment of the input profiles, but the simple autocorrelation method used does not perform well with multi-peak profiles, so care should be taken in these cases.


While proceeding, this script will output plots into a directory inside your data directory called `plots` in the given data directory (the `plots` directory will be created if it does not exist). The plots produced include:
* A ''Joy Division'' style waterfall plot of raw profiles
* A ''Joy Division'' plot of profiles after all cleaning and alignment

If the S/N and nulling analysis is performed, plots of
* average profiles selected based on S/N (representing Null and On modes) and
* nulling probability over time
are also produced. 


The output plots should be examined after this step to ensure the cleaning and alignment are correct and sufficient.


### Principal Component Analysis of the profiles
* * *

#### Background
* * *

Most previous analyses of profile variation have run a Gaussian Process on the observations, for each bin in the on-pulse region (i.e., a dimensionality of Nbin by Nobs). The bins were treated as independent from each other. One possible way to improve upon this is to run a 2D Gaussian Process, but this is computationally expensive. 

Another method is to first use PCA to extract relevant eigenvectors and their eigenvalues for each observation, and then to run the Gaussian Process on the eigenvalues for the most significant eigenvectors (this is usually 2-5 eigenvectors). The predicted eigenvalues can be used to reconstruct profiles for any date in the time range, and the smoothed values can be compared with nu-dot or other parameters to find correlations. In addition, the eigenvectors directly represent the variations in the profile, making for easy interpretation. 


#### Running the analysis

Run
> `jupyter [lab|notebook] second_profile_pca.ipynb`

and follow the steps outlined. As with the previous step, most plots are also saved to the `<data_dir>/plots` directory. 

The first step in this part of the analysis is to set the region used by the PCA. It is not optimal to include excessive amounts of the off-pulse region, but the calculation of uncertainties on the eigenvalues requires some off-pulse region to be included. These regions are defined automatically (with the exception of any interpulse region) but can be modified if necessary. (Note that the region input to the PCA need not be contiguous, so the ''jump'' from the `off_max` value to the `ip_min` value is fine.) 


The PCA is run on the input profiles (with the `PCA` class from `scikit-learn`) using a mask to select the desired bin range. The output eigenvectors are contained in `out_pca.components_` and the eigenvalues in `out_comps_all`. (The PCA is hard-coded to retain the 30 most significant components/eigenvectors; this has been fine for my work but can be adjusted if necessary.) 


It is important to check the eigenvectors and eigenvalues for any outliers, whether due to misaligned profiles or RFI or simply noisy profiles. In particular, check the plot of the eigenvalues for the first 4 eigenvectors versus MJD. You can select outliers from this plot and check the input profile, and the MJD is printed to allow you to add it to a `bad_mjds` list if necessary. Note that you will need to re-run the first part of the analysis after identifying bad profiles for removal, and then to re-run this second part of the analysis. 


At this time, there is a redimentary ''binning'' of the eigenvalues at the end of this part, but this is not crucial to the analysis and will be removed. 


### Gaussian Process on the eigenvalues
* * *

The third part of the analysis uses Gaussian Processes, optimised using Markov-Chain Monte Carlo, to model the eigenvalues and allow for the creation of a smooth dataset without gaps. The GP is implemented using `celerite`, using a Matern-3/2 kernel (plus a Jitter/white noise term), and the parameter optimisation is performed by `emcee` with flat priors for all parameters. The length scale of the kernel is restricted to values determined from the dataset: the lower bound is the 97th percentile of the distribution of lags between observation epochs, and the upper bound is half the total timespan. 

Run
> `jupyter [lab|notebook] third_eigenvalue_gps.ipynb`

and follow the steps outlined. As with the previous steps, most plots are also saved to the `<data_dir>/plots` directory. 

The output of this notebook is a `.npz` file containing an array of MJDs with roughly 1 day cadence spanning the input observation range, and the associated predicted values (and variances) from the GPs for each selected eigenvector. An array of MJDs, e.g., those used for nudot GPs, can be given to the main function of this notebook, or one will be generated. 


### Testing for correlations between eigenvalues and nudot values
* * *

The final step of this analysis is to test for correlations between the smoothed eigenvalues and nudot values. This is most easily done when the MJDs from the nudot dataset are given directly to the eigenvalue GP function in the previous step. (At present, the analysis does not work if the nudot MJDs are not used for the eigenvalue GP predictions.) 

This is still work in progress, but there exists a notebook with some rough steps. Run 
> `jupyter [lab|notebook] fourth_nudot_correlations.ipynb`

and follow the steps. The correlation values will be printed (and saved to a text file in future), and a plot is made of the nudot values and eigenvalues for which the correlation coefficient is significant (`|rho| > 0.3`). 

