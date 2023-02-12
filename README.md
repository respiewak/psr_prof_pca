# Using PCA and GPs to analyse pulsar profile variation
* * *

## Dependencies
* * *

These scripts require several python packages both included in standard installations and otherwise. Conda is recommended for managing the environment, and an example package list for conda is given in `package-list.txt`.

* logging
* numpy
* matplotlib
* scikit-learn
* scipy
* multiprocessing
* george
* emcee
* corner
* cmasher

In addition, scripts from David Kaplan's [pulse nulling code](github.com/dlakaplan/nulling-pulsars) are required for the optional nulling analysis.

I also recommend installing the `jupyter` package in order to utilise the notebooks. 


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

Run
> `$ jupyter [lab|notebook] second_profile_pca.ipynb`

and follow the steps outlined. As with the previous step, most plots are also saved to the `<data_dir>/plots` directory. 


### Gaussian Process on the eigenvalues
* * *

Run
> `$ jupyter [lab|notebook] third_eigenvalue_gps.ipynb`

and follow the steps outlined. As with the previous steps, most plots are also saved to the `<data_dir>/plots` directory. 


