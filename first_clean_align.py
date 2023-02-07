import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as col
import cmasher as cmr
import argparse as ap
import scipy.optimize as op
from nudot_prof_functions import (do_rem_aln, aligndata, smart_align, removebaseline, calc_snr, check_null_prob,
                                  _find_off_pulse, rem_extra_noisy, rem_base_outs, get_rms_bline, _gauss_2,
                                  find_bright, findbrightestprofile, read_pdv, plot_joydivision, setup_log)

plt.rcParams["figure.figsize"] = (6, 10)


pars = ap.ArgumentParser()
pars.add_argument('psr_list', nargs='+')
pars.add_argument('-f', '--frq_list', nargs='+', default=1400, type=int)
pars.add_argument('-s', '--do_snrs', action='store_true')
pars.add_argument('-d', '--data_dir', default='/home/s86932rs/research/nudot_stuff/')
args = vars(pars.parse_args())

#data_dir = '/home/s86932rs/research/nudot_stuff/'
#psr = '1828-11'
psr_list = args['psr_list']
if type(psr_list) is str:
    psr_list = [psr_list]
    
frq_list = args['frq_list']
if type(frq_list) is int:
    frq_list = [frq_list]
    
data_dir = args['data_dir']
plots_dir = os.path.join(data_dir, 'plots')
do_snrs = args['do_snrs']

logger = setup_log(os.path.join(data_dir, 'nudot_proc.log'))

for psr in psr_list:
    for freq in frq_list:
        var_dict = {}
        for BE, be in zip(['AFB', 'DFB'], ['afb', 'dfb']):
            desc = "{}_{}_{}".format(psr, be, freq)
            DESC = "{}_{}_{}".format(psr, BE, freq)
            
            # find and read the pdv file
            pdv_file = os.path.join(data_dir, desc+'.pdv')
            if not os.path.exists(pdv_file):
                logger.warning("No data found for "+DESC)
                continue
                
            if os.path.exists(pdv_file.split('.pdv')[0]+'_new.pdv'):
                pdv_file = pdv_file.split('.pdv')[0]+'_new.pdv'

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

            plot_joydivision(raw_data, psr, savename=os.path.join(plots_dir, '{}_bk.png'.format(desc)), show=False)
            logger.info("Saved the joy division plot of raw data in "+os.path.join(plots_dir, '{}_bk.png'.format(desc)))

            # Add pulsars to these dictionaries of 'bad_mjds' after checking plots below
            bms_dict = {'2148+63_AFB_1400': 55381.3914371,
                        '1826-17_AFB_1400': [47379.9200344, 47626.1386156, 47631.2112292, 47634.0669695, 50759.7810235,
                                             50761.7664049, 50771.7181768, 50788.7252139, 50791.7040895, 50796.6790317,
                                             50800.6708258, 50837.594493,  50846.5306266, 50850.5216113, 50861.5069245,
                                             50910.3779423, 50947.2830917, 50951.9923754, 50966.1319925, 51022.0960915,
                                             51026.0849019, 51066.9231281, 51075.9278415, 51079.8892275, 51086.8757217,
                                             51092.8922606, 51096.866509,  51113.8001648, 51117.789548,  51127.7635914,
                                             51132.5119356, 51141.7505947, 51151.7311032, 51188.6240659, 51245.4028014,
                                             51308.1857909, 50828.6093899],
                        '2035+36_AFB_1400': [47153.9141866, 47156.9317142, 52823.3011286, 53290.0783178, 53551.3606083, 
                                             53867.5300664, 55311.8630421, 55324.5942195, 55359.4151889],
                        '0059+65_AFB_400':  [46784.9133219, 49767.0803575, 49432.8934948],
                        '0059+65_AFB_1400': [50725.3780112, 50729.2428495, 50769.0516674, 53493.5413218, 53675.9785209, 
                                             53699.1880196, 53734.2054306, 53764.0060906, 53788.0223723, 53803.9752749, 
                                             53866.5855581, 53902.7462029, 55304.8817443, 55320.7009743, 55334.750839,
                                             54429.2466849, 54121.0728087, 54635.8111413, 53807.9162362, 52763.6320561,
                                             50587.9635399, 52312.1220483, 53245.5518727, 51978.9119667, 50721.4258096,
                                             47153.3009316, 53174.4400886, 49631.1182132, 52798.5791639, 55203.0455693,
                                             54952.5366289, 49853.7472581, 55328.7362167, 52873.5607087, 54787.1767526,
                                             51805.689859,  55189.8829371, 51972.7459064, 50683.2954845, 54239.7728381,
                                             50387.8399387, 50685.5804531, 51885.7940866, 49512.8589906, 54351.4852092,
                                             54944.7675783, 52746.8116873, 50861.0453587, 50458.002315,  54543.9519631,
                                             54306.2462975, 49976.2625019, 55073.1626653],
                        '0059+65_DFB_1400': [55367.6631706, 57762.892351,  58416.1013078, 59779.2209669, 55656.848322,
                                             55897.1459373, 56157.5730639, 55239.9523247, 55203.0457396, 56988.1159176,
                                             55836.1827918],
                        '0105+65_DFB_1400': [59600.1514232, 56997.0202052, 55294.6519431, 57134.2771141, 58308.3660989,
                                             55559.9753237, 55865.0805497, 55897.1505985, 56997.0202052, 57417.8522612,
                                             57984.4555957, 58281.7087026, 58284.7013698, 58304.673917,  58341.3859481,
                                             58344.3767329, 58505.9637583, 58549.9953883, 58591.7144683, 58729.3310657,
                                             58798.5646731, 59147.0892765, 59385.5106392, 59488.351544,  59519.1683602,
                                             55187.1769081, 58343.3803444, 57489.5455148, 59125.4021092],
                        '0105+65_AFB_1400': [54438.0320864, 50771.3171731, 54773.0750428, 50334.512413,  50850.128813,
                                             54744.4513105, 51081.4488753, 53286.1331697, 54306.2509605, 52820.4252122,
                                             49343.6963994, 50679.3151838, 51274.6969456, 52478.3639154, 52608.1828337,
                                             53015.2394265, 53169.7290063, 53247.4508523, 53286.1331697, 53422.8253291,
                                             53549.7367486, 54429.2421402, 54438.0320864, 54472.9045933, 55038.1067002,
                                             54977.7695759, 51504.0846946, 53154.4579642, 50861.0499897, 51544.0375946,
                                             50789.0356424, 53414.6136466, 52798.5843742, 53591.9976536],
                        '0144+59_AFB_1400': 51820.2742387,
                        '0329+54_AFB_1400': [49090.7603093, 49466.7091834, 49243.4860953, 52465.2651231, 51956.872769,
                                             52494.1919405, 50641.5466752, 49505.4985513, 49439.5577478, 51402.1523014,
                                             51373.4948777, 52482.2152742, 51432.2807255, 51399.3388713, 52075.6435553,
                                             51372.3731208, 51375.6069209, 51376.7095308, 51386.3844638, 51388.299187,
                                             51391.5639895, 51392.5425549, 51393.4073588, 51394.3455276, 51395.6778106,
                                             51396.3317295, 51397.3805647, 51400.3503527, 51401.328434,  51405.4286168,
                                             51403.235319,  51404.2954221, 51435.4715615, 51622.8470687, 51307.6110845,
                                             50642.2916273, 49338.9836759, 51370.4925001, 52480.2211126, 53847.4870631,
                                             47800.3588073, 51369.4861072],
                        '0329+54_DFB_1400': [55802.3061775, 56314.5949707, 58744.3060784, 55853.2120838, 55890.1974306,
                                             55889.2333185, 55886.2158506, 55881.1668281, 55891.215034,  55924.9288624,
                                             55900.061694,  55982.9070965, 55947.9370838, 55466.3723666],
                        '0329+54_AFB_400':  [49435.5963684, 45953.6667351, 45965.0373907, 45982.0089368, 49810.5429632,
                                             45952.9490213, 45965.0340622, 45965.0364654, 45961.0492898, 45967.9232928,
                                             45990.8170404, 45991.8649349, 45995.9283481, 46048.017633,  46048.1027242,
                                             46062.0144607, 45150.2027432, 46417.2216718, 49338.6511645, 48506.4048371,
                                             47850.920023,  47852.5416656, 48344.68682,   49498.7319174],
                        '0329+54_AFB_600':  [50430.5919993, 47208.4347412, 49492.1522022, 55468.3857741, 55467.356001,
                                             50259.0596246, 54471.537135,  55468.3375307, 56470.0659171, 55506.6994529,
                                             50204.2122568, 49484.1867855, 49641.7399622, 48775.5811318, 50325.8344417,
                                             54245.1545333, 48665.6162682, 54893.6599822, 48665.8162508, 49436.3062204,
                                             55537.6736351, 49692.6611113, 49429.3259034, 49473.2077032, 49476.193719,
                                             55442.8743533, 55611.4169473, 49552.9974741, 50432.589855,  56556.8219223,
                                             49749.5096388, 55867.7189397, 54047.519714,  51428.8627889, 56504.9585333,
                                             56225.7308683, 54170.362885,  49434.3053514, 50211.1959587, 50431.5875895,
                                             54472.5369484, 53798.3773533, 56527.9089052, 56614.6724453, 49558.9664737,
                                             49553.9832503, 49477.1937305, 50333.8609263, 55925.5573029, 55868.7206169],
                        '1540-06_AFB_1400': [48629.1884107, 49091.9501664, 50355.4800225, 51086.5334786, 51468.3883089,
                                             52812.7590864, 52861.5981396, 53217.7602016, 53515.7791475, 54302.6798173,
                                             54815.2335074, 54897.0781401, 55073.5846445, 55333.8398418, 53426.0453499,
                                             50711.4929945, 52813.7851289, 52808.8224225, 54787.3694924, 54783.3686705,
                                             54775.4149237, 50390.4008051, 53972.6738923, 53952.5866584, 53904.7741823,
                                             53899.7530486, 53864.9454431, 53072.0356047, 52818.7342184, 53937.7425857,
                                             54799.3018991, 50353.4672182, 50086.268606,  52989.256162],
                        '1540-06_DFB_1400': [55271.0386086, 55325.9286343, 55369.7215846, 55532.5974662, 55533.3664932,
                                             55540.2641896, 55661.911316,  56358.9959424, 56732.0321809, 56851.6952469,
                                             57750.2349012, 58335.6074191, 58341.5925801, 58343.5868422, 58352.5618575,
                                             58358.5458972, 58372.4961166, 58522.0873325, 58524.0820964, 58355.5539521,
                                             58527.0738927, 58528.0677662, 58530.0622914, 58538.0437366, 58541.0355955,
                                             58599.8744361, 58600.8716086, 58601.8715318, 58602.8668938, 58608.8500791,
                                             58609.8472986, 58610.8451271, 58612.8386223, 58618.8227168, 58646.7464336,
                                             58648.741092,  58649.7377832, 58657.7162533, 58692.6207629, 58713.5628829,
                                             58721.5420498, 58722.5330041, 58744.4784002, 58751.457688,  58758.4405389,
                                             58780.3811642, 58899.1913396, 59005.7631382, 59006.7605395, 59019.7251003,
                                             59022.7163506, 59025.7087615, 59030.6948764, 59036.6789281, 59041.6645776,
                                             59042.6615971, 59044.6571983, 59056.6239945, 59131.4191046, 59163.3815552,
                                             59181.2824594, 59195.2447054, 59196.2420636, 59200.2309229, 59201.2281422,
                                             59254.083383,  59261.0640083, 59271.0371523, 59273.0306734, 59274.0301385,
                                             59347.8356379, 59366.7836908, 59383.7361275, 59515.3744726, 59516.3653563,
                                             59530.3309193, 59531.3312901, 59533.3265199, 59748.7366274, 59757.7124556,
                                             59758.7089755, 59765.6900726, 59772.6712299, 59774.6662147, 59775.6629657,
                                             59829.5152745, 59840.4807044, 59847.4659419, 59848.4642149, 59909.2979258,
                                             55526.4553332, 55550.2186894, 56144.5803235, 58281.7950567, 58284.7872283,
                                             58305.7302475, 58357.5479258, 58579.9268321, 58607.8516137, 58630.7899388,
                                             58666.6905994, 58753.4527314, 59049.6428265, 59087.5392411, 59292.9757437,
                                             59384.7340889, 59532.329427,  59760.7017727, 55529.3836909, 58334.6114349,
                                             58342.5897645, 58391.4448102, 58661.7049862, 59052.6342603, 59204.2201114,
                                             59373.7633862, 59517.3627465, 55394.7444929, 56866.6687792, 57321.4070177,
                                             58591.8963806, 59032.6895012, 59060.6136557, 59544.2955632, 58540.0384949,
                                             55531.3276573],
                        '2043+2740_AFB_1400': [51539.4654747, 53249.2282073, 53243.1473972, 52796.4861477, 52886.2337152,
                                               52826.3356262, 53495.494642,  52310.7135316, 51112.1238785, 51643.6335539,
                                               52704.2038436],
                        '2043+2740_DFB_1400': [55543.9133187, 59005.3876277, 58348.1866203, 58374.1164336, 58540.6602727,
                                               58645.366326,  59252.6542538, 59013.3653791, 59027.3274195, 58714.1854038,
                                               58347.1905069, 58643.3788171, 59004.3903466, 58781.0018067, 59291.5481363,
                                               58349.1838564, 59030.3188367, 58338.2140868, 59025.3329744, 59665.6714128,
                                               58659.3331818, 58999.4040148, 58377.1083701, 58355.1676547, 59313.5511421,
                                               58340.2145589, 58844.7594928, 58214.5484125, 58361.1523117, 58396.0563061,
                                               59023.3383211]}
            if not DESC in bms_dict:
                bms_dict[DESC] = None

            logger.info("Cleaning data without removing low S/N observations")
            try:
                var_dict[BE+'_aligned'], var_dict[BE+'_template'], var_dict[BE+'_mjds_null'], var_dict[BE+'_tobs'] = do_rem_aln(
                    raw_data, raw_mjds, raw_tobs, bad_mjds=bms_dict[DESC], thrsh=1.25, logg=logger)
            except RuntimeError:
                logger.error('Proceeding to next dataset')
                var_dict[BE+'_aligned'] = None
                var_dict[BE+'_template'] = None
                var_dict[BE+'_mjds_new'] = None
                var_dict[BE] = True
                continue

            plt.close('all')

            if do_snrs:
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
                        plt.title('{}, {}, {} MHz, summed profiles (after trimming)'.format(psr, BE, freq))
                        plt.ylabel('Normalised intensity')
                        plt.xlabel('Phase bins')
                        plt.xlim(int(len(prof1)/8), 3*int(len(prof1)/8))
                        plt.legend()
                        plt.savefig(os.path.join(plots_dir, desc+'_on_off_profs.png'), bbox_inches='tight')

                var_dict[BE+"_null_prob"] = check_null_prob(var_dict[BE+'_aligned'], peak_bin=100, ip=False, on_min=None, onf_range=None, off_min=None)
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
                var_dict[BE+'_aligned'], var_dict[BE+'_template'], var_dict[BE+'_mjds_new'], var_dict[BE+'_tobs'] = do_rem_aln(
                    raw_data, raw_mjds, raw_tobs, bad_mjds=bms_dict[DESC], thrsh=1.25, logg=logger, cut_snr=True)
            except RuntimeError:
                logger.error('Proceeding to next dataset')
                var_dict[BE+'_aligned'] = None
                var_dict[BE+'_template'] = None
                var_dict[BE+'_mjds_new'] = None
                var_dict[BE] = True
                continue
                
            var_dict[BE+'_aligned'] = np.nan_to_num(var_dict[BE+'_aligned'])

            plot_joydivision(var_dict[BE+'_aligned'], psr, savename=os.path.join(plots_dir, '{}_aligned_bk.png'.format(desc)), show=False)
            logger.info("Saved the joy division plot of cleaned data in "+os.path.join(plots_dir, '{}_aligned_bk.png'.format(desc)))
            plt.close('all')

        npz_file = os.path.join(data_dir, '{}_{}_arrs.npz'.format(psr, freq))
        if 'AFB' in var_dict.keys() and 'DFB' in var_dict.keys():
            logger.warning("No data to save to a .npz file, skipping")
            if os.path.exists(npz_file):
                os.remove(npz_file)
        else:
            np.savez(npz_file, **var_dict)
    
 