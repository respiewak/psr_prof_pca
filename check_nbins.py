import numpy as np
import argparse as ap


def read_file(filename):
    nbins = []
    data = []
    with open(filename, 'r') as f:
        lines = ''
        n = 0
        for line in f.readlines():
            if line[:5] == 'File:':
                if n != 0:
                    data.append(lines)
                    lines = ''

                nbins.append(int(line.split()[-3]))

            lines += line
            if n == 0:
                n = 1

        data.append(lines)

    nbins = np.array(nbins)
    data = np.array(data)

    return(nbins, data)


pars = ap.ArgumentParser()
pars.add_argument('filename', nargs='+')
args = vars(pars.parse_args())

for filen in args['filename']:
    fbins, fdata = read_file(filen)
    #print(np.unique(fbins))
    
    if len(np.unique(fbins)) == 1:
        continue
    
    count, vals = np.histogram(fbins, bins=40)
    imax = np.argmax(count)
    #print(count, vals, imax)
    lim = np.logical_and(fbins >= vals[imax], fbins < vals[imax+1])
    if vals[imax+1] == vals[-1]:
        lim = np.logical_and(fbins >= vals[imax], fbins <= vals[imax+1])

    print("Removing {} observations from {}, n_bins_good == {}"
          .format(len(fbins)-len(fbins[lim]), filen, fbins[lim][0]))
    
    new_filen = filen.split('.')[0]+'_new.pdv'
    with open(new_filen, 'w') as f:
        for str_line in fdata[lim]:
            f.write(str_line)


    
