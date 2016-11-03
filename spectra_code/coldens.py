import numpy as np
from playspec import *

def optthin(width, f, clambda):
    return 1.130 * (10.0**12.0) * width/(f*clambda/(10.0**8.0))

def optthick(width, f, clambda, b = 10.0):
    lightc = 3.0*(10.0**10.0)
    b = b*(10.0**5.0)
    clambda = clambda/(10.0**8.0)
    return 46.29*(b/(f*clambda))*(np.exp((lightc*width/(2.0*b))**2.0))

def getcoldens(fname, widthfile, writename):
    writefile = open(writename, 'a')
    clambdas = 10.0*np.loadtxt(fname, usecols = [0])
    fs = 10.0**(np.loadtxt(fname, usecols = [1]))
    widths = np.loadtxt(fname, usecols = [-1])
    elows = np.loadtxt(fname, usecols = [5])
    optthins = np.empty_like(widths)
    optthicks = np.empty_like(optthins)
    print len(widths), len(clambdas)
    for i in range(len(clambdas)):
        optthins[i] =  optthin(widths[i]/clambdas[i], fs[i], clambdas[i])
        optthicks[i] =  optthick(widths[i]/clambdas[i], fs[i], clambdas[i])
        #print optthins[i]/(10.0**12.0), optthicks[i]/(10.0**12.0)
    for j in range(0, len(optthins)):
        line = (str(clambdas[j]) + ' ' + str(widths[j]) + ' ' + str(elows[j])
            + ' ' + str(optthins[j]) + ' ' + str(optthicks[j]) + '\n')
        writefile.write(line)
    return optthins, optthicks

if __name__ == '__main__':
    thins, thicks = getcoldens('long_XSFe_I_list.txt', 'XSCFe_I_testlines.txt',
        'Fe_I_master.txt')
    thins, thicks = getcoldens('long_XSFe_II_list.txt', 'XSFe_II_testlines.txt',
        'Fe_II_master.txt')
    thins, thicks = getcoldens('long_XSCa_II_list.txt', 'XSCa_II_testlines.txt',
        'Ca_II_master.txt')
