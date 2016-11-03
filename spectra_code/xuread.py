import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def listfiles():
    fnames = listdir('.')
    texfiles = []
    figfiles = []
    for fname in fnames:
        if fname.startswith('xu2016'):
            if 'fig' not in fname:
                texfiles.append(fname)
            else:
                figfiles.append(fname)
    return texfiles, figfiles

def getclambda(texfile):
    cutfile = texfile[:-9]
    clambda = cutfile[-4:]
    return clambda

def lambdaget(vel, clambda):
    lightvel = 299792.458
    wave = float(clambda)*(1.0 + vel/lightvel)
    return wave

def readtxt(fname):
    fdata = np.loadtxt(fname)
    x = np.asarray(fdata[:,0], dtype = np.longdouble)
    flux = np.asarray(fdata[:,1], dtype = np.longdouble)
    if abs(np.min(x)) > 400.0:
        wave = x
    else:
        clambda = getclambda(fname)
        wave = lambdaget(x, clambda)
        #plt.plot(wave, flux)
        #plt.show()
    return wave, flux

if __name__ == '__main__':
    texfiles, figfiles = listfiles()
    masterwave = []
    masterflux = []
    for texfile in texfiles:
        wave, flux = readtxt(texfile)
        masterwave = np.append(masterwave, wave)
        masterflux = np.append(masterflux, flux)
    for figfile in figfiles:
        wave, flux = readtxt(figfile)
        masterwave = np.append(masterwave, wave)
        masterflux = np.append(masterflux, flux)

    temparr = [masterwave, masterflux]

    masterflux = masterflux[np.argsort(temparr[0])]
    masterwave = masterwave[np.argsort(temparr[0])]
    plt.plot(masterwave, masterflux)
    plt.show()
    np.save('xupaper', [masterwave, masterflux])
