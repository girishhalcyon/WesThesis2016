import numpy as np
import matplotlib.pyplot as plt
from playspec import *
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import leastsq as optmin
#from lmfit import minimize as optmin
from lmfit import Parameters
import pandas as pd
from astropy.io import ascii as asc

def addparams(params):
    depth = params[3]
    start = params[0]
    end = params[1]
    offset = params[2]
    lmpars = Parameters()
    lmpars.add('depth', value = depth)
    lmpars.add('start', value = start, vary = True, min = -200.0, max = -0.1)
    lmpars.add('end', value = end, vary = True, min = -0.1, max = 500.0)
    lmpars.add('offset', value = offset)
    return lmpars

def getparams(lmpars):
    depth = lmpars['depth'].value
    start = lmpars['start'].value
    end = lmpars['end'].value
    offset = lmpars['offset'].value
    return [start, end, offset, depth]

def boxmodel(params, x):
    params = getparams(params)
    depth = params[3]
    start = params[0]
    end = params[1]
    offset = params[2]
    y = np.ones_like(x)
    for i in range(0,len(y)):
        if x[i] < start:
            y[i] = 1.0 + offset
        if start < x[i] < end:
            y[i] = 1.0 + offset - depth
        if x[i] > end:
            y[i] = 1.0 + offset
    return y

def boxmodel2(params, x):
    depth = params[3]
    start = params[0]
    end = params[1]
    offset = params[2]
    y = np.ones_like(x)
    for i in range(0,len(y)):
        if x[i] < start:
            y[i] = 1.0 + offset

        if start < x[i] < end:
            y[i] = 1.0 + offset - depth

        if x[i] > end:
            y[i] = 1.0 + offset
    return y

def trapmodel(params, x):
    amplitude = params[0]
    startshift = params[1]
    width = params[2]
    center = params[3]
    offset = params[4]
    endshift = params[5]
    x2 = center - (width / 2.0)
    x3 = center + (width / 2.0)
    x1 = x2 + startshift
    x4 = x3 + endshift
    slope_1 = amplitude/(x2 - x1)
    slope_2 = -amplitude/(x4 - x3)
    # Compute model values in pieces between the change points
    range_start = np.logical_and(x < x1, x > -1000.0)
    range_a = np.logical_and(x >= x1, x < x2)
    range_b = np.logical_and(x >= x2, x < x3)
    range_c = np.logical_and(x >= x3, x < x4)
    range_end = np.logical_and(x >= x4, x < 1000.0)
    val_start = offset
    val_a = offset + slope_1*(x - x1)
    val_b = amplitude + offset
    val_c = offset + amplitude + slope_2*(x - x3)
    val_end = offset
    return np.select([range_start, range_a, range_b, range_c, range_end], [val_start, val_a, val_b, val_c, val_end], default = offset)

def trapminfunc(params, x, y, yerr = []):
    model = trapmodel(params, x)
    if len(yerr) > 0:
        return (((model - y)**2.0)/yerr)
    else:
        return ((model-y)**2.0)


def boxminfunc(params, x, y, yerr = []):
    model = boxmodel2(params, x)
    if len(yerr) > 0:
        return np.sum(((model - y)**2.0)/yerr)
    else:
        return np.sum((model-y)**2.0)
def boxeq(params):
    depth = params[3]
    start = params[0]
    end = params[1]
    offset = params[2]
    return abs((end - start)*depth/offset)

def getcs(datax, datay, modelx, modely, start, end, offset):
    shifty = datay - offset
    modelyinterp = interp1d(modelx, modely)
    interpy = modelyinterp(datax)
    csy = datay - interpy
    mask = np.where((datax >= start) & (datax <= end))
    return datax[mask], csy[mask]

def funcwidth(x, y):
    area = abs(simps(x,y))
    width = area/1.0
    return width

def trapwidth(x,y):
    diffy = np.asarray(([y[i+1] - y[i] for i in range(0,len(y) - 1)]))
    diffx = np.asarray(([x[i+1] - x[i] for i in range(0,len(x) - 1)]))
    area = abs(np.sum(diffx*diffy))
    width = area/1.0
    return width

def sigtest(x,y,start,end, depth):
    mask = np.where((x < start) & (x > end))
    datasig = np.std(y[mask])
    significance = abs(depth)/datasig
    return significance

def fullxread():
    xshoot = idlread('../spectra_data/allxshooter3.var')
    xwave1 = np.append(xshoot.wave_uvballfeb, xshoot.wave_visallfeb)
    xwave2 = np.append(xshoot.wave_uvballmar, xshoot.wave_visallmar)
    xwave3 = np.append(xshoot.wave_uvballapr, xshoot.wave_visallapr)
    xflux1 = np.append(xshoot.flux_uvballfeb, xshoot.flux_visallfeb)
    xflux2 = np.append(xshoot.flux_uvballmar, xshoot.flux_visallmar)
    xflux3 = np.append(xshoot.flux_uvballapr, xshoot.flux_visallapr)
    xerr1 = np.append(xshoot.err_uvballfeb, xshoot.err_visallfeb)
    xerr2 = np.append(xshoot.err_uvballmar, xshoot.err_visallmar)
    xerr3 = np.append(xshoot.err_uvballapr, xshoot.err_visallapr)
    x1data = [xwave1, xflux1, xerr1]
    x2data = [xwave2, xflux2, xerr2]
    x3data = [xwave3, xflux3, xerr3]
    return x1data, x2data, x3data

def velboxplot(limits, clambda, kdata= [], xdata1 = [], xdata2 = [],
    xdata3 = [], paperdata = [], model = [], moff = +42.0,
    title = 'Comparison Plot', plotx = None, ploty = None, mode = 'SAVE',
    writename = 'ion_results.txt', ion = 'Fe II'):

    plow = limits[0]
    phigh = limits[1]



    fluxlist = []
    vellist = []



    kecksig = 0.0
    xsig1 = 0.0
    xsig2 = 0.0
    xsig3 = 0.0
    xusig = 0.0
    keckres = [-0.5, -50, 100, 0.0, 0.0, 100.0]
    x1res = [-0.5, -50, 100, 0.0, 0.0, 100.0]
    x2res = [-0.5, -50, 100, 0.0, 0.0, 100.0]
    x3res = [-0.5, -50, 100, 0.0, 0.0, 100.0]
    xures = [-0.5, -50, 100, 0.0, 0.0, 100.0]
    xshootsig = 0.0
    xcsflux1= []
    xcsflux2 = []
    xcsflux3 = []
    xucsflux = []
    kcsflux = []
    xcompflux1 = []
    xcompflux2 = []
    xcompflux3 = []
    kcompflux = []
    xucompflux = []

    #print 'Point 1'
    if len(model) > 0:
        mfmask = np.where((np.isfinite(model[0])) & (np.isfinite(model[1])))
        mwave = (model[0][mfmask])
        mflux = model[1][mfmask]
        mrmask = np.where((mwave >= plow - 50) & (mwave <= phigh + 50))
        mrwave = mwave[mrmask]
        mrflux = mflux[mrmask]
        if len(mrwave) > 0:
            mnflux = mrflux/np.median(mrflux)
            mnflux = mnflux/np.median(mnflux)
            mrmask = np.where((0.0 < mrflux/np.median(mrflux)) & (mrflux/np.median(mrflux) < 2.5))
            mrwave = mrwave[mrmask]
            mrflux = mrflux[mrmask]
            mnflux = mrflux
            mnflux = mnflux/np.median(mnflux)
            mrvel =wave2vel(vac2air(vel2wave(wave2vel(mrwave, clambda) + moff, clambda)), clambda) + 100.0
            #mrvel = wave2vel(mrwave, clambda)
            #ax.plot(mrvel, mnflux, '--r', label = 'Stellar Model')
            modelfunc = interp1d(mrvel, mnflux)

    if len(mrwave) > 0:
        xshoot = xdata1
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])) & (np.isfinite(xshoot[2])))
        xwave = xshoot[0][xfmask]
        xflux = xshoot[1][xfmask]
        xerr = xshoot[2][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]
        xrerr = xerr[xrmask]

        if len(xrwave) > 0:
            xrflux = xrflux/np.median(xrflux)
            xrflux = xrflux/np.median(xrflux)
            xrmask = np.where((0.0 < xrflux/np.median(xrflux)) & (xrflux/np.median(xrflux) < 1.5))
            xrwave = xrwave[xrmask]
            xrflux = xrflux[xrmask]
            xrerr = xrerr[xrmask]
            xnflux = xrflux
            xnerr = xrerr/np.median(xrflux)
            xnflux = xnflux/np.median(xnflux)

            xrvel = wave2vel(xrwave, clambda)
            csflux = xnflux - modelfunc(xrvel)
            xcsflux1 = csflux
            xdatavel1 = xrvel
            xcompflux1 = xnflux
            xerr1 = xnerr

            x1res= optmin(trapminfunc, x1res, args = (xrvel, csflux, xnerr))
            params = x1res[0]


            amplitude = params[0]
            startshift = params[1]
            width = params[2]
            center = params[3]
            offset = params[4]
            endshift = params[5]
            x2 = center - (width / 2.0)
            x3 = center + (width / 2.0)
            x1 = x2 + startshift
            x4 = x3 + endshift
            slope_1 = amplitude/(x2 - x1)
            slope_2 = -amplitude/(x4 - x3)

            x1start = x1
            x1end = x4
            x1amp = amplitude

            trapsigma = np.median(xnerr)

            x1sigma = abs(amplitude)/trapsigma

        xshoot = xdata2
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])) & (np.isfinite(xshoot[2])))
        xwave = xshoot[0][xfmask]
        xflux = xshoot[1][xfmask]
        xerr = xshoot[2][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]
        xrerr = xerr[xrmask]

        if len(xrwave) > 0:
            xrflux = xrflux/np.median(xrflux)
            xrflux = xrflux/np.median(xrflux)
            xrmask = np.where((0.0 < xrflux/np.median(xrflux)) & (xrflux/np.median(xrflux) < 1.5))
            xrwave = xrwave[xrmask]
            xrflux = xrflux[xrmask]
            xrerr = xrerr[xrmask]
            xnflux = xrflux
            xnerr = xrerr/np.median(xrflux)
            xnflux = xnflux/np.median(xnflux)

            xrvel = wave2vel(xrwave, clambda)
            csflux = xnflux - modelfunc(xrvel)
            xcsflux2 = csflux
            xdatavel2 = xrvel
            xcompflux2 = xnflux
            xerr2 = xnerr

            x2res= optmin(trapminfunc, x2res, args = (xrvel, csflux, xnerr))
            params = x2res[0]

            amplitude = params[0]
            startshift = params[1]
            width = params[2]
            center = params[3]
            offset = params[4]
            endshift = params[5]
            x2 = center - (width / 2.0)
            x3 = center + (width / 2.0)
            x1 = x2 + startshift
            x4 = x3 + endshift
            slope_1 = amplitude/(x2 - x1)
            slope_2 = -amplitude/(x4 - x3)

            x2start = x1
            x2end = x4
            x2amp = amplitude

            trapsigma = np.median(xnerr)

            x2sigma = abs(amplitude)/trapsigma

        xshoot = xdata3
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])) & (np.isfinite(xshoot[2])))
        xwave = xshoot[0][xfmask]
        xflux = xshoot[1][xfmask]
        xerr = xshoot[2][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]
        xrerr = xerr[xrmask]

        if len(xrwave) > 0:
            xrflux = xrflux/np.median(xrflux)
            xrflux = xrflux/np.median(xrflux)
            xrmask = np.where((0.0 < xrflux/np.median(xrflux)) & (xrflux/np.median(xrflux) < 1.5))
            xrwave = xrwave[xrmask]
            xrflux = xrflux[xrmask]
            xrerr = xrerr[xrmask]
            xnflux = xrflux
            xnerr = xrerr/np.median(xrflux)
            xnflux = xnflux/np.median(xnflux)

            xrvel = wave2vel(xrwave, clambda)
            csflux = xnflux - modelfunc(xrvel)
            xcsflux3 = csflux
            xdatavel3 = xrvel
            xcompflux3 = xnflux
            xerr3 = xnerr

            x3res= optmin(trapminfunc, x3res, args = (xrvel, csflux, xnerr))
            params = x3res[0]

            amplitude = params[0]
            startshift = params[1]
            width = params[2]
            center = params[3]
            offset = params[4]
            endshift = params[5]
            x2 = center - (width / 2.0)
            x3 = center + (width / 2.0)
            x1 = x2 + startshift
            x4 = x3 + endshift
            slope_1 = amplitude/(x2 - x1)
            slope_2 = -amplitude/(x4 - x3)

            x3start = x1
            x3end = x4
            x3amp = amplitude

            trapsigma = np.median(xnerr)

            x3sigma = abs(amplitude)/trapsigma


        allxsigma = [x1sigma, x2sigma, x3sigma]
        allxstart = [x1start, x2start, x3start]
        allxend = [x1end, x2end, x3end]
        allxamp = [x1amp, x2amp, x3amp]

        conditions = (np.max(allxstart) <= moff) & (np.min(allxend) >= moff) & (np.max(allxamp) <= 0.0)
        if conditions == True:
            xshootsig = np.min(allxsigma)
        else:
            xshootsig = 5.0


    if xshootsig >=3.0:
        if  '4923' in title:
            keckfile = idlread('../spectra_data/wd1145+017.sav')
            greenwave = keckfile.sgreen.wave[0]
            greenflux = keckfile.sgreen.flux[0]
            greenerr = keckfile.sgreen.err[0]
            kecktemp = [greenwave[0], greenflux[0], greenerr[0]]
            kdata = binarray(kecktemp[0], kecktemp[1], kecktemp[2], len(kecktemp[0])/5.0)
        if len(kdata) > 0:
            kfmask = np.where((np.isfinite(kdata[0])) & (np.isfinite(kdata[1])))
            kwave = kdata[0][kfmask]
            kflux = kdata[1][kfmask]
            kerr = kdata[2][kfmask]
            ktemp = [kwave, kflux, kerr]
            kwave = kwave[np.argsort(ktemp[0])]
            kflux = kflux[np.argsort(ktemp[0])]
            kerr = kerr[np.argsort(ktemp[0])]
            krmask = np.where((kwave >= plow) & (kwave <= phigh))
            krflux = kflux[krmask]
            krerr = kerr[krmask]
            krwave = kwave[krmask]
            if len(krwave) > 0:
                krflux = krflux/np.median(krflux)
                krflux = krflux/np.median(krflux)
                krmask = np.where((0.0 < krflux/np.median(krflux)) & (krflux/np.median(krflux) < 1.5))
                krwave = krwave[krmask]
                krerr = krerr[krmask]
                krflux = krflux[krmask]
                knflux = krflux
                knerr = krerr/np.median(knflux)
                knflux = knflux/np.median(knflux)
                krvel = wave2vel(krwave, clambda)
                csflux = knflux - modelfunc(krvel)
                kcsflux = csflux
                kcompflux = knflux
                keckerr = krerr
                kdatavel = krvel


                keckres = optmin(trapminfunc, [-0.5, -20, 20, 0.0, 0.0, 100.0], args = (krvel, csflux, knerr))
                params = keckres[0]

                amplitude = params[0]
                startshift = params[1]
                width = params[2]
                center = params[3]
                offset = params[4]
                endshift = params[5]
                x2 = center - (width / 2.0)
                x3 = center + (width / 2.0)
                x1 = x2 + startshift
                x4 = x3 + endshift
                slope_1 = amplitude/(x2 - x1)
                slope_2 = -amplitude/(x4 - x3)
                boxsigma = np.std(csflux[np.where((krvel <= x1) | (krvel >= x4))])
                kecksigma = abs(amplitude)/boxsigma

    if xshootsig >= 3.0:
        paperhires = paperdata
        if len(paperhires) > 0:
            pfmask = np.where((np.isfinite(paperhires[0])) & (np.isfinite(paperhires[1]))
                & (paperhires > 0.0))
            pwave = paperhires[0][pfmask]
            pflux = paperhires[1][pfmask]
            prmask = np.where((pwave <= phigh) & (pwave >= plow))
            prwave = pwave[prmask]
            prflux = pflux[prmask]

            if ((len(prwave) > 0) & (title != 'CaII 8498')):
                #pnflux = prflux/iterfit(prwave, prflux, degree = 3, niter= 25)
                if np.isfinite(np.median(prflux)) & (np.median(prflux) != 0.0):
                    prflux = prflux/np.median(prflux)
                else:
                    prflux = prflux[np.where((np.isfinite(prflux)) & (prflux > 0.0))]
                prmask = np.where((0.0 < prflux/np.median(prflux)) & (prflux/np.median(prflux) < 1.5))
                prwave = prwave[prmask]
                prflux = prflux[prmask]
                pnflux = prflux
                if np.isfinite(np.median(pnflux)) & (np.median(pnflux) != 0.0):
                    pass
                else:
                    prwave = prwave[np.where((np.isfinite(pnflux)) & (pnflux > 0.0))]
                    pnflux = pnflux[np.where((np.isfinite(pnflux)) & (pnflux > 0.0))]
                pnflux = pnflux/np.median(pnflux)
                prvel = wave2vel(prwave, clambda) + 0.0
                csflux = pnflux - modelfunc(prvel)
                xucsflux = csflux
                xucompflux = pnflux
                xudatavel = prvel

                xures = optmin(trapminfunc, [-0.5, -20, 20, 0.0, 0.0, 100.0], args = (prvel, csflux))
                params = xures[0]
                amplitude = params[0]
                slope = params[1]
                width = params[2]
                center = params[3]
                offset = params[4]
                endshift = params[5]
                x2 = center - (width / 2.0)
                x3 = center + (width / 2.0)
                x1 = x2 + startshift
                x4 = x3 + endshift
                slope_1 = amplitude/(x2 - x1)
                slope_2 = -amplitude/(x4 - x3)
                boxsigma = np.std(csflux[np.where((prvel <= x1) | (prvel >= x4))])
                xusigma = abs(amplitude)/boxsigma


    allxtrapflux = np.append(xcsflux1, xcsflux2)
    allxtrapflux = np.append(allxtrapflux, xcsflux3)
    allxtrapflux = np.append(allxtrapflux, kcsflux)
    allxtrapflux = np.append(allxtrapflux, xucsflux)


    allxcompflux = np.append(xcompflux1, xcompflux2)
    allxcompflux = np.append(allxcompflux, xcompflux3)
    allxcompflux = np.append(allxcompflux, kcompflux)
    allxcompflux = np.append(allxcompflux, xucompflux)


    if xshootsig >= 3.0:
                #print clambda, resx[-1]/(1.0*boxsigma)
            mainfig = plt.figure()
            mainfig.suptitle(title)

            trapplot = plt.subplot(1,2,1)
            compplot = plt.subplot(1,2,2)
            trapmax = np.max(allxtrapflux)
            trapmin = np.min(allxtrapflux)
            trapplotrange = trapmax - trapmin
            traplow = trapmin - 0.1*trapplotrange
            traphigh = trapmax + 0.1*trapplotrange
            trapplot.set_xlim((-300.0, +300.0))
            trapplot.set_ylim((traplow, traphigh))
            compmax = np.max(allxcompflux)
            compmin = np.min(allxcompflux)
            compplotrange = compmax - compmin
            complow = compmin - 0.1*compplotrange
            comphigh = compmax + 0.1*compplotrange
            compplot.set_xlim((-300.0, 300.0))
            compplot.set_ylim((complow, comphigh))
            compplot.plot(mrvel, mnflux, '-r')
            compplot.plot([moff]*100, np.linspace(complow, comphigh, 100), '-b')


            if len(kcsflux) > 0:
                trapplot.errorbar(kdatavel, kcsflux, yerr = keckerr, alpha = 0.1, fmt = '.k', label = 'November Keck')
                compplot.errorbar(kdatavel, kcompflux, yerr = keckerr, alpha = 0.1, fmt = '.k')

            if len(xucsflux) > 0:
                trapplot.plot(xudatavel, xucsflux, ':c', label = 'Xu Keck')
                compplot.plot(xudatavel, xucompflux,':c')


    if xshootsig >= 3.0:
        bestx = np.argmax(allxsigma)
        allxvel = [xdatavel1, xdatavel2, xdatavel3]
        allxflux = [xcsflux1, xcsflux2, xcsflux3]
        allcomps = [xcompflux1, xcompflux2, xcompflux3]
        allxerr = [xerr1, xerr2, xerr3]
        xlabels = ['Feb VLT', 'Mar VLT', 'Apr VLT']
        bestflux = allxflux[bestx]
        allxres = [x1res[0], x2res[0], x3res[0]]
        mask = np.where((allxvel[bestx] >= allxstart[bestx]) & (allxvel[bestx] <= allxend[bestx]))
        intwave = vel2wave(allxvel[bestx][mask], clambda)
        width = trapwidth(intwave, bestflux[mask])
        writetext = (ion + ' ' + str(clambda) + ' ' + str(allxsigma[bestx])
            + ' ' + str(width) + ' ' + str(allxstart[bestx]) + ' '
            + str(allxend[bestx]) + ' ' + str(abs(allxres[bestx][0])))
        with open(writename, 'a') as writefile:
            #print writetext
            writefile.write(writetext + '\n')
        trapplot.errorbar(allxvel[bestx], allxflux[bestx], yerr = allxerr[bestx], fmt = '--g', label = xlabels[bestx])
        trapplot.plot(allxvel[bestx], trapmodel(allxres[bestx], allxvel[bestx]), '-m', label = 'Trapezoid Model')
        compplot.plot(allxvel[bestx], allcomps[bestx], '--g')
        trapplot.legend(loc = 'best')
        if mode == 'SAVE':
            fname = title.replace(' ', '_') + '.pdf'
            plt.savefig('trap_plots/'+ fname)
        else:
            plt.show()
        plt.clf()

if __name__ == '__main__':
    #clambdas = [4923.921]
    #titles = ['Fe II 4923-921']
    #clambdas = [4351.735, moff49.465, moff33.162, moff83.829, 4923.921, 5316.609,
    #    3933.653, 3968.476]
    #titles = ['Fe I 4351-735', 'Fe I moff49-465', 'Fe II moff33-162',
    #    'Fe II moff83-829', 'Fe II 4923-921', 'Fe II 5316-609', 'Ca II 3933-653',
    #    'Ca II 3968-476']
    keck = keckread()
    xshoot = xshootread()
    model = conmodelread()
    esi = readesi()
    hires = readhires()
    pmodel = readmodel()
    x1data, x2data, x3data = fullxread()


    fname = '../spectra_data/seth_line_list.txt'
    df = asc.read(fname).to_pandas()
    clambdas = df.RWave
    temptitles = df.Ion
    comments = df.Comments
    mask = np.where((clambdas > 3000.0) & (clambdas < 9100.0))[0]
    clambdas = np.array(clambdas[mask])
    temptitles = np.array(temptitles[mask])

    titles = [temptitles[i] + ' ' + str(clambdas[i]).replace('.', '-') for i in range(0,len(clambdas))]
    limitarr = [(vel2wave(-400.0, clambda), vel2wave(+400.0, clambda)) for clambda in clambdas]
    for i in range(220,len(clambdas)):
        try:
            velboxplot(limitarr[i], clambda = clambdas[i], kdata = keck, model = model,
                title = titles[i], mode = 'SAVE', xdata1 = x1data, xdata2 = x2data,
                xdata3 = x3data, paperdata = hires,
                writename = 'ion_results.txt', ion = temptitles[i])
            print i, 'out of ', len(clambdas)
        except:
            print temptitles[i], ' ', clambdas[i], ' failed'
    print (fname + ' done')
