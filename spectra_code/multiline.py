import numpy as np
import matplotlib.pyplot as plt
from playspec import *
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize as optmin
# from lmfit import minimize as optmin
from lmfit import Parameters


def addparams(params):
    depth = params[3]
    start = params[0]
    end = params[1]
    offset = params[2]
    lmpars = Parameters()
    lmpars.add('depth', value=depth)
    lmpars.add('start', value=start, vary=True, min=-200.0, max=-0.1)
    lmpars.add('end', value=end, vary=True, min=-0.1, max=500.0)
    lmpars.add('offset', value=offset)
    return lmpars


def getparams(lmpars):
    depth = lmpars['depth'].value
    start = lmpars['start'].value
    end = lmpars['end'].value
    offset = lmpars['offset'].value
    return [start, end, offset, depth]


def boxmodel(params, x):
    params = getparams(params)
    start = params[0]
    end = params[1]
    offset = params[2]
    depth = params[3]
    y = np.ones_like(x)
    for i in range(0, len(y)):
        if x[i] < start:
            y[i] = 1.0 + offset
        if start < x[i] < end:
            y[i] = 1.0 + offset - depth
        if x[i] > end:
            y[i] = 1.0 + offset
    return y


def boxmodel2(params, x):
    start = params[0]
    end = params[1]
    offset = params[2]
    depth = params[3]
    y = np.ones_like(x)
    for i in range(0, len(y)):
        if x[i] < start:
            y[i] = offset

        if start < x[i] < end:
            y[i] = offset - depth

        if x[i] > end:
            y[i] = offset
    return y


def boxminfunc(params, x, y, yerr=[]):
    model = boxmodel2(params, x)
    start = params[0]
    end = params[1]
    offset = params[2]
    depth = params[3]
    if depth <= 0.0:
        return np.inf
    if start >= end:
        return np.inf
    if start >= 0.0:
        return -np.inf
    if end <= 0.0:
        return -np.inf

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
    area = abs(simps(x, y))
    width = area/1.0
    return width


def trapwidth(x, y):
    diffy = np.asarray(([y[i+1] - y[i] for i in range(0, len(y) - 1)]))
    diffx = np.asarray(([x[i+1] - x[i] for i in range(0, len(x) - 1)]))
    area = abs(np.sum(diffx*diffy))
    width = area/1.0
    return width


def sigtest(x, y, start, end, depth):
    mask = np.where((x < start) & (x > end))
    datasig = np.std(y[mask])
    significance = abs(depth)/datasig
    return significance


def velboxplot(limits, clambda, data=[], xdata=[], paperdata=[],
               model=[], moff=55.0, dataoff=0.0, title=None, plotx=None,
               ploty=None, mode='SAVE', datatype='Keck',
               writename='XS_Fe_II_testlines.txt'):

    plow = limits[0]
    # print plow, wave2vel(plow, clambda)
    phigh = limits[1]
    # print phigh, wave2vel(phigh, clambda)
    ax = plt.subplot(111)
    fluxlist = []
    vellist = []
    title = title + ' Circumstellar Absorption'

    kdata = data
    kecksig = 0.0
    xshootsig = 0.0
    xusig = 0.0
    keckres = [0.5, -100.0, 200.0, 0.0, 0.5]
    xshootres = [0.5, -100.0, 200.0, 0.0, 0.5]
    xures = [0.5, -100.0, 200.0, 0.0, 0.5]

    # print 'Point 1'
    if len(model) > 0:
        mfmask = np.where((np.isfinite(model[0])) & (np.isfinite(model[1])))
        mwave = (model[0][mfmask])
        mflux = model[1][mfmask]
        mrmask = np.where((mwave >= plow - 100) & (mwave <= phigh + 100))
        mrwave = mwave[mrmask]
        mrflux = mflux[mrmask]
        if len(mrwave) > 0:
            # mnflux = mrflux/iterfit(mrwave, mrflux, degree = 3, niter = 25)
            mrmask = np.where((0.0 < mrflux/np.median(mrflux)) &
                              (mrflux/np.median(mrflux) < 1.5))
            mrwave = mrwave[mrmask]
            mrflux = mrflux[mrmask]
            mnflux = mrflux
            mnflux = mnflux/np.median(mnflux) + 0.1
            mrvel = wave2vel(vac2air(vel2wave(wave2vel(mrwave, clambda) + moff,
                             clambda)), clambda)
            # ax.plot(mrvel, mnflux, '--r', label = 'Stellar Model')
            modelfunc = interp1d(mrvel, mnflux)
    data = xdata
    datatype = 'X-Shoot'
    if datatype == 'X-Shoot':
        xshoot = data
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])))
        xwave = xshoot[0][xfmask]*10.0
        xflux = xshoot[1][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]

        if len(xrwave) > 0:
            # xnflux = xrflux/iterfit(xrwave, xrflux, degree = 3, niter= 25)
            xrmask = np.where((0.0 < xrflux/np.median(xrflux)) &
                              (xrflux/np.median(xrflux) < 1.5))
            xrwave = xrwave[xrmask]
            xrflux = xrflux[xrmask]
            xnflux = xrflux
            xnflux = xnflux/np.median(xnflux)
            xrvel = wave2vel(xrwave, clambda) + 15.0
            csflux = xnflux - modelfunc(xrvel)
            res = optmin(boxminfunc, [-100.0, 200.0, 0.0, 0.5],
                         args=(xrvel, csflux),
                         method='Nelder-Mead', tol=0.001)
            # resx = getparams(res.params)
            resx = res.x
            boxsigma = np.std(csflux[np.where((xrvel <= resx[0]) |
                              (xrvel >= resx[1]))])
            xbox = boxsigma
            if resx[-1] >= 3.0*boxsigma:
                # print clambda, resx[-1]/(1.0*boxsigma)
                ax.plot(xrvel, csflux, '-g', label='X-Shooter')
                datavel = xrvel
                fluxlist = np.append(fluxlist, csflux)
                xshootsig = resx[-1]/boxsigma
                xshootresx = resx
                bestflux = csflux
            else:
                xshootsig = 0.0
                xshootres = [-100.0, 200.0, 0.0, 0.5]
    datatype = 'Keck'
    # print 'Point 2'
    data = kdata
    if datatype == 'Keck':
        if '4923' in title:
            keckfile = idlread('wd1145+017.sav')
            greenwave = keckfile.sgreen.wave[0]
            greenflux = keckfile.sgreen.flux[0]
            greenerr = keckfile.sgreen.err[0]
            kecktemp = [greenwave[0], greenflux[0], greenerr[0]]
            data = binarray(kecktemp[0], kecktemp[1], kecktemp[2],
                            len(kecktemp[0])/10.0)
        if len(data) > 0:
            kfmask = np.where((np.isfinite(data[0])) & (np.isfinite(data[1])))
            kwave = data[0][kfmask]
            kflux = data[1][kfmask]
            kerr = data[2][kfmask]
            ktemp = [kwave, kflux, kerr]
            kwave = kwave[np.argsort(ktemp[0])]
            kflux = kflux[np.argsort(ktemp[0])]
            kerr = kerr[np.argsort(ktemp[0])]
            krmask = np.where((kwave >= plow) & (kwave <= phigh))
            krflux = kflux[krmask]
            krerr = kerr[krmask]
            krwave = kwave[krmask]
            if len(krwave) > 0:
                # knflux = krflux/iterfit(krwave, krflux, degree=3, niter=25)
                krmask = np.where((0.0 < krflux/np.median(krflux)) &
                                  (krflux/np.median(krflux) < 1.5))
                krwave = krwave[krmask]
                krerr = krerr[krmask]
                krflux = krflux[krmask]
                knflux = krflux
                knflux = knflux/np.median(knflux)
                krvel = wave2vel(krwave, clambda) + dataoff
                csflux = knflux - modelfunc(krvel)
                res = optmin(boxminfunc, [-100.0, 200.0, 0.0, 0.5],
                             args=(krvel, csflux),
                             method='Nelder-Mead', tol=0.001)
                # resx = getparams(res.params)
                resx = res.x
                boxsigma = np.std(csflux[np.where((krvel <= resx[0]) |
                                  (krvel >= resx[1]))])
                if xshootsig >= 3.0:
                    fluxlist = np.append(fluxlist, csflux)
                    vellist = np.append(vellist, krvel)
                    # print clambda, resx[-1]/(1.0*boxsigma)
                    ax.plot(krvel, csflux, '-k', label='November Keck')
                    datavel = krvel
                    kecksig = resx[-1]/boxsigma
                    keckres = res
                else:
                    kecksig = 0.0
                    keckres = [-100.0, 200.0, 0.0, 0.5]

    datatype = 'Xu-HIRES'
    # print 'Point 3'
    if datatype == 'Xu-HIRES':
        paperhires = paperdata
        if len(paperhires) > 0:
            pfmask = np.where((np.isfinite(paperhires[0])) &
                              (np.isfinite(paperhires[1])) &
                              (paperhires > 0.0))
            pwave = paperhires[0][pfmask]
            pflux = paperhires[1][pfmask]
            prmask = np.where((pwave <= phigh) & (pwave >= plow))
            prwave = pwave[prmask]
            prflux = pflux[prmask]

            if ((len(prwave) > 0) & (title != 'CaII 8498')):
                # pnflux = prflux/iterfit(prwave, prflux, degree=3, niter=25)
                if np.isfinite(np.median(prflux)) & (np.median(prflux) != 0.0):
                    prflux = prflux/np.median(prflux)
                else:
                    prflux = prflux[np.where((np.isfinite(prflux)) &
                                    (prflux > 0.0))]
                prmask = np.where((0.0 < prflux/np.median(prflux)) &
                                  (prflux/np.median(prflux) < 1.5))
                prwave = prwave[prmask]
                prflux = prflux[prmask]
                pnflux = prflux
                if np.isfinite(np.median(pnflux)) & (np.median(pnflux) != 0.0):
                    pass
                else:
                    prwave = prwave[np.where((np.isfinite(pnflux)) &
                                    (pnflux > 0.0))]
                    pnflux = pnflux[np.where((np.isfinite(pnflux)) &
                                    (pnflux > 0.0))]
                pnflux = pnflux/np.median(pnflux)
                prvel = wave2vel(prwave, clambda) + 0.0
                csflux = pnflux - modelfunc(prvel)
                res = optmin(boxminfunc, [-100.0, 200.0, 0.0, 0.5],
                             args=(prvel, csflux),
                             method='Nelder-Mead', tol=0.001)
                # resx = getparams(res.params)
                resx = res.x
                boxsigma = np.std(csflux[np.where((prvel <= resx[0]) |
                                  (prvel >= resx[1]))])
                if xshootsig >= 3.0:
                    # print clambda, resx[-1]/(1.0*boxsigma)
                    ax.plot(prvel, csflux, '--',
                            color='orange', label='Xu HIRES')
                    datavel = prvel
                    fluxlist = np.append(fluxlist, csflux)
                    xusig = resx[-1]/boxsigma
                    xures = res
                else:
                    xusig = 0.0
                    xures = [0.5, -100.0, 200.0, 0.0, 0.5]
    # print 'Point 4'
    if len(fluxlist) > 0:
        xstart = xshootresx[0]
        xend = xshootresx[1]
        mask = np.where((xrvel >= xstart) & (xrvel <= xend))
        intwave = xrwave + vel2wave(15.0, clambda)
        width = trapwidth(intwave[mask], bestflux[mask])
        writetext = (str(clambda) + ' ' + str(xshootresx[-1]/xbox)
                     + ' ' + str(width) + ' ' + str(xstart) + ' ' + str(xend)
                     + ' ' + str(xshootresx[-1]))
        with open(writename, 'a') as writefile:
            print writetext
            writefile.write(writetext + '\n')
        allsigs = [kecksig, xshootsig, xusig]
        allres = [keckres, xshootres, xures]
        plt.title(title)
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('Normalized Flux')
        plt.plot(datavel, boxmodel2(resx, datavel), '--m', label='Box Model')
        if plotx is None:
            xlow = limits[0]
            xhigh = limits[1]
            xdiff = xhigh - xlow
            plotlow = xlow + 0.25*xdiff
            plothigh = xhigh - 0.25*xdiff
            plt.xlim((-600.0, +600.0))
            if title is None:
                title = 'Comparison Plot from ' + str(plotlow) + 'to '
                title = title + str(plothigh)
        else:
            plt.xlim(plotx)
            if title is None:
                title = 'Comparison Plot from ' + str(plotrange[0]) + 'to '
                title = title + str(plotrange[1])
        # print 'Point 5'
        if ploty is None:
            # masknew = np.where((-600.0 < vellist) & (vellist < 600.0))
            # ydiff = np.max(fluxlist[masknew]) - np.min(fluxlist[masknew])
            # maxplot = np.max(fluxlist[masknew]) + 0.1*ydiff
            # minplot = np.min(fluxlist[masknew])- 0.1*ydiff
            plt.ylim(-0.5, 0.2)
            plt.plot([42.0]*100, np.linspace(-0.5, 0.2, 100), '-b')

        else:
            plt.ylim(ploty)
            plt.title(title.replace('-', '.'))
        plt.legend()
        if mode == 'SAVE':
            fname = title.replace(' ', '_') + datatype + '.pdf'
            # plt.savefig('masterplots/' +fname)
            # plt.savefig(fname)
            plt.show()
        else:
            plt.show()
        plt.clf()
        return res, xrvel, bestflux
    return '1', '2', '3'
