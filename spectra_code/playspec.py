from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.optimize import minimize as optmin
from scipy.integrate import simps
from tempfile import TemporaryFile


def wave2vel(wave, clambda):
    dellambda = wave - clambda
    lightvel = 299792.458
    vel = dellambda*lightvel/clambda
    return vel

def vel2wave(vel, clambda):
    lightvel = 299792.458
    wave = clambda*(1.0 + vel/lightvel)
    return wave

def vac2air(vacwave):
    a = np.longdouble(6.4328e-5)
    b = np.longdouble(2.94981e-2)
    c = np.longdouble(2.5540e-4)
    d = np.longdouble(146.0)
    e = np.longdouble(41.0)
    sigterm = np.longdouble(1.0e4)/vacwave
    term1 = b/(d - sigterm**2.0)
    term2 = c/(e - sigterm**2.0)
    denomterm = 1.0 + a + term1 + term2
    airwave = vacwave/denomterm
    return airwave
def datread(fname):
    datfile = open(fname)
    lines = datfile.readlines()
    wave = np.empty(len(lines))
    flux = np.empty(len(lines))
    for i in range(0,len(lines)):
        wave[i] = np.longdouble(lines[i].strip().split()[0])
        flux[i] = np.longdouble(lines[i].strip().split()[1])
    datfile.close()
    return wave, flux

def gaussconvolve(wave, flux, fwhm = 9.0):
    gauss_kernel = Gaussian1DKernel(stddev = fwhm/2.3548)
    gauss_wave = convolve(wave, gauss_kernel)
    gauss_flux = convolve(flux, gauss_kernel)
    return gauss_wave[35:-35], gauss_flux[35:-35]

def idlread(fname):
    savfile = readsav(fname, verbose = True)
    return savfile

def resample(oldarray, factor):
    xp = np.arange(0, len(oldarray), factor)
    lin = interp1d(np.arange(len(oldarray)), oldarray)
    newarray = lin(xp)
    return newarray

def testlength(arrays):
    domshape = np.shape(arrays[0])
    for i in range(0,np.shape(arrays)[0]):
        if np.shape(arrays[1]) != domshape:
            print 'Array lengths do not match'
            raise ValueError

def linfit(x, y):
    return np.polyfit(x,y, deg = 1)

def singlefit(x,y, degree, nsigma = 2.0):
    fit = np.polyfit(x,y,deg = degree)
    resids = y - np.poly1d(fit)(x)
    outliers = np.where(abs(resids - np.median(resids)) >= nsigma*np.std(resids))
    return outliers, np.poly1d(fit)

def iterfit(x,y,degree, niter):
    originalx = x
    originaly = y
    i = 0
    while i <= niter:
        outliers, fit = singlefit(x,y,degree = degree)
        if len(outliers) == 0:
            break
        else:
            x = np.delete(x,outliers)
            y = np.delete(y, outliers)
        i +=1
    return fit(originalx)

def meritfunc(y,refy, x):
    contrefy = iterfit(x,refy, degree = 3, niter = 25)
    conty = iterfit(x,y, degree = 3, niter = 25)
    return np.sum(abs(contrefy-conty))

def shiftmeasure(shift, y , x, refxfunc):
    refeval = refxfunc(x)
    return meritfunc(y+shift, refeval, x)

def polynorm(x, y, refxfunc, npoints = 800, degree = 3, niter = 25):

    splitx = np.array_split(x, len(x)/npoints)
    splity = np.array_split(y, len(y)/npoints)
    normy = np.array([])
    normx = np.array([])
    for i in range(0,np.shape(splitx)[0]):
        fit = iterfit(splitx[i], splity[i], degree = degree, niter = niter)
        window = (splity[i]/(fit))
        currentwindow = window - np.median(window - 1.0)
        shiftinit = -0.1
        finalshift = optmin(shiftmeasure, shiftinit, args = (currentwindow, splitx[i], refxfunc))

        normy = np.append(normy, currentwindow + finalshift.x[0])
        normx = np.append(normx, splitx[i])
    return normy, normx

def polynorm2(x,y, npoints = 800, degree = 3, niter = 25):
    splitx = np.array_split(x, len(x)/npoints)
    splity = np.array_split(y, len(y)/npoints)
    normy = np.array([])
    normx = np.array([])
    for i in range(0,np.shape(splitx)[0]):
        fit = iterfit(splitx[i], splity[i], degree = degree, niter = niter)
        window = (splity[i]/(fit))
        currentwindow = window - np.median(window - 1.0)

        normy = np.append(normy, currentwindow)
        normx = np.append(normx, splitx[i])
    return normy, normx

def savenp(outfile, x, y, yerr = []):
    if len(yerr) > 0:
        structarray = np.array([x, y, yerr])
    else:
        structarray = np.array([x,y])
    np.save(outfile, structarray)
    print 'Saved file'

def binarray(x, y, yerr, nbins):
    listarr = np.array([x,y,yerr])
    splitarr  = np.array_split(listarr, nbins, axis = 1)
    newxtot = np.array([])
    newytot = np.array([])
    newyerrtot = np.array([])
    residerrtot = np.array([])
    #print np.shape(np.array(splitarr))
    for i in range(0,len(splitarr)):
        newytot = np.append(newytot, np.mean(splitarr[i][1]))
        newxtot = np.append(newxtot, np.mean(splitarr[i][0]*splitarr[i][1])/newytot[i])
        linparams = linfit(splitarr[i][0], splitarr[i][1])
        residtest = splitarr[i][1] - linparams[0]*splitarr[i][0] - linparams[1]
        newyerrtot = np.append(newyerrtot, np.std(residtest))
        #newyerrtot = np.append(newyerrtot, np.std(splitarr[i][1]))
        #newyerrtot = np.append(newyerrtot, np.sqrt(np.sum(splitarr[i][2]*splitarr[i][2]))/len(splitarr[i][2]))

    return np.array([newxtot, newytot, newyerrtot])


def linecont(x,y, method = 'Mouse'):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(x[0],y[0], '.')
    ax.set_ylim((-0.1, 1.35))
    ax2 = fig.add_subplot(122)
    ax2.set_ylim((-0.1, 1.35))
    ax2.plot(x[1], y[1], '.')
    x = x[0]
    y = y[0]
    global tlinepos
    tlinepos = []
    contoffset = []
    def onclick(event):

        tlinepos.append(event.xdata)
        return tlinepos
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))
    plt.show()
    plt.clf()
    if method != 'Mouse':
        #Plot then close and:
        #Ask for manual input
        pas
    else:
        #Interactive plot: define range of continuum before & after line
        #Find median of before and after
        #Continuum level = mean of the two medians
        #2nd Interactive plot to define region of line
        linepos = np.sort(tlinepos)
        beforex = [linepos[0], linepos[1]]
        afterx = [linepos[4], linepos[5]]
        lineposition = [linepos[2], linepos[3]]
        beforey = y[np.where((x <= beforex[1]) & (x >= beforex[0]))]
        aftery = y[np.where((x <= afterx[1]) & (x >= afterx[0]))]

        contoffset = np.mean([np.median(beforey), np.median(aftery)])
    return contoffset, lineposition

def equilwidth(x, y, linepos, contoffset, model = True):
    if model:
        datax = x[0]
        datay = y[0]
        modelx = x[1]
        modely = y[1]

        modelyinterp = interp1d(modelx, modely)
        modely2 = modelyinterp(datax)
        modelx2 = datax

        datalinepos = linepos[0]
        datacont = contoffset[0]
        modellinepos = linepos[1]
        modelcont = contoffset[1]

        newy = (datay - datacont)
        newmody = (modely - modelcont)
        shiftmody = (datay - (modely2 - (modelcont - datacont)))

        mlinemask = np.where((modelx >= modellinepos[0]) & (modelx <= modellinepos[1]))
        dlinemask = np.where((datax >= datalinepos[0]) & (datax <= datalinepos[1]))

        modarea = simps(newmody[mlinemask], modelx[mlinemask])
        modwidth = -modarea/modelcont

        datarea = simps(newy[dlinemask], datax[dlinemask])
        datwidth = -datarea/datacont

        subarea = simps(shiftmody[dlinemask], modelx2[dlinemask])
        subwidth = -subarea/datacont

        area = [datarea, modarea, subarea]
        eqwidth = [datwidth, modwidth, subwidth]
        print eqwidth, datwidth - modwidth

    else:
        datax = x
        datay = y

        datalinepos = linepos
        datacont = contoffset

        newy = -1.0*(datay - datacont)
        newmody = -1.0*(modely - modelcont)

        dlinemask = np.where((datax >= datalinpos[0]) & (datax <= datalinepos[1]))

        datarea = simps(newy[dlinemask], datax[dlinemask])
        datwidth = datarea/datacont

        area = datarea
        eqwidth = datwidth

    return area, eqwidth

def stitchwindows(multiwinarray):
    nwin = np.shape(multiwinarray)[0]
    newarray = []
    for i in range(0,nwin):
        newarray = np.append(newarray, multiwinarray[i])
    return newarray

def fillerror(x, y, yerr = [], xerr = [], lim = (0, -1), color = 'k', linestyle = '-', palpha = 1.0, ealpha = 0.5):
    ax = plt.figure()
    fig1 = plt.subplot()
    fig1.plot(x[lim[0]:lim[1]], y[lim[0]:lim[1]], color, linestyle, alpha = palpha)
    if len(yerr) == 0:
        if len(xerr) == 0:
            return ax
        else:
            fig1.fill_between(y, x - xerr, x + xerr, facecolor = 'b', ealpha = 0.5)
            return ax
    elif len(xerr) == 0:
        fig1.fillbetween(x,y-xerr, x+xerr, facecolor = 'r', ealpha = 0.5)
        return ax
    else:
        fig1.fill_between(y, x - xerr, x + xerr, facecolor = 'b', ealpha = 0.5)
        fig1.fillbetween(x,y-xerr, x+xerr, facecolor = 'r', ealpha = 0.5)
        return ax

def readbin():
    savfile = idlread('wd1145+017.sav')
    bluewave = savfile.sblue.wave[0]
    redwave = savfile.sred.wave[0]
    greenwave = savfile.sgreen.wave[0]
    blueerr = savfile.sblue.err[0]
    rederr = savfile.sred.err[0]
    greenerr = savfile.sgreen.err[0]
    blueflux = savfile.sblue.flux[0]
    redflux = savfile.sred.flux[0]
    greenflux = savfile.sgreen.flux[0]
    blue = [stitchwindows(bluewave), stitchwindows(blueflux), stitchwindows(blueerr)]
    green = [stitchwindows(greenwave), stitchwindows(greenflux), stitchwindows(greenerr)]
    red = [stitchwindows(redwave), stitchwindows(redflux), stitchwindows(rederr)]

    blue5 = binarray(blue[0], blue[1], blue[2], len(blue[0])/10)
    red5 = binarray(red[0], red[1], red[2], len(red[0])/10)
    green5 = binarray(green[0], green[1], green[2], len(green[0])/10)

    masterwave = np.append(blue5[0], green5[0])
    masterwave = np.append(masterwave, red5[0])
    masterflux = np.append(blue5[1], green5[1])
    masterflux = np.append(masterflux, red5[1])
    mastererr = np.append(blue5[2], green5[2])
    mastererr = np.append(mastererr, red5[2])
    finitemask = np.where(np.isfinite(masterwave))
    masterwave = masterwave[finitemask]
    masterflux = masterflux[finitemask]
    mastererr = mastererr[finitemask]

    np.save('databin10', [masterwave, masterflux, mastererr])

def modelread():
    wave, flux = datread('model2_dk.dat')
    finitemask2 = np.where((np.isfinite(wave) & (np.isfinite(flux))))
    wave = wave[finitemask2]
    flux = flux[finitemask2]

    rawinterp = interp1d(wave, flux)
    shift = [wave[i+1] - wave[i] for i in range(0, len(wave) - 1)]
    wave = np.arange(np.min(wave), np.max(wave), np.median(shift))
    flux = rawinterp(wave)
    restrictwave, restrictflux = gaussconvolve(wave, flux)

    print 'Convolved model'

    np.save('modelconvolve9', [restrictwave, restrictflux])

def keckread(fname = 'databin5.npy'):
    return np.load(fname)

def conmodelread(fname = 'model2convolve9.npy'):
    return np.load(fname)

def xshootread(fname = 'xshooter_all.var'):
    xshooter = idlread(fname)
    twave = np.append(xshooter.wave_uvball, xshooter.wave_visall)
    xwave = np.append(twave, xshooter.wave_nirall)
    tflux = np.append(xshooter.flux_uvball, xshooter.flux_visall)
    xflux = np.append(tflux, xshooter.flux_nirall)
    return [xwave, xflux]

def readhires(fname = 'HIRES.WD1145+017.txt', skip = 502):
    tempfile = np.loadtxt(fname, skiprows = skip)
    wave = tempfile[:,0]
    flux = tempfile[:,1]
    return [wave,flux]

def readesi(fname = 'ESI.WD1145+017.txt', skip = 394):
    return readhires(fname = fname, skip = skip)

def readmodel(fname = 'dbf1.txt', skip = 11):
    return readhires(fname = fname, skip = skip)

def waveplot(limits, keck = [], xshoot = [], paperhires = [],
    paperesi = [], papermodel = [], model = [],
    koff = 0.0, xoff = 0.0, moff = 0.0, poff = 0.0,
    title = None, plotx = None, ploty = None, mode = 'SAVE'):

    plow = limits[0]
    phigh = limits[1]
    ax = plt.subplot(111)
    fluxlist = []

    if  '4924' in title:
        keckfile = idlread('wd1145+017.sav')
        greenwave = keckfile.sgreen.wave[0]
        greenflux = keckfile.sgreen.flux[0]
        greenerr = keckfile.sgreen.err[0]
        kecktemp = [greenwave[0], greenflux[0], greenerr[0]]
        keck = binarray(kecktemp[0], kecktemp[1], kecktemp[2], len(kecktemp[0])/10.0)
    if len(keck) > 0:
        kfmask = np.where((np.isfinite(keck[0])) & (np.isfinite(keck[1])))
        kwave = keck[0][kfmask] + koff
        kflux = keck[1][kfmask]
        kerr = keck[2][kfmask]
        ktemp = [kwave, kflux, kerr]
        kwave = kwave[np.argsort(ktemp[0])]
        kflux = kflux[np.argsort(ktemp[0])]
        kerr = kerr[np.argsort(ktemp[0])]
        krmask = np.where((kwave >= plow) & (kwave <= phigh))
        krwave = kwave[krmask]
        krflux = kflux[krmask]
        krerr = kerr[krmask]
        if len(krwave) > 0:
            #knflux = krflux/iterfit(krwave, krflux, degree = 3, niter = 25)
            knflux = krflux
            knflux = knflux/np.median(knflux)
            ax.plot(krwave, knflux, '-b', alpha = 0.3)
            ax.fill_between(krwave, knflux - krerr, knflux + krerr, facecolor = 'b', alpha = 0.05)
            fluxlist = np.append(fluxlist, knflux)
            if title == None:
                kecksave = str(plow) + '_' + str(phigh) + '_keck'

            else:
                kecksave = title.replace(' ', '_') + '_keck'
            #np.save(kecksave, [krwave, knflux, krerr])
    if len(xshoot) > 0:
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])))
        xwave = xshoot[0][xfmask]*10.0 + xoff
        xflux = xshoot[1][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]
        if len(xrwave) > 0:
            #xnflux = xrflux/iterfit(xrwave, xrflux, degree = 3, niter= 25)
            xnflux = xrflux
            xnflux = xnflux/np.median(xnflux)
            ax.plot(xrwave, xnflux, '-g')
            fluxlist = np.append(fluxlist, xnflux)
            if title == None:
                xsave = str(plow) + '_' + str(phigh) + '_xshoot'

            else:
                xsave = title.replace(' ', '_') + '_xshoot'
            #np.save(xsave, [xrwave, xnflux])
    if len(model) > 0:
        mfmask = np.where((np.isfinite(model[0])) & (np.isfinite(model[1])))
        mwave = vac2air(model[0][mfmask]) + moff
        mflux = model[1][mfmask]
        mrmask = np.where((mwave >= plow) & (mwave <= phigh))
        mrwave = mwave[mrmask]
        mrflux = mflux[mrmask]
        if len(mrwave) > 0:
            #mnflux = mrflux/iterfit(mrwave, mrflux, degree = 3, niter = 25)
            mnflux = mrflux
            mnflux = mnflux/np.median(mnflux)
            ax.plot(mrwave, mnflux, '--r')
            fluxlist = np.append(fluxlist, mnflux)
            if title == None:
                msave = str(plow) + '_' + str(phigh) + '_model'

            else:
                msave = title.replace(' ', '_') + '_model'
            #np.save(msave, [mrwave, mnflux])

    if len(paperhires) > 0:
        pfmask = np.where((np.isfinite(paperhires[0])) & (np.isfinite(paperhires[1]))
            & (paperhires > 0.0))
        pwave = paperhires[0][pfmask] + poff
        pflux = paperhires[1][pfmask]
        prmask = np.where((pwave >= plow) & (pwave <= phigh))
        prwave = pwave[prmask]
        prflux = pflux[prmask]
        if ((len(prwave) > 0) & (title != 'CaII 8498')):
            #pnflux = prflux/iterfit(prwave, prflux, degree = 3, niter= 25)
            pnflux = prflux
            pnflux = pnflux/np.median(pnflux)
            ax.plot(prwave, pnflux, ':', color = 'orange')
            fluxlist = np.append(fluxlist, pnflux)
            if title == None:
                psave = str(plow) + '_' + str(phigh) + '_paper_hires'

            else:
                psave = title.replace(' ', '_') + '_paper_hires'
            #np.save(psave, [prwave, pnflux])

    if len(paperesi) > 0:
        pfmask = np.where((np.isfinite(paperesi[0])) & (np.isfinite(paperesi[1])))
        pwave = paperesi[0][pfmask] + poff
        pflux = paperesi[1][pfmask]
        prmask = np.where((pwave >= plow) & (pwave <= phigh))
        prwave = pwave[prmask]
        prflux = pflux[prmask]
        if ((len(prwave) > 0) & (title != 'CaII 8498') & ('Na D' not in title)):
            #pnflux = prflux/iterfit(prwave, prflux, degree = 3, niter= 25)
            pnflux = prflux
            pnflux = pnflux/np.median(pnflux)
            ax.plot(prwave, pnflux, '-.', color = 'orange')
            fluxlist = np.append(fluxlist, pnflux)
            if title == None:
                psave = str(plow) + '_' + str(phigh) + '_paper_esi'

            else:
                psave = title.replace(' ', '_') + '_paper_esi'
            #np.save(psave, [prwave, pnflux])

    if len(papermodel) > 0:
        pfmask = np.where((np.isfinite(papermodel[0])) & (np.isfinite(papermodel[1])))
        pwave = papermodel[0][pfmask] + poff
        pflux = papermodel[1][pfmask]
        prmask = np.where((pwave >= plow) & (pwave <= phigh))
        prwave = pwave[prmask]
        prflux = pflux[prmask]
        if ((len(prwave) > 0) & (title != 'CaII 8498')):
            #pnflux = prflux/iterfit(prwave, prflux, degree = 3, niter= 25)
            pnflux = prflux
            pnflux = pnflux/np.median(pnflux)
            ax.plot(prwave, pnflux, '-', color = 'orange')
            fluxlist = np.append(fluxlist, pnflux)
            if title == None:
                psave = str(plow) + '_' + str(phigh) + '_paper_model'

            else:
                psave = title.replace(' ', '_') + '_paper_model'
            #np.save(psave, [prwave, pnflux])

    plt.title(title)
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.ylabel('Normalized Flux')

    if plotx == None:
        xlow = limits[0]
        xhigh = limits[1]
        xdiff = xhigh - xlow
        plotlow = xlow + 0.25*xdiff
        plothigh = xhigh - 0.25*xdiff
        plt.xlim((plotlow, plothigh))
        if title == None:
            title = 'Comparison Plot from ' + str(plotlow) + 'to ' + str(plothigh)
    else:
        plt.xlim(plotx)
        if title == None:
            title = 'Comparison Plot from ' + str(plotrange[0]) + 'to ' + str(plotrange[1])

    if ploty == None:
        ydiff = np.max(fluxlist) - np.min(fluxlist)
        maxplot = np.max(fluxlist) + 0.1*ydiff
        minplot = np.min(fluxlist)- 0.1*ydiff
        plt.ylim((minplot, maxplot))
    else:
        plt.ylim(ploty)

    plt.title(title)
    if mode == 'SAVE':
        fname = title.replace(' ', '_') + '.pdf'
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()

def velplot(limits, clambda, keck = [], xshoot = [], paperhires = [],
    paperesi = [], papermodel = [], model = [],
    koff = 0.0, xoff = +15.0, poff = 0.0, moff = +55.0,
    title = None, plotx = None, ploty = None, mode = 'SAVE'):

    plow = limits[0]
    phigh = limits[1]
    ax = plt.subplot(111)
    fluxlist = []
    vellist = []
    title = title + ' Velocity'

    if  '4923' in title:
        keckfile = idlread('wd1145+017.sav')
        greenwave = keckfile.sgreen.wave[0]
        greenflux = keckfile.sgreen.flux[0]
        greenerr = keckfile.sgreen.err[0]
        kecktemp = [greenwave[0], greenflux[0], greenerr[0]]
        keck = binarray(kecktemp[0], kecktemp[1], kecktemp[2], len(kecktemp[0])/10.0)
    if len(keck) > 0:
        kfmask = np.where((np.isfinite(keck[0])) & (np.isfinite(keck[1])))
        kwave = keck[0][kfmask]
        kflux = keck[1][kfmask]
        kerr = keck[2][kfmask]
        ktemp = [kwave, kflux, kerr]
        kwave = kwave[np.argsort(ktemp[0])]
        kflux = kflux[np.argsort(ktemp[0])]
        kerr = kerr[np.argsort(ktemp[0])]
        krmask = np.where((kwave >= plow) & (kwave <= phigh))
        krflux = kflux[krmask]
        krerr = kerr[krmask]
        krwave = kwave[krmask]
        if len(krwave) > 0:
            #knflux = krflux/iterfit(krwave, krflux, degree = 3, niter = 25)
            krmask = np.where((0.0 < krflux/np.median(krflux)) & (krflux/np.median(krflux) < 1.5))
            krwave = krwave[krmask]
            krerr = krerr[krmask]
            krflux = krflux[krmask]
            knflux = krflux
            knflux = knflux/np.median(knflux)
            krvel = wave2vel(krwave, clambda) + koff
            ax.plot(krvel, knflux, '-k', alpha = 0.3, label = 'November Keck HIRES')
            ax.fill_between(krvel, knflux - krerr, knflux + krerr, facecolor = 'k', alpha = 0.05)
            fluxlist = np.append(fluxlist, knflux)
            vellist = np.append(vellist, krvel)
            if title == None:
                kecksave = str(plow) + '_' + str(phigh) + '_keck'

            else:
                kecksave = title.replace(' ', '_') + '_keck'
            #np.save(kecksave, [krvel, knflux, krerr])
    if len(xshoot) > 0:
        xfmask = np.where((np.isfinite(xshoot[0])) & (np.isfinite(xshoot[1])))
        xwave = xshoot[0][xfmask]*10.0
        xflux = xshoot[1][xfmask]
        xrmask = np.where((xwave >= plow) & (xwave <= phigh))
        xrwave = xwave[xrmask]
        xrflux = xflux[xrmask]

        if len(xrwave) > 0:
            #xnflux = xrflux/iterfit(xrwave, xrflux, degree = 3, niter= 25)
            xrmask = np.where((0.0 < xrflux/np.median(xrflux)) & (xrflux/np.median(xrflux) < 1.5))
            xrwave = xrwave[xrmask]
            xrflux = xrflux[xrmask]
            xnflux = xrflux
            xnflux = xnflux/np.median(xnflux)
            xrvel = wave2vel(xrwave, clambda) + xoff
            ax.plot(xrvel, xnflux, '-g', label = 'XSHOOTER')
            fluxlist = np.append(fluxlist, xnflux)
            if title == None:
                xsave = str(plow) + '_' + str(phigh) + '_xshoot'

            else:
                xsave = title.replace(' ', '_') + '_xshoot'
            #np.save(xsave, [xrvel, xnflux])
    if len(model) > 0:
        mfmask = np.where((np.isfinite(model[0])) & (np.isfinite(model[1])))
        mwave = (model[0][mfmask])
        mflux = model[1][mfmask]
        mrmask = np.where((mwave >= plow) & (mwave <= phigh))
        mrwave = mwave[mrmask]
        mrflux = mflux[mrmask]
        if len(mrwave) > 0:
            #mnflux = mrflux/iterfit(mrwave, mrflux, degree = 3, niter = 25)
            mrmask = np.where((0.0 < mrflux/np.median(mrflux)) & (mrflux/np.median(mrflux) < 1.5))
            mrwave = mrwave[mrmask]
            mrflux = mrflux[mrmask]
            mnflux = mrflux
            mnflux = mnflux/np.median(mnflux)
            mrvel = wave2vel(vac2air(vel2wave(wave2vel(mrwave, clambda) + moff, clambda)), clambda)
            ax.plot(mrvel, mnflux, '--r', label ='Stellar Model')
            fluxlist = np.append(fluxlist, mnflux)
            vellist = np.append(vellist, mrvel)
            if title == None:
                msave = str(plow) + '_' + str(phigh) + '_model'

            else:
                msave = title.replace(' ', '_') + '_model'
            #np.save(msave, [mrvel, mnflux])

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
            prvel = wave2vel(prwave, clambda) + poff
            ax.plot(prvel, pnflux, ':', color = 'orange', label = 'Xu Keck HIRES')
            fluxlist = np.append(fluxlist, pnflux)
            vellist = np.append(vellist, prvel)
            if title == None:
                psave = str(plow) + '_' + str(phigh) + '_paper_hires'

            else:
                psave = title.replace(' ', '_') + '_paper_hires'
            #np.save(psave, [prvel, pnflux])

    if len(paperesi) > 0:
        pfmask = np.where((np.isfinite(paperesi[0])) & (np.isfinite(paperesi[1])) & (paperesi > 0.0))
        pwave = paperesi[0][pfmask]
        pflux = paperesi[1][pfmask]
        prmask = np.where((pwave <= phigh) & (pwave >= plow))
        prwave = pwave[prmask]
        prflux = pflux[prmask]
        pnflux = prflux
        if ((len(prwave) > 0) & (title != 'CaII 8498') & ('Na D' not in title)):
            #pnflux = prflux/iterfit(prwave, prflux, degree = 3, niter= 25)
            if np.isfinite(np.median(pnflux)) & (np.median(pnflux) != 0.0):
                pass
            else:
                prwave = prwave[np.where((np.isfinite(pnflux)) & (pnflux > 0.0))]
                pnflux = pnflux[np.where((np.isfinite(pnflux)) & (pnflux > 0.0))]
            pnflux = pnflux/np.median(pnflux)
            prmask = np.where((0.0 < prflux/np.median(prflux)) & (prflux/np.median(prflux) < 1.5))
            prwave = prwave[prmask]
            prflux = prflux[prmask]
            pnflux = prflux
            if np.isfinite(np.median(pnflux)) & (np.median(pnflux) != 0.0):
                pass
            else:
                pnflux = pnflux[np.where((np.isfinite(pnflux)) & (pnflux > 0.0))]
            pnflux = pnflux/np.median(pnflux)
            prvel = wave2vel(prwave, clambda) + poff
            ax.plot(prvel, pnflux, '-.', color = 'orange', label = 'Xu Keck ESI')
            fluxlist = np.append(fluxlist, pnflux)
            vellist = np.append(vellist, prvel)
            if title == None:
                psave = str(plow) + '_' + str(phigh) + '_paper_esi'

            else:
                psave = title.replace(' ', '_') + '_paper_esi'
            #np.save(psave, [prvel, pnflux])

    plt.title(title)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Normalized Flux')

    if plotx == None:
        xlow = limits[0]
        xhigh = limits[1]
        xdiff = xhigh - xlow
        plotlow = xlow + 0.25*xdiff
        plothigh = xhigh - 0.25*xdiff
        plt.xlim((-600.0, +600.0))
        if title == None:
            title = 'Comparison Plot from ' + str(plotlow) + 'to ' + str(plothigh)
    else:
        plt.xlim(plotx)
        if title == None:
            title = 'Comparison Plot from ' + str(plotrange[0]) + 'to ' + str(plotrange[1])

    if ploty == None:
        masknew = np.where((-600.0 < vellist) & (vellist < 600.0))
        ydiff = np.max(fluxlist[masknew]) - np.min(fluxlist[masknew])
        maxplot = np.max(fluxlist[masknew]) + 0.1*ydiff
        minplot = np.min(fluxlist[masknew])- 0.1*ydiff
        if minplot < 0.0:
            minplot = 0.0
        if maxplot > 1.5:
            maxplot = 1.5
        plt.ylim((minplot, maxplot))
        plt.plot([42.0]*100, np.linspace(minplot, maxplot, 100), '-b')

    else:
        plt.ylim(ploty)

    plt.title(title.replace('-', '.'))
    plt.legend()
    if mode == 'SAVE':
        fname = title.replace(' ', '_') + '.pdf'
        plt.savefig('masterplots/' +fname)
        #plt.savefig(fname)
    else:
        plt.show()
    plt.clf()
if __name__ == '__main__':
    #limitarr = [(3571,3593), (3621, 3643), (3637, 3659), (3724, 3746), (3738, 3760)]
    #offsetarr = [-0.4, -0.4, -0.4, -0.45, -0.45]
    #plotx = [(3370,3377), (3386, 3392), (3390.5, 3398), (3683, 3691), (3899, 3903)]
    #ploty = [(0.5,1.25), (0.7, 1.2), (0.7,1.25), (0.6,1.25), (0.5,1.3)]
    keck = keckread()
    xshoot = xshootread()
    model = conmodelread()
    esi = readesi()
    hires = readhires()
    pmodel = readmodel()
    fname = 'Ca_II_master.txt'
    #temptitles = np.loadtxt(fname, usecols = [3,4], dtype = 'string')
    clambdas = np.loadtxt(fname, usecols = [0])
    #mask = np.where((clambdas > 5316.0) & (clambdas < 5317.0))[0]
    #clambdas = clambdas[mask]
    #temptitles = temptitles[mask]
    titles = ['Ca' + ' ' + 'II' + ' ' + str(clambdas[i]).replace('.', '-') for i in range(0,len(clambdas))]
    limitarr = [(vel2wave(-700.0, clambda), vel2wave(+700.0, clambda)) for clambda in clambdas]

    print len(clambdas)
    #Done with 0,12000
    #Total = 22000
    for i in range(0,len(clambdas)):
        velplot(limitarr[i], clambda = clambdas[i], keck = keck, xshoot = xshoot,
            model = model, paperesi = esi, paperhires = hires, papermodel = pmodel,
            title = titles[i], mode = 'SAVE')

    '''
    testfunc = interp1d(masterwave, masterflux)
    mask = np.where((restrictwave > np.sort(masterwave)[0]) & (restrictwave < np.sort(masterwave)[-1]))
    normflux, normwave = polynorm(restrictwave[mask], restrictflux[mask], testfunc)
    np.save('flat5model', [normwave, normflux])
    print 'Continuum normalized model against raw data'

    masterflux2, masterwave2 = polynorm2(masterwave, masterflux)
    np.save('pnormdatabin5', [masterwave2, masterflux2, mastererr])

    print 'Polynomial normalized model '
    testfunc = interp1d(masterwave2, masterflux2)
    mask = np.where((restrictwave > np.sort(masterwave2)[0]) & (restrictwave < np.sort(masterwave2)[-1]))
    normflux2, normwave2 = polynorm(restrictwave[mask], restrictflux[mask], testfunc)

    print 'Continuum normalized model against polynomial flattened data'

    #np.save('databin5', [masterwave, masterflux, mastererr])
    np.save('pnormdatabin5', [masterwave2, masterflux2, mastererr])
    #np.save('modelconvolve9', [restrictwave, restrictflux])

    np.save('flat5model', [normwave, normflux])
    np.save('poly5model', [normwave2, normflux2])

    print 'Saved npy files'

    databin5 = np.load('databin5.npy')
    poly5model = np.load('poly5model.npy')
    pnormdatabin5 = np.load('pnormdatabin5.npy')
    flat5model = np.load('flat5model.npy')

    normwave = flat5model[0]
    normflux = flat5model[1]
    normwave2 = poly5model[0]
    normflux2 = poly5model[1]

    masterwave = databin5[0]
    masterflux = databin5[1]
    masterwave2 = pnormdatabin5[0]
    masterflux2 = pnormdatabin5[1]
    mastererr = pnormdatabin5[2]

    print normwave, normwave2, masterwave, masterwave2
    print normflux, normflux2, masterflux, masterflux2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(masterwave, masterflux, yerr = mastererr, fmt = '.k', alpha = 0.4)
    ax.plot(normwave, normflux, '-r')

    #plt.xlim((3050,3150))
    ax.set_ylim((-0.1, 1.5))
    def onclick(event):
        print event.xdata
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))

    plt.show()
    plt.clf()

    ranges = [(4230,4242), (3683,3690), (3702,3710), (3768,3774), (3346,3355),
            (3498, 3504), (3512, 3520), (3628, 3636), (3740, 3746), (3746, 3754),
            (4176, 4184), (4348, 4358), (4518, 4530), (4544, 4556), (4554, 4562),
            (4575, 4600), (4620, 4645), (4680, 4760), (5305, 5330), (5560, 5620)]

    areas = np.empty((len(ranges), 3))
    widths = np.empty((len(ranges), 3))

    for i in range(0,len(ranges)):
        datamask = np.where((masterwave2 >= ranges[i][0] - 5.0) & (masterwave2 <= ranges[i][1] + 5.0))
        normmask = np.where((normwave >= ranges[i][0] - 8.0) & (normwave <= ranges[i][1] + 8.0))
        datay = masterflux2[datamask]
        datax = masterwave2[datamask]
        modely = normflux[normmask]
        modelx = normwave[normmask]



        datay = datay[np.where(np.isfinite(datay))]
        datax = datax[np.where(np.isfinite(datax))]
        modely = modely[np.where(np.isfinite(modely))]
        modelx = modelx[np.where(np.isfinite(modelx))]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        def onclick(event):
            print event.xdata
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))

        plt.errorbar(datax, datay, yerr = mastererr[datamask], fmt = '.k', alpha = 0.4)
        plt.plot(modelx, modely, '-r')
        plt.show()

        modelcont, modelline = linecont([modelx, datax], [modely, datay])
        datacont, dataline = linecont([datax, modelx], [datay, modely])

        eqx = [datax, modelx]
        eqy = [datay, modely]
        eqline = [dataline, modelline]
        eqcont = [datacont, modelcont]

        area, eqwidth = equilwidth(eqx, eqy, eqline, eqcont)
        areas[i] = area
        widths[i] = eqwidth
        np.save('areas', areas)
        np.save('eqwidths', widths)

    widths = np.load('eqwidths.npy')
    print widths[:,2]
    print widths[:,0] - widths[:,1]
    '''
