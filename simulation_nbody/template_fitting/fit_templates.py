import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pandas as pd
import scipy.optimize as op
from make_template import make_template_func
from compare_templates import read_templates


def eval_template_no_start(params, x, t_1_func, start):
    width, amp, offset = params
    end = start + width
    flux = np.ones_like(x)*offset
    mask = np.where((x > start) & (x <= end))
    pos = np.array([(x[mask][i] - start)/(end - start)
                    for i in range(0, len(mask[0]))])
    pos_fine = np.linspace(pos[0], pos[-1], 1000)
    flux_remove = amp*t_1_func(pos_fine)
    bin_indices = np.digitize(pos_fine, pos)
    flux[mask] = offset - amp*t_1_func(pos)
    bin_test = np.array([flux_remove[bin_indices == i].mean()
                        for i in range(1, len(pos))])
    flux[mask] = offset - bin_test
    flux[np.where(flux < 0.0)] = 0.0
    return flux


def eval_template(params, x, t_1_func):
    start, width, amp, offset = params
    end = start + width
    flux = np.ones_like(x)*offset
    mask = np.where((x > start) & (x <= end))
    pos = np.array([(x[mask][i] - start)/(end - start)
                    for i in range(0, len(mask[0]))])
    try:
        pos_fine = np.linspace(pos[0], pos[-1], 1000)
    except:
        return flux
    flux_remove = amp*t_1_func(pos_fine)
    bin_indices = np.digitize(pos_fine, pos)
    flux[mask] = offset - amp*t_1_func(pos)
    bin_test = np.array([flux_remove[bin_indices == i].mean()
                        for i in range(1, len(pos))])
    flux[mask] = offset - bin_test
    flux[np.where(flux < 0.0)] = 0.0
    return flux


def lnprior_no_start(params):
    width, amp, offset = params
    if 0.0 < width < 0.1 and 0.0 < amp and np.isfinite(offset):
        return 0.0
    return -np.inf


def lnlike_1_no_err_start(params, x, y, t_1_func, start, noise_est=0.01):
    flux = eval_template(params, x, t_1_func, start)
    chi2 = ((y - flux)/noise_est)**2.0
    return -0.5*np.sum(chi2)


def lnprob_1_no_err_start(params, x, y, t_1, start):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_1_no_err(params, x, y, t_1, start)


def lnprior(params):
    start, width, amp, offset = params
    if 0.0 < width < 0.1 and 0.0 < amp and np.isfinite(offset):
        if -0.05 < start < 0.05:
            return 0.0
    return -np.inf


def lnlike_1_no_err(params, x, y, t_1_func, noise_est=0.01):
    flux = eval_template(params, x, t_1_func)
    chi2 = ((y - flux)/noise_est)**2.0
    return -0.5*np.sum(chi2)


def lnprob_1_no_err(params, x, y, t_1):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_1_no_err(params, x, y, t_1)


def eval_template_multi(params, flux, x, t_func):
    start, width, amp = params
    end = start + width
    flux = np.ones_like(x)*offset
    mask = np.where((x > start) & (x <= end))
    pos = np.array([(x[mask][i] - start)/(end - start)
                    for i in range(0, len(mask[0]))])
    pos_fine = np.linspace(pos[0], pos[-1], 1000)
    flux_remove = amp*t_1_func(pos_fine)
    bin_indices = np.digitize(pos_fine, pos)
    flux[mask] = offset - amp*t_1_func(pos)
    bin_test = np.array([flux_remove[bin_indices == i].mean()
                        for i in range(1, len(pos))])
    flux[mask] = offset - bin_test
    flux[np.where(flux < 0.0)] = 0.0
    return flux


def lnprior_templates(params):
    start, width, amp = params
    if 0.0 < width < 0.1 and 0.0 < amp:
        if -0.1 < start < 0.2:
            return 0.0
    return -np.inf


def lnlike_2_no_err(params, x, y, t_1_func, t_2_func, noise_est=0.01):
    t_1_params = params[0:3]
    t_2_params = params[3:6]
    offset = params[-1]
    flux = offset*np.ones_like(x)
    flux = eval_template(t_1_params, flux, x, t_1_func)
    flux = eval_template(t_2_params, flux, x, t_2_func)
    chi2 = ((y - flux)/noise_est)**2.0
    return -0.5*np.sum(chi2)


def lnprob_2_no_err(params, x, y, t_1_func, t_2_func):
    lp = lnprior_templates(params[0:3]) + lnprior_templates(params[3:6])
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(params[-1]):
        return -np.inf
    return lp + lnlike_2_no_err(params, x, y, t_1_func, t_2_func)


def lnlike_3_no_err(params, x, y,
                    t_1_func, t_2_func, t_3_func, noise_est=0.1):
    t_1_params = params[0:3]
    t_2_params = params[3:6]
    t_3_params = params[6:9]
    offset = params[-1]
    flux = offset*np.ones_like(x)
    flux = eval_template(t_1_params, flux, x, t_1_func)
    flux = eval_template(t_2_params, flux, x, t_2_func)
    flux = eval_template(t_3_params, flux, x, t_3_func)
    chi2 = ((y - flux)/noise_est)**2.0
    return -0.5*np.sum(chi2)


def lnprob_3_no_err(params, x, y, t_1_func, t_2_func, t_3_func):
    lp = lnprior_templates(params[0:3]) + lnprior_templates(params[3:6])
    lp = lp + lnprior_templates(params[6:9])
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(params[-1]):
        return -np.inf
    return lp + lnlike_3_no_err(params, x, y, t_1_func, t_2_func, t_3_func)


def read_gansicke():
    gdata = pd.read_csv('gansicke2017_data.txt', delim_whitespace=True)
    data_time = np.array(gdata['Time'])
    data_flux = np.array(gdata['Flux'])
    return data_time, data_flux


def fit_all_single(data_time, data_flux, mask_low, mask_high, initguess_1,
                   burn_1_steps=250, burn_2_steps=250, production_steps=1000,
                   nwalkers=100, best_plot='SAVE', triangle='SAVE',
                   signal_label='6'):
    mask = np.where((data_time > mask_low) & (data_time < mask_high))
    x = data_time[mask] - data_time[mask][0]
    y = data_flux[mask]

    for t_1 in read_templates():
        t_1_func = make_template_func(t_1)
        nll = lambda *args: -lnlike_1_no_err(*args)
        result = op.minimize(nll, initguess_1, args=(x, y, t_1_func),
                             bounds=[(-0.05, 0.05),
                                     (0.0, 0.1), (0.0, None), (0.85, 1.15)],
                             method='COBYLA')
        initguess = result["x"]
        ndim = 4
        pos = [initguess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim, lnprob_1_no_err,
                                        args=(x, y, t_1_func))

        print 'Running Burn-in 1'
        p0, prob, state = sampler.run_mcmc(pos, burn_1_steps)
        sampler.reset()
        print 'Running Burn-in 2'
        p = p0[np.argmax(prob)]
        p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
        p0, _, _ = sampler.run_mcmc(p0, burn_2_steps)
        sampler.reset()
        print 'Running production'
        p0, prob, state = sampler.run_mcmc(p0, production_steps)
        p = p0[np.argmax(prob)]
        print (-2.0*np.max(prob)/3.0), p0[np.argmax(prob)]

        model_y = eval_template(p, x, t_1_func)
        plt.plot(x, y, '.')
        plt.plot(x, model_y, '-')
        plotname = 'Best Fit for ' + t_1[:-10]
        plot_save_name = "best_model_" + signal_label + '_'
        plot_save_name = plot_save_name + t_1[:-10] + '.png'
        plt.title(plotname)
        if best_plot == 'SAVE':
            plt.savefig(plot_save_name)
        else:
            plt.show()
        plt.clf()
        plt.close()

        samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
        fig = corner.corner(samples, labels=["$l$", "$w$", "$A$", "Offset"])
        save_name = "triangle_" + signal_label + '_' + t_1[:-10] + '.png'
        trianglename = 'Posteriors for ' + t_1[:-10]
        plt.suptitle(trianglename)
        if triangle == 'SAVE':
            fig.savefig(save_name)
        else:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    data_time, data_flux = read_gansicke()
    fit_all_single(data_time, data_flux, 368.445, 369.0,
                   [-0.001, 0.018, 0.36, 0.9],
                   signal_label='f4')
