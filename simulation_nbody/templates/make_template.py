import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import interpolate


def read_lc(fname='lc_fine_0_max_0.csv'):
    df = pd.read_csv(fname)
    flux = np.array(df.Flux)
    time = np.array(df.Time)
    return time, flux


def filter_flux(fname='lc_fine_0_max_0.csv'):
    time, flux = read_lc(fname)
    deep_loc = np.argmin(flux[-500:])
    cut_loc = -2*(500 - deep_loc)
    time = time[cut_loc:]
    flux = flux[cut_loc:]
    filter_flux_9 = sig.savgol_filter(flux, 9, 2)
    template_filter = -1.0*(filter_flux_9 - 1.0)
    template_filter = template_filter*(1.0/np.max(template_filter))
    template_filter[np.where(template_filter < 0.0)] = 0.0
    return time, template_filter


def get_template(fname='lc_fine_0_max_0.csv'):
    time, template_filter = filter_flux()
    transit_loc = np.linspace(0.0, 1.0, len(time))
    return transit_loc, template_filter


def save_template(save_name, fname='lc_fine_0_max_0.csv'):
    transit_loc, template_filter = get_template(fname)
    template_array = np.array([transit_loc, template_filter])
    array_save = save_name + '_array.npy'
    np.save(array_save, template_array)


def make_template_func(array_name):
    template_array = np.load(array_name)
    transit_loc = template_array[0]
    template_filter = template_array[1]
    template_func = interpolate.interp1d(transit_loc, template_filter,
                                         kind='cubic',
                                         bounds_error=False,
                                         fill_value=(0.0, 0.0))
    return template_func


if __name__ == '__main__':
    save_template('4_43_1_35')
