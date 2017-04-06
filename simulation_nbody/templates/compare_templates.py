import numpy as np
import matplotlib.pyplot as plt
from make_template import make_template_func
from os import listdir as ls
from itertools import combinations


def calc_resids(t_1, t_2):
    return t_1 - t_2


def calc_chi_sq(resids):
    return np.sum(resids**2.0)


def inject_noise(noise_amp, template):
    noise = noise_amp*np.random.normal(0, 1.0, len(template))
    return template + noise


def read_templates():
    fnames = ls('.')
    template_arrays = [fname
                       for fname in fnames if fname.endswith('array.npy')]
    return template_arrays


def compare_N_chi_squared(N=5):
    template_array_names = read_templates()
    test_x = np.linspace(0.0, 1.0, 360)
    template_ys = np.empty((12, 360))
    for i in range(0, len(template_array_names)):
        template_func = make_template_func(template_array_names[i])
        template_ys[i, :] = template_func(test_x)
    template_names = [template_array_name[:-10]
                      for template_array_name in template_array_names]
    comp_names = []
    chi_values = []
    for i in range(0, 12):
        for j in range(i, 12):
            if i != j:
                resids = calc_resids(template_ys[i], template_ys[j])
                test_name = template_names[i] + ' - ' + template_names[j]
                comp_names = np.append(comp_names,
                                       test_name)
                # label_name = template_names[i] + ' - ' + template_names[j]
                # label_name += '_ Chi^2 = '
                # plt.plot(test_x, resids, '.k')
                # plt.xlim(0.0, 1.0)
                # plt.ylim(-0.8, 0.8)
                chi_count = 0.0
                for q in range(0, 2000):
                    t_1 = inject_noise(0.1, template_ys[i])
                    t_2 = inject_noise(0.1, template_ys[j])
                    noise_resids = calc_resids(t_1, t_2)
                    chi_count += calc_chi_sq(noise_resids)
                    # plt.plot(test_x, noise_resids, '-b', alpha=0.01)
                chi_count = chi_count/2000.0
                chi_values = np.append(chi_values, chi_count)
                # label_name += str(chi_count)
                # plt.title(label_name)
                # plt.show()
                # plt.close()
                # plt.clf()
            else:
                pass
    all_combos = list(combinations(template_names, N))
    result_test = all_combos[0]
    result_chi = 0.0
    for q in range(0, len(all_combos)):
        current_test = all_combos[q]
        current_test_chi = 0.0
        for i in range(0, N):
            for j in range(i, N):
                if i != j:
                    name_loc = [q for q in range(0, len(chi_values)) if
                                ((current_test[i] in comp_names[q]) &
                                (current_test[j] in comp_names[q]))]
                    current_test_chi += chi_values[name_loc][0]
        if current_test_chi >= result_chi:
            result_chi = current_test_chi
            result_test = current_test
            print result_test, result_chi
    print result_test, result_chi


if __name__ == '__main__':
    compare_N_chi_squared(4)
