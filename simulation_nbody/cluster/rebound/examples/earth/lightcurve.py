import numpy as np
from csv_manage import *
from visualize import plot_light_curve


def get_wd_pos(wd_hash, hash_df, step_df):
    wd_rad = hash_df[' Radius'][wd_hash]
    wd_x = step_df[' x'][wd_hash]
    wd_y = step_df[' y'][wd_hash]
    wd_z = step_df[' z'][wd_hash]
    return wd_rad, wd_x, wd_y, wd_z


def box_mask(wd_hash, step_df, hash_df, box_factor=1.15):
    wd_loc = step_df.index.get_loc(wd_hash)
    wd_rad, wd_x, wd_y, wd_z = get_wd_pos(wd_hash, hash_df, step_df)
    particles = step_df.drop(step_df.index[wd_loc])
    front_mask = np.where(particles[' x'] < 0.0)
    front_particles = particles.drop(particles.index[front_mask])
    y_low = wd_y - box_factor*wd_rad
    y_high = wd_y + box_factor*wd_rad
    z_low = wd_z - box_factor*wd_rad
    z_high = wd_z + box_factor*wd_rad
    box_mask = np.where((y_low > front_particles[' y']) |
                        (front_particles[' y'] > y_high) |
                        (z_low > front_particles[' z']) |
                        (front_particles[' z'] > z_high))
    box_particles = front_particles.drop(front_particles.index[box_mask])
    box_particles[' x'] = box_particles[' x'] - wd_x
    box_particles[' y'] = box_particles[' y'] - wd_y
    box_particles[' z'] = box_particles[' z'] - wd_z
    return box_particles


def make_pixel_array(wd_rad, box_factor=1.15, wd_pix_num=5.0e5):
    pixel_width = np.sqrt(np.pi*(wd_rad**2.0)/wd_pix_num)
    pixel_row_num = int(2.0*box_factor*wd_rad/pixel_width)
    return np.zeros((pixel_row_num, pixel_row_num)), pixel_width


def coord_to_pixel_loc(coord, pixel_width, wd_rad, box_factor=1.15):
    return (coord + box_factor*wd_rad)/pixel_width


def pixel_loc_to_coord(pixel_loc, pixel_width, wd_rad, box_factor=1.15):
    return (pixel_loc*pixel_width) - (box_factor*wd_rad)


def change_pixel_value(pixel_array, x_coord, y_coord, particle_rad,
                       particle_inflate, pixel_width, wd_rad,
                       box_factor=1.15, value_change=2.0):
    array_shape = np.shape(pixel_array)
    arr_1 = np.arange(0, array_shape[0], 1)
    arr_2 = np.arange(0, array_shape[1], 1)
    x = pixel_loc_to_coord(arr_1, pixel_width, wd_rad, box_factor)
    y = pixel_loc_to_coord(arr_1, pixel_width, wd_rad, box_factor)
    xx, yy = np.meshgrid(x, y)
    pixel_pos = (xx - x_coord)**2.0 + (yy - y_coord)**2.0
    mask = np.where(pixel_pos <= (particle_rad*particle_inflate)**2.0)
    pixel_array[mask] = value_change
    return pixel_array


def change_wd_pixel_value(pixel_array, x_coord, y_coord, particle_rad,
                          pixel_width, wd_rad,
                          box_factor=1.5, value_change=2.0):
    return change_pixel_value(pixel_array, x_coord, y_coord, particle_rad,
                              1.0, pixel_width, wd_rad,
                              box_factor=1.15, value_change=value_change)


def r_const(particles, index_loc, inflate_factor):
    return inflate_factor


def particles_pixel_change(particles, pixel_array,
                           pixel_width, wd_rad, inflate_func=r_const,
                           box_factor=1.15, value_change=1.0,
                           inflate_args=1.0):

    for i in range(0, len(particles.index)):
        pixel_array = change_pixel_value(pixel_array, particles[' y'].iloc[i],
                                         particles[' z'].iloc[i],
                                         particles['Radius'].iloc[i],
                                         inflate_func(particles, i,
                                                      inflate_args),
                                         pixel_width, wd_rad,
                                         box_factor=box_factor,
                                         value_change=value_change)
    return pixel_array


def get_flux_timestep(month_num, step_pos, hash_df,
                      box_factor=1.15, wd_pix_num=3.0e3,
                      wd_value=2.0, particle_value=1.0,
                      inflate_func=r_const, inflate_args=1.0,
                      wd_pixel_array=None):
    N = get_N_particles(hash_df)
    step_df = read_time_step(month_num, N, step_pos)
    wd_hash = get_wd_hash()
    particles = box_mask(wd_hash, step_df, hash_df, box_factor=box_factor)
    wd_rad, wd_x, wd_y, wd_z = get_wd_pos(wd_hash, hash_df, step_df)
    pixel_array, pixel_width = make_pixel_array(wd_rad,
                                                box_factor=box_factor,
                                                wd_pix_num=wd_pix_num)
    if wd_pixel_array is None:
        pixel_array = change_wd_pixel_value(pixel_array, 0.0, 0.0, wd_rad,
                                            pixel_width, wd_rad,
                                            box_factor=box_factor,
                                            value_change=wd_value)
        wd_pixel_array = np.copy(pixel_array)
    else:
        pixel_array = np.copy(wd_pixel_array)
    wd_count = float(np.shape(np.where(pixel_array == wd_value))[1])
    pixel_array = particles_pixel_change(particles, pixel_array,
                                         pixel_width, wd_rad,
                                         inflate_func=inflate_func,
                                         box_factor=box_factor,
                                         value_change=particle_value,
                                         inflate_args=inflate_args)
    block_count = float(np.shape(np.where(pixel_array == particle_value))[1])
    return 1.0 - block_count/wd_count, wd_pixel_array


def make_light_curve(month_num,
                     box_factor=1.15, wd_pix_num=5.0e3, wd_value=2.0,
                     particle_value=1.0, inflate_func=r_const,
                     inflate_args=1.0, month_suffix='_months.csv',
                     dt=10.0, month=2.628e6, wd_pixel=None,
                     time_min=0.0, time_max=None, time_step=1.0):
    hash_df = read_hash()
    N = get_N_particles(hash_df)
    if time_max is None:
        timesteps = num_time_steps(month_num, N, month_suffix=month_suffix)
    month_start = month_num*month + time_min*dt
    month_end = month_start + timesteps*dt
    time = np.arange(month_start, month_end, time_step*dt, dtype=float)
    flux = np.empty_like(time)
    test = np.arange(0, timesteps, time_step, dtype=int)
    for i in range(0, len(test)):
        flux[i], wd_pixel = get_flux_timestep(month_num,
                                              test[i], hash_df,
                                              box_factor=box_factor,
                                              wd_pix_num=wd_pix_num,
                                              wd_value=wd_value,
                                              particle_value=particle_value,
                                              inflate_func=inflate_func,
                                              inflate_args=inflate_args,
                                              wd_pixel_array=wd_pixel)
        print time[i], ' out of ', time[-1]
    return time, flux


def save_light_curve(time, flux, save_name):
    light_curve = pd.DataFrame({'Time': time, 'Flux': flux})
    light_curve.to_csv(save_name)


def coarse_light_curves_all(coarse_step=6.0, fmt='.k',
                            box_factor=1.15, wd_pix_num=5.0e3,
                            wd_value=2.0, particle_value=1.0,
                            inflate_func=r_const, inflate_args=1.0,
                            month_suffix='_months.csv', dt=10.0,
                            month=2.628e6, wd_pixel=None,
                            time_min=0, time_max=None):
    _, month_nums, _ = get_month_num(month_suffix)
    for month_num in month_nums:
        time, flux = make_light_curve(month_num,
                                      box_factor=box_factor,
                                      wd_pix_num=wd_pix_num,
                                      wd_value=wd_value,
                                      particle_value=particle_value,
                                      inflate_func=inflate_func,
                                      inflate_args=inflate_args,
                                      month_suffix=month_suffix,
                                      dt=dt, month=month, wd_pixel=wd_pixel,
                                      time_min=time_min, time_max=time_max,
                                      time_step=coarse_step)
        save_name = 'lc_coarse_' + str(time_min) + '_'
        if time_max is None:
            save_name = save_name + 'max_'
        else:
            save_name = save_name + str(time_max) + '_'
        save_name = save_name + str(month_num) + '.csv'
        plot_name = save_name[:-4]
        plot_title = 'Month ' + str(month_num) + ' Coarse Light Curve'
        save_light_curve(time, flux, save_name)
        plot_light_curve(time, flux, plot_title,
                         fmt, savetitle=plot_name, mode='SAVE')


def fine_light_curves_all(coarse_step=1.0, fmt='.k',
                          box_factor=1.15, wd_pix_num=5.0e3,
                          wd_value=2.0, particle_value=1.0,
                          inflate_func=r_const, inflate_args=1.0,
                          month_suffix='_months.csv', dt=10.0,
                          month=2.628e6, wd_pixel=None,
                          time_min=0, time_max=None):
    _, month_nums, _ = get_month_num(month_suffix)
    for month_num in month_nums:
        time, flux = make_light_curve(month_num,
                                      box_factor=box_factor,
                                      wd_pix_num=wd_pix_num,
                                      wd_value=wd_value,
                                      particle_value=particle_value,
                                      inflate_func=inflate_func,
                                      inflate_args=inflate_args,
                                      month_suffix=month_suffix,
                                      dt=dt, month=month, wd_pixel=wd_pixel,
                                      time_min=time_min, time_max=time_max,
                                      time_step=coarse_step)
        save_name = 'lc_fine_' + str(time_min) + '_'
        if time_max is None:
            save_name = save_name + 'max_'
        else:
            save_name = save_name + str(time_max) + '_'
        save_name = save_name + str(month_num) + '.csv'
        plot_name = save_name[:-4]
        plot_title = 'Month ' + str(month_num) + 'Light Curve'
        save_light_curve(time, flux, save_name)
        plot_light_curve(time, flux, plot_title,
                         fmt, savetitle=plot_name, mode='SAVE')


if __name__ == '__main__':
    coarse_light_curves_all()
    fine_light_curves_all()
