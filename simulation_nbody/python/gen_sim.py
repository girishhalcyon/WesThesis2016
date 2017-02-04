import numpy as np
import pandas as pd

packing_path = str('/Volumes/westep/girish/' +
                   'WesThesis2016/simulation_nbody/python/packings_dir')


def plot_2D_simulation(rebsim, color_mass=False, color_dens=True, mode='SHOW'):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    N_tot = rebsim.N
    particle_arr = rebsim.particles
    xs = np.empty((N_tot))
    ys = np.empty_like(xs)
    zs = np.empty_like(xs)
    rad_arr = np.empty_like(xs)
    if any([color_mass, color_dens]):
        mass_arr = np.empty_like(xs)
    else:
        pass
    for i in range(0, N_tot):
        particle = particle_arr[i]
        xs[i] = particle.x
        ys[i] = particle.y
        zs[i] = particle.z
        rad_arr[i] = particle.r
        if any([color_mass, color_dens]):
            mass_arr[i] = particle.m
    argloc = np.argmax(mass_arr)
    xs = np.delete(xs, argloc)
    ys = np.delete(ys, argloc)
    zs = np.delete(zs, argloc)
    rad_arr = np.delete(rad_arr, argloc)
    mass_arr = np.delete(mass_arr, argloc)
    if color_dens:
        dens_arr = mass_arr/((4.0/3.0)*np.pi*(rad_arr**3.0))
        mappable = ax1.scatter(xs, ys, c=dens_arr)
        ax1.set_aspect('equal')
        ax1.set_title('Top-down view')
        map_2 = ax2.scatter(xs, zs, c=dens_arr)
        # ax2.set_ylim(ax2.get_xlim())
        ax2.set_title('Edge-on view')
    else:
        mappable = ax1.scatter(xs, ys, c=mass_arr)
        ax1.set_aspect('equal')
        ax1.set_title('Top-down view')
        map_2 = ax2.scatter(xs, zs, c=mass_arr)
        # ax2.set_ylim(ax2.get_xlim())
        ax2.set_title('Edge-on view')
    plt.colorbar(mappable, ax=ax1)
    plt.colorbar(map_2, ax=ax2)
    if mode == 'SHOW':
        plt.show()
    elif mode == 'RETURN':
        return fig
    else:
        savename = mode + '.pdf'
        plt.savefig(savename)


def calc_axis(m_wd, m_body, P, grav):
    m_tot = m_wd + m_body
    num = grav*m_tot*(P**2.0)
    denom = 4.0*(np.pi**2.0)
    return (num/denom)**(1.0/3.0)


def get_rad_bulk(mass_bulk, rho_bulk):
    vol = mass_bulk/rho_bulk
    rad_cubed = vol*3.0/(4.0*np.pi)
    rad_bulk = rad_cubed**(1.0/3.0)
    return rad_bulk


def get_rad_core(rad_bulk, frac_core):
    rad_core = rad_bulk * np.sqrt(frac_core)
    return rad_core


def get_rho_core(rho_bulk, fracs=[0.35, 0.65], k_rhos=[1.0, 0.25]):
    factors = np.array([fracs[i]*k_rhos[i] for i in range(0, len(fracs))])
    rho_core = rho_bulk/(np.sum(factors))
    return rho_core


def get_packing(packing_path):
    np.random.seed(seed)
    fnames = os.listdir(path)
    csv_names = [fname if fname.endswith('.csv') for fname in fnames]
    seed_names = np.array([csv_name if csv_name.startswith('seed')
                          for csv_name in csv_names])
    packing = pd.DataFrame(np.random.choice(seed_names))
    return packing


def scale_packing(packing, rad_bulk):
    x = packing.x
    y = packing.y
    z = packing.z
    r = packing.r
    x_max = np.max(x)
    y_max = np.max(y)
    z_max = np.max(z)
    x_min = np.abs(np.min(x))
    y_min = np.abs(np.min(y))
    z_min = np.abs(np.min(z))
    rad_now = np.median([x_max, x_min, y_max, y_min, z_max, z_min])
    rad_factor = rad_bulk/rad_now
    packing.x = rad_factor*packing.x
    packing.y = rad_factor*packing.y
    packing.z = rad_factor*packing.z
    packing.r = rad_factor*packing.r
    return packing


def get_rad_layers(fracs, rad_bulk):
    rad_high = np.zeros_like(fracs)
    rad_low = np.zeros_like(fracs)
    rad_high[-1] = rad_high
    for i in range(0, len(fracs) - 1):
        rad_high[i] = ((rad_bulk**3.0)*fracs[i] +
                       (rad_high[i]**3.0))**(1.0/3.0)
        rad_low[i+1] = rad_high[i]
    return rad_high, rad_low


def get_rad_wd(mass_wd=0.6):
    rad_wd = 0.0127*(mass_wd**(-1.0/3.0))*np.sqrt(1.0 -
                                                  0.607*(mass_wd**(4.0/3.0)))
    return rad_wd


def add_wd(sim, mass_wd=0.6):
    rad_wd = get_rad_wd(mass_wd)
    sim.units = {'m', 's', 'kg'}
    mass_wd = mass_wd*1.99e30
    rad_wd = rad_wd*6.957e8
    sim.add(m=mass_wd, r=rad_wd, hash='wd')
    return sim


def calc_axis(mass_wd, mass_bulk, P, grav):
    m_tot = mass_wd + mass_bulk
    num = grav*m_tot*(P**2.0)
    denom = 4.0*(np.pi**2.0)
    return (num/denom)**(1.0/3.0)


def add_scaled_packing(sim, packing, mass_bulk, rho_bulk,
                       fracs=[0.35, 0.65], k_rhos=[1.0, 0.25],
                       position=(0, 0, 0), velocity=(0, 0, 0)):
    rad_bulk = get_rad_bulk(mass_bulk, rho_bulk)
    rad_high, rad_low = get_rad_layers(fracs, rad_bulk)
    x_arr = packing.x
    y_arr = packing.y
    z_arr = packing.z
    rad_arr = packing.r
    pos_r = np.sqrt(x_arr**2.0 + y_arr**2.0 + z_arr**2.0)
    pos_x = x_arr + origin[0]
    pos_y = y_arr + origin[1]
    pos_z = z_arr + origin[2]
    vel_x = np.zeros_like(x_arr) + velocity[0]
    vel_y = np.zeros_like(y_arr) + velocity[1]
    vel_z = np.zeros_like(z_arr) + velocity[2]
    rho_core = get_rho_core(rho_bulk, fracs=fracs, k_rhos=k_rhos)
    for i in range(0, len(fracs)):
        r_low = rad_low[i]
        r_high = rad_high[i]
        rho_layer = k_rho[i]*rho_core
        mass_layer = rho_layer*((4.0*np.pi)/(3.0))*(np.median(rad_arr)**3.0)
        layer_mask = np.where((r_low < pos_r) & (pos_r < r_high))
        layer_x = pos_x[layer_mask]
        layer_y = pos_y[layer_mask]
        layer_z = pos_z[layer_mask]
        layer_r = rad_arr[layer_mask]
        layer_vx = vel_x[layer_mask]
        layer_vy = vel_y[layer_mask]
        layer_vz = vel_z[layer_mask]
        hash_names = ['l_%i_%i' % (i, j) for j in range(0, len(layer_x))]
        for q in range(0, len(layer_x)):
            sim.add(m=mass_layer[q], r=layer_r[q],
                    x=layer_x[q], y=layer_y[q], z=layer_z[q],
                    vx=layer_vx[q], vy=layer_vy[q], vz=layer_vz[q],
                    hash=hash_names[q])
    return sim


def add_pile_orbit(sim, mass_bulk, rho_bulk,
                   P=4.495, e=0.0, inc=0.0,
                   Omega=0.0, omega=0.0, f=0.0,
                   center_hash='wd',
                   fracs=[0.35, 0.65], k_rhos=[1.0, 0.25]):
    center_particle = sim.particles[center_hash]
    mass_center = center_particle.m
    mass_body = mass_bulk
    axis = calc_axis(mass_center, mass_body, sim.G)
    sim.add(a=axis, e=e, inc=inc, Omega=Omega, omega=omega, f=f, hash='test')
    position = (0.0, 0.0, 0.0)
    velocity = (0.0, 0.0, 0.0)
    position[0] = sim.particles['test'].x
    position[1] = sim.particles['test'].y
    position[2] = sim.particles['test'].z
    velocity[0] = sim.particles['test'].vx
    velocity[1] = sim.particles['test'].vy
    velocity[2] = sim.particles['test'].vz
    sim.remove(hash='test')
    packing = get_packing(packing_path)
    rad_bulk = get_rad_bulk(mass_bulk, rho_bulk)
    packing = scale_packing(packing, rad_bulk)
    return add_scaled_packing(sim, packing, mass_bulk, rho_bulk,
                              fracs, k_rhos, position, velocity)
