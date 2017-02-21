import numpy as np
import rebound
import os
import matplotlib.pyplot as plt
import pandas as pd

packing_path = str('../../../3k_packings_dir/')

save_path = str('.')


def plot_bisection(rebsim, color_mass=False, color_dens=True, mode='SHOW',
                   inflate=1.0):
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
    bisection_mask = np.where(zs < np.median(zs))
    ys = ys[bisection_mask]
    zs = zs[bisection_mask]
    mass_arr = mass_arr[bisection_mask]
    rad_arr = rad_arr[bisection_mask]
    xs = xs[bisection_mask]
    if color_dens:
        dens_arr = mass_arr/((4.0/3.0)*np.pi*(rad_arr**3.0))
        mappable = plt.scatter(xs, ys, c=dens_arr, s=inflate*rad_arr)
    else:
        mappable = plt.scatter(xs, ys, c=mass_arr, s=inflate*rad_arr)
    plt.colorbar()
    if mode == 'SHOW':
        plt.show()
    elif mode == 'RETURN':
        return fig
    else:
        savename = mode + '.pdf'
        plt.savefig(savename)
    plt.clf()


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
    fnames = os.listdir(packing_path)
    csv_names = [fname for fname in fnames if fname.endswith('.csv')]
    seed_names = np.array([csv_name for csv_name in csv_names
                           if csv_name.startswith('3k_seed')])
    packing_name = packing_path + '/' + np.random.choice(seed_names)
    packing = pd.read_csv(packing_name)
    return packing


def scale_packing(packing, rad_bulk):
    x = packing.x
    y = packing.y
    z = packing.z
    r = packing.r
    pos_r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    rad_now = np.max(pos_r)
    rad_factor = rad_bulk/rad_now
    packing.x = rad_factor*packing.x
    packing.y = rad_factor*packing.y
    packing.z = rad_factor*packing.z
    packing.r = rad_factor*packing.r
    return packing


def get_rad_layers(fracs, rad_bulk):
    rad_high = np.zeros_like(fracs)
    rad_low = np.zeros_like(fracs)
    rad_high[-1] = rad_bulk
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
                       position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0],
                       n_frags=0, frag_mass_bounds=[4.35e20, 1.45e20],
                       frag_rho_bounds=[5.0e3, 2.0e3], body=1,
                       factor=0.0031395):
    rad_bulk = get_rad_bulk(mass_bulk, rho_bulk)
    rad_high, rad_low = get_rad_layers(fracs, rad_bulk)
    x_arr = packing.x
    y_arr = packing.y
    z_arr = packing.z
    rad_arr = np.array(packing.r)
    if n_frags != 0:
        frag_indices = np.random.choice(len(rad_arr), n_frags)
        frag_rhos = np.random.uniform(low=frag_rho_bounds[1],
                                      high=frag_rho_bounds[0],
                                      size=n_frags)
        frag_masses = np.random.uniform(low=frag_mass_bounds[1],
                                        high=frag_mass_bounds[0],
                                        size=n_frags)
        frag_rads = get_rad_bulk(mass_bulk=frag_masses, rho_bulk=frag_rhos)
    pos_r = np.sqrt(x_arr**2.0 + y_arr**2.0 + z_arr**2.0)
    pos_x = np.array(x_arr) + position[0]
    pos_y = np.array(y_arr) + position[1]
    pos_z = np.array(z_arr) + position[2]
    vel_x = np.zeros_like(x_arr) + velocity[0]
    vel_y = np.zeros_like(y_arr) + velocity[1]
    vel_z = np.zeros_like(z_arr) + velocity[2]
    rho_core = get_rho_core(rho_bulk, fracs=fracs, k_rhos=k_rhos)
    m_tot = 0.0
    count = 0
    for i in range(0, len(fracs)):
        r_low = rad_low[i]
        r_high = rad_high[i]
        rho_layer = k_rhos[i]*rho_core
        mass_layer = rho_layer*((4.0*np.pi)/(3.0))*(np.median(rad_arr)**3.0)
        mass_layer = mass_layer/factor
        layer_mask = np.where((r_low <= pos_r) & (pos_r <= r_high))[0]
        layer_x = pos_x[layer_mask]
        layer_y = pos_y[layer_mask]
        layer_z = pos_z[layer_mask]
        layer_r = rad_arr[layer_mask]
        layer_vx = vel_x[layer_mask]
        layer_vy = vel_y[layer_mask]
        layer_vz = vel_z[layer_mask]
        hash_names = ['%i_l_%i_%i' % (body, i, j) for j in range(0,
                                                                 len(layer_x))]
        for q in range(0, len(layer_x)):
            count = count + 1
            if n_frags != 0:
                if (count-1) in frag_indices:
                    loc = np.where(frag_indices == (count - 1))
                    sim.add(m=frag_masses[loc], r=frag_rads[loc],
                            x=layer_x[q], y=layer_y[q], z=layer_z[q],
                            vx=layer_vx[q], vy=layer_vy[q], vz=layer_vz[q],
                            hash=hash_names[q])
                    m_tot += frag_masses[loc]
                else:
                    sim.add(m=mass_layer, r=layer_r[q],
                            x=layer_x[q], y=layer_y[q], z=layer_z[q],
                            vx=layer_vx[q], vy=layer_vy[q], vz=layer_vz[q],
                            hash=hash_names[q])
                    m_tot += mass_layer
            else:
                sim.add(m=mass_layer, r=layer_r[q],
                        x=layer_x[q], y=layer_y[q], z=layer_z[q],
                        vx=layer_vx[q], vy=layer_vy[q], vz=layer_vz[q],
                        hash=hash_names[q])
                m_tot += mass_layer

    print 'Ratio is:', m_tot/(rho_bulk*(4.0*np.pi/3.0)*(rad_bulk**3.0))
    return sim


def add_pile_orbit(sim, mass_bulk, rho_bulk,
                   P=4.495, e=0.0, inc=0.0,
                   Omega=0.0, omega=0.0, f=0.0,
                   fracs=[0.35, 0.65], k_rhos=[1.0, 0.25],
                   n_frags=0, frag_mass_bounds=[4.35e20, 1.45e20],
                   frag_rho_bounds=[5.0e3, 2.0e3], body=1,
                   factor=0.0031395):
    center_particle = sim.particles['wd']
    mass_center = center_particle.m
    mass_body = mass_bulk
    axis = calc_axis(mass_center, mass_body, P*3600.0, sim.G)
    sim.add(a=axis, e=e, inc=inc, Omega=Omega, omega=omega, f=f, hash='test')
    position = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    position[0] = sim.particles['test'].x
    position[1] = sim.particles['test'].y
    position[2] = sim.particles['test'].z
    velocity[0] = sim.particles['test'].vx
    velocity[1] = sim.particles['test'].vy
    velocity[2] = sim.particles['test'].vz
    packing = get_packing(packing_path)
    rad_bulk = get_rad_bulk(mass_bulk, rho_bulk)
    packing = scale_packing(packing, rad_bulk)
    return add_scaled_packing(sim, packing, mass_bulk, rho_bulk,
                              fracs, k_rhos, position, velocity,
                              n_frags=n_frags,
                              frag_mass_bounds=frag_mass_bounds,
                              frag_rho_bounds=frag_rho_bounds,
                              body=body,
                              factor=factor)


def setup_simulation(dt=10.0, integrator='WHFAST', collision='direct',
                     collision_resolve='hardsphere',
                     archive_name='archive.bin', interval=2.7e6,
                     mass_wd=0.6, N_piles=1, pile_mass_bulks=[2.9e22],
                     pile_rho_bulks=[3.8e3],
                     pile_orbits=[[4.495, 0.0, 0.0, 0.0, 0.0, 0.0]],
                     pile_fracs=[[0.35, 0.65]],
                     pile_k_rhos=[[1.0, 0.25]],
                     n_frags=[0],
                     frag_mass_bounds=[[4.35e20], [1.45e20]],
                     frag_rho_bounds=[[5.0e3], [2.0e3]],
                     save_loc=save_path,
                     seed=0.0,
                     factor=[0.0031395]):

    np.random.seed(seed)
    sim = rebound.Simulation()
    sim = add_wd(sim, mass_wd)
    for i in range(0, N_piles):
        sim = add_pile_orbit(sim, pile_mass_bulks[i], pile_rho_bulks[i],
                             P=pile_orbits[i][0], e=pile_orbits[i][1],
                             inc=pile_orbits[i][2], Omega=pile_orbits[i][3],
                             omega=pile_orbits[i][4], f=pile_orbits[i][5],
                             fracs=pile_fracs[i], k_rhos=pile_k_rhos[i],
                             n_frags=n_frags[i],
                             frag_mass_bounds=[frag_mass_bounds[0][i],
                                               frag_mass_bounds[1][i]],
                             frag_rho_bounds=[frag_rho_bounds[0][i],
                                              frag_rho_bounds[1][i]],
                             body=i,
                             factor=factor[i])

    save_name = save_loc + '/' + archive_name
    sim.dt = dt
    sim.integrator = integrator
    sim.collision = collision
    sim.collision_resolve = collision_resolve
    sim.initSimulationArchive(archive_name, interval=interval)
    # sim.ri_whfast.corrector = 11
    sim.save(archive_name)
    return sim


def integrate_csv(sim, t_integrate, P_output=2.628e6, P_light=2.16e4):
    dt = sim.dt
    times = np.arange(sim.t, t_integrate, dt)
    for time in times:
        sim.integrate(time, exact_finish_time=0)
        P_output = 2.628e6
        P_light = 2.16e4
        test = sim.t/P_output
        floor_test = np.floor(test)
        temp_test = test - floor_test
        ratio = P_light/P_output
        print "Time is: %.6f days\n" % (sim.t/86400.0)
        if temp_test < ratio:
            temp_name = int(floor_test)
            save_name = (sim.t - floor_test*P_output)/dt
            f_name = "%i_%g.csv" % (temp_name, save_name)
            fp = open(f_name, "w+")
            fp.write("Hash, x, y, z\n")
            for i in range(0, sim.N):
                fp.write("%s, %.16f, %.16f, %.16f\n" %
                         (sim.particles[i].hash, sim.particles[i].x,
                          sim.particles[i].y,
                          sim.particles[i].z))
            fp.close()


def parse_params(param_file):
    f = open(param_file)
    archive_name = f.readline()[15:-1] + '.bin'
    mass_wd = float(f.readline()[10:-1])
    n_piles = int(f.readline()[10:-1])
    seed = int(float(f.readline()[7:-1]))
    pile_masses = np.empty((n_piles))
    pile_rho_bulks = np.empty_like(pile_masses)
    pile_orbits = np.empty((n_piles, 6))
    pile_layer_fracs = [np.empty((len(pile_masses), 2), dtype=list)]
    pile_layer_k_rhos = [np.empty((len(pile_masses), 2), dtype=list)]
    pile_factors = np.empty((n_piles))
    pile_n_frags = np.empty((n_piles), dtype=int)
    frag_mass_highs = np.empty_like(pile_masses)
    frag_mass_lows = np.empty_like(pile_masses)
    frag_rho_highs = np.empty_like(pile_masses)
    frag_rho_lows = np.empty_like(pile_masses)
    for i in range(0, n_piles):
        pile_masses[i] = float(f.readline()[14:-1])
        pile_rho_bulks[i] = float(f.readline()[18:-1])
        for j in range(0, 6):
            pile_orbits[i, j] = float(f.readline()[17:-1])
        pile_layer_num = int(f.readline()[16:-1])
        pile_layer_fracs[i] = [float(f.readline()[22:-1])
                               for k in np.arange(0, pile_layer_num)]
        pile_layer_k_rhos[i] = [float(f.readline()[23:-1])
                                for k in np.arange(0, pile_layer_num)]
        pile_layer_fracs = pile_layer_fracs[0:n_piles]
        pile_layer_k_rhos = pile_layer_k_rhos[0:n_piles]
        pile_factors[i] = float(f.readline()[9:-1])
        pile_n_frags[i] = int(f.readline()[10:-1])
        frag_mass_highs[i] = float(f.readline()[16:-1])
        frag_mass_lows[i] = float(f.readline()[15:-1])
        frag_rho_highs[i] = float(f.readline()[15:-1])
        frag_rho_lows[i] = float(f.readline()[14:-1])
    return [archive_name, mass_wd,
            seed, n_piles,
            pile_masses, pile_rho_bulks,
            pile_orbits, pile_layer_fracs,
            pile_layer_k_rhos, pile_n_frags,
            [frag_mass_highs, frag_mass_lows],
            [frag_rho_highs, frag_rho_lows],
            pile_factors]


def run_params(params):
    sim = setup_simulation(archive_name=params[0], mass_wd=params[1],
                           seed=params[2], N_piles=params[3],
                           pile_mass_bulks=params[4], pile_rho_bulks=params[5],
                           pile_orbits=params[6], pile_fracs=params[7],
                           pile_k_rhos=params[8], n_frags=params[9],
                           frag_mass_bounds=params[10],
                           frag_rho_bounds=params[11],
                           factor=params[12])
    return sim


if __name__ == '__main__':
    # sim = setup_simulation(pile_fracs=[[0.35, 0.65]],
    #                       pile_k_rhos=[[1.0, 0.25]])
    # plt.set_cmap('Set1')
    # plot_2D_simulation(sim, inflate=0.0025, mode='SHOW', wd_status='HIDE')
    params = parse_params('earth.txt')
    for i in range(0, len(params)):
        print params[i]
    sim = run_params(params)
    # plot_2D_simulation(sim, inflate=0.0025, mode='SHOW', wd_status='HIDE')
