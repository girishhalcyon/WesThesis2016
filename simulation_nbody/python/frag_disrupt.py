import rebound as reb
import reboundx as rebx
import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
#from random_packing import plot_3D_simulation, plot_2D_simulation
from random_packing import plot_2D_simulation
# np.random.seed(1234)


def rad_const(rad_kwargs=[1.0]):
    """
    Args:
        r_kwargs (list): Contains the value for particle radius
    Returns:
        A value for the particle radius, given a constant input
    """
    return rad_kwargs[0]


def rho_const(rho_kwargs, rho_parent, N):
    return np.asarray([rho_kwargs[0]*rho_parent for i in range(0, N)])


def rad_calc(m, rho):
    vol = m/rho
    return ((3.0*vol)/(4.0*np.pi))**(1.0/3.0)


def calc_axis(m_wd, m_body, P, grav):
    m_tot = m_wd + m_body
    num = grav*m_tot*(P**2.0)
    denom = 4.0*(np.pi**2.0)
    return (num/denom)**(1.0/3.0)


def kep_calc(a, M, grav):
    return np.sqrt(grav*M/a)


def add_wd(sim, m_wd=0.6, r_wd=0.02):
    sim.units = {'m', 's', 'kg'}
    m_wd = m_wd*1.99e30
    r_wd = r_wd*6.957e8
    sim.add(m=m_wd, r=r_wd, hash='wd')
    return sim


def add_parent(sim, rho_bulk=6.2e3, m_parent=0.35*1.0e22, P=4.5):
    P = P*60.0*60.0  # Convert orbital period from hours to seconds
    a = calc_axis(m_wd=sim.particles['wd'].m, m_body=m_parent,
                  P=P, grav=sim.G)
    rad = rad_calc(m=m_parent, rho=rho_bulk)
    sim.add(m=m_parent, r=rad, a=a, hash='parent')
    return sim


def add_N_frags(sim, N=6, rho_low=0.1, rho_high=1.0, mass_low=0.1,
                mass_high=0.3):
    base_pos_x = sim.particles['parent'].x
    base_pos_y = sim.particles['parent'].y
    base_pos_z = sim.particles['parent'].z
    base_rad = sim.particles['parent'].r
    base_vel_x = sim.particles['parent'].vx
    base_vel_y = sim.particles['parent'].vy
    base_vel_z = sim.particles['parent'].vz
    base_mass = sim.particles['parent'].m
    base_dens = base_mass/((4.0*np.pi/3.0)*(base_rad**3.0))
    frag_hashes = ['frag_' + str(i) for i in range(1, N+1)]
    mass_arr = np.random.uniform(low=mass_low*base_mass,
                                 high=mass_high*base_mass, size=N)
    dens_arr = np.random.uniform(low=rho_low*base_dens,
                                 high=rho_high*base_dens, size=N)
    frag_rads = rad_calc(mass_arr, dens_arr)
    rad_arr = base_rad
    u_arr = np.random.uniform(low=-1.0, high=1.0, size=N)
    v_arr = np.random.uniform(size=N)
    thetas = 2.0*np.pi*v_arr
    x_pos = base_pos_x + rad_arr*np.sqrt(1.0 - u_arr**2.0)*np.cos(thetas)
    y_pos = base_pos_y + rad_arr*np.sqrt(1.0 - u_arr**2.0)*np.sin(thetas)
    z_pos = base_pos_z + rad_arr*u_arr
    for j in range(0, N):
        sim.add(m=mass_arr[j], r=frag_rads[j],
                x=x_pos[j], y=y_pos[j], z=z_pos[j], hash=frag_hashes[j],
                vx=base_vel_x, vy=base_vel_y, vz=base_vel_z)
    return sim


def fill_sphere(sim, rho_fill_func=rho_const, rad_fill_func=rad_const,
                N_tot=3000, sphere_mass=1.0e22, rho_bulk=3.2e3,
                rho_kwargs=[0.25], rad_kwargs=[100.0]):
    rad_parent = sim.particles['parent'].r
    m_parent = sim.particles['parent'].m
    dens_parent = m_parent/((4.0/3.0)*np.pi*(rad_parent**3.0))
    parent_x = sim.particles['parent'].x
    parent_y = sim.particles['parent'].y
    parent_z = sim.particles['parent'].z
    base_vel_x = sim.particles['parent'].vx
    base_vel_y = sim.particles['parent'].vy
    base_vel_z = sim.particles['parent'].vz
    m_tot = np.sum(np.array([sim.particles[i].m for i in range(0, sim.N)]))
    m_tot = m_tot - sim.particles['wd'].m
    m_particle = (sphere_mass - m_tot)/N_tot
    rho_fill = rho_fill_func(rho_kwargs, rho_parent=dens_parent, N=N_tot)
    m_fill = np.ones((N_tot))*m_particle  # fix later with proper function
    rad_fill = rad_calc(m_fill, rho_fill)
    u = np.random.uniform(low=-1.0, high=1.0, size=N_tot)
    theta = 2.0*np.pi*np.random.uniform(size=N_tot)
    rad = rad_parent+np.random.uniform(low=0.0, high=rad_fill, size=N_tot)
    x_pos = parent_x + rad*np.sqrt(1.0 - u**2.0)*np.cos(theta)
    y_pos = parent_y + rad*np.sqrt(1.0 - u**2.0)*np.sin(theta)
    z_pos = parent_z + rad*u
    for j in range(0, N_tot):
        temp_hash = "fill_" + str(j)
        sim.add(m=m_fill[j], r=rad_fill[j],
                x=x_pos[j], y=y_pos[j], z=z_pos[j],
                vx=base_vel_x, vy=base_vel_y, vz=base_vel_z, hash=temp_hash)
    return sim


def cor_bridges(r, v):
        return 0.1


if __name__ == '__main__':
    sim = add_wd(reb.Simulation())
    sim = add_parent(sim)
    # plot_2D_simulation(sim)
    print sim.particles['parent'].r
    sim = add_N_frags(sim, N=6)
    sim.N_active = sim.N
    # plot_2D_simulation(sim)
    sim.status()
    sim = fill_sphere(sim, N_tot=3000)
    sim.status()
    for particle in sim.particles:
        print particle.hash
    # plot_2D_simulation(sim)
    sim.dt = 10.0
    sim.integrator = 'IAS15'
    sim.collision = 'direct'
    sim.collision_resolve = "hardsphere"
    sim.move_to_com()
    sim.initSimulationArchive("archive.bin", interval=1.e1)
    sim.integrate(0.0)
    # sim.coefficient_of_restitution = cor_bridges
    #plot_2D_simulation(sim)
    # sim.coefficient_of_restitution = cor_bridges
    # sim.status()
    # timestep = 24.0*3600.0
    # for o in sim.calculate_orbits():
        # print(o)
    # for i in range(0, int(2.0*365.25)):
        # fig = reb.OrbitPlot(sim)
        # if i < 10:
            # num = '00' + str(i)
        # elif i < 100:
            # num = '0' + str(i)
        # else:
            # num = str(i)
        # savename = 'frag_disrupt_plots_2/' + num + '_days.pdf'
        # fig.savefig(savename)
        # plt.clf()
        # sim.status()
        # sim.integrate(timestep*i)
        # print sim.t/86400.0
    # sim.status()
    # for o in sim.calculate_orbits():
        # print (o)
