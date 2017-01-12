import rebound as reb
import reboundx as rebx
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

np.random.seed(1234)


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


def add_parent(sim, rho_bulk=3.2e3, m_parent=1.0e22, P=4.5):
    P = P*60.0*60.0  # Convert orbital period from hours to seconds
    a = calc_axis(m_wd=sim.particles['wd'].m, m_body=m_parent,
                  P=P, grav=sim.G)
    rad = rad_calc(m=m_parent, rho=rho_bulk)
    sim.add(m=m_parent, r=rad, x=a, vy=kep_calc(a=a,
                                                M=sim.particles['wd'].m,
                                                grav=sim.G), hash='parent')
    return sim


def add_N_frags(sim, N=6, rho_low=1.0e3, rho_high=3.2e3, rad_low=1.0e3,
                rad_high=5.0e3):
    base_pos = sim.particles['parent'].x
    base_rad = sim.particles['parent'].r
    base_vel = sim.particles['parent'].vy
    frag_hashes = ['frag_' + str(i) for i in range(1, N+1)]
    rad_arr = base_rad + np.random.uniform(low=rad_low, high=rad_high, size=N)
    dens_arr = np.random.uniform(low=rho_low, high=rho_high, size=N)
    mass_arr = dens_arr*((4.0/3.0)*np.pi*(rad_arr**3.0))
    u_arr = np.random.uniform(low=-1.0, high=1.0, size=N)
    v_arr = np.random.uniform(size=N)
    thetas = 2.0*np.pi*v_arr
    x_pos = base_pos + rad_arr*np.sqrt(1.0 - u_arr**2.0)*np.cos(thetas)
    y_pos = rad_arr*np.sqrt(1.0 - u_arr**2.0)*np.sin(thetas)
    z_pos = rad_arr*u_arr
    for j in range(0, N):
        sim.add(m=mass_arr[j], r=rad_arr[j] - base_rad, vy=base_vel,
                x=x_pos[j], y=y_pos[j], z=z_pos[j], hash=frag_hashes[j])
    print rad_arr - base_rad
    return sim


def fill_sphere(sim, rho_fill_fun=rho_const, rad_fill_func=rad_const,
                N_tot=3000, sphere_rad=1.0e6,
                rho_kwargs=[0.25], rad_kwargs=[100.0]):
    rad_parent = sim.particles['parent'].r
    m_parent = sim.particles['parent'].m
    dens_parent = m_parent/((4.0/3.0)*np.pi*(rad_parent**3.0))
    parent_x = sim.particles['parent'].x
    base_vel = sim.particles['parent'].vy
    rho_fill = rho_fill_func(rho_kwargs, rho_parent=dens_parent, N=N_tot)
    rad_fil = rad_const(rad_kwargs, N=N_tot)
    m_fill = (rad_fill**3.0)*(4.0/3.0)*np.pi*rho_fill
    u = np.random.uniform(low=-1.0, high=1.0, size=N_tot)
    theta = 2.0*np.pi*np.random.uniform(size=N_tot)
    rad = rad_parent+rad_fill
    x_pos = parent_x + rad*np.sqrt(1.0 - u**2.0)*np.cos(theta)
    y_pos = rad*np.sqrt(1.0 - u**2.0)*np.sin(theta)
    z_pos = rad*u
    for j in range(0, N_tot):
        sim.add(m=m_fill[j], r=rad_fill[j], vy=base_vel,
                x=x_pos[j], y=y_pos[j], z=z_pos[j])
    return sim

if __name__ == '__main__':
    sim_1 = add_wd(reb.Simulation())
    sim_1 = add_parent(sim_1)
    sim_1 = add_N_frags(sim_1)
    sim_1.status()
    print sim_1.particles['wd']
    print sim_1.particles['parent']
    for i in range(0, sim_1.N):
        print sim_1.particles[i].r
