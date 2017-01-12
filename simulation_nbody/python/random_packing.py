import rebound as reb
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

np.random.seed(1234)


def r_const(r_kwargs=[1.0]):
    """
    Args:
        r_kwargs (list): Contains the value for particle radius
    Returns:
        A value for the particle radius, given a constant input
    """
    return r_kwargs[0]


def m_const(m_kwargs=[1.0], r_value=1.0):
    """
    Args:
        m_kwargs (list): A list containing the value for the particle mass

        r_value (float): Radius of particle, given for consistency with other
                         mass functions that may depend on radius
    Returns:
        A value for the mass of a particle, given a constant input
    """
    return m_kwargs[0]


def convert_sphere_xyz((r, theta, phi), origin=(0, 0, 0)):
    """
    Args:
        (r, theta, phi) (tuple): A tuple containing the spherical
                                to be transformed to cartesian
                                (elements can be arrays or lists)
        origin (tuple): A tuple containing the cartesian coordinates
                        of the origin to be added to each of the
                        transformed coordinates.
    Returns:
        (x,y,z) (tuple): A tuple containing the transformed inputs
    """
    x = r*np.sin(theta)*np.cos(phi) + origin[0]
    y = r*np.sin(theta)*np.cos(phi) + origin[1]
    z = r*np.cos(theta) + origin[2]
    return (x, y, z)


def convert_xyz_sphere((x, y, z), origin=(0, 0, 0)):
    """
    Args:
        (x, y, z) (tuple): A tuple containing the cartesian coordinates
                            to be transformed to spherical coordinates

        origin (tuple): A tuple containing the spherical coordinates of
                        the origin to be added to each of the
                        transformed coordinates
    Returns:
        sphere_coords (tuple): A tuple containing the transformed inputs
    """
    r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta = np.arctan(y/x)
    phi = np.arccos(z/r)
    return (r, theta, phi)


def N_random_coords_shell(N_tot, (r_min, r_max)=(0.0, 1.0)):
    """
    Args:
        N_tot (int): The estimated total number of particles to be added
                     to the shell.

        r_min (float): The minimum radius of the shell to be packed.
                       All coordinates are drawn from a random uniform
                       distribution between min,max values provided.

        r_max (float): The maximum radius

        theta_min (float): The minimum theta

        theta_max (float): The maximum theta

        phi_min (float): The minimum phi

        phi_max (float): the maximum phi

    Returns:
        N_tot (int): The total number of particles in the shell

        np.asarray([rs, thetas, phis]) (np.array): The spherical coords
                                                   of N_tot particles in the
                                                   shell.

        np.asarray([xs,ys,zs]) (np.array): The cartesian coords of N_tot
                                           particles in the shell.
    """
    V_shell = (4.0/3.0)*np.pi*(r_max**3.0 - r_min**3.0)
    rho_shell = 1.3*N_tot/V_shell
    N_est = int(rho_shell*((2.0*r_max)**3.0))
    xs = np.random.uniform(low=-1.0*r_max, high=r_max, size=N_est)
    ys = np.random.uniform(low=-1.0*r_max, high=r_max, size=N_est)
    zs = np.random.uniform(low=-1.0*r_max, high=r_max, size=N_est)
    rads = np.sqrt(xs**2.0 + ys**2.0 + zs**2.0)
    cut_mask = np.where((rads < r_max) & (rads >= r_min))
    xs_cut = xs[cut_mask]
    ys_cut = ys[cut_mask]
    zs_cut = zs[cut_mask]
    N_tot = len(xs_cut)
    return N_tot, convert_xyz_sphere(np.asarray([xs_cut, ys_cut, zs_cut])), np.asarray([xs_cut, ys_cut, zs_cut])


def M_limit_sphere(M_tot, sphere_rad, origin=(0, 0, 0),
                   m_func=m_const, r_func=r_const,
                   m_kwargs=[1.0], r_kwargs=[1.0], N_est=10000):
    M_count = 0.0
    N_count = 0
    x_coords = np.empty((N_est))
    y_coords = np.empty_like(x_coords)
    z_coords = np.empty_like(x_coords)
    m_list = np.empty_like(x_coords)
    r_list = np.empty_like(x_coords)
    while M_count < M_tot:
        if N_count < N_est:
            p_coords = random_sphere_coords(sphere_rad, origin=(0, 0, 0))
            x_coords[N_count] = p_coords[0]
            y_coords[N_count] = p_coords[1]
            z_coords[N_count] = p_coords[2]
            r_list[N_count] = r_func(r_kwargs)
            m_list[N_count] = m_func(m_kwargs, r_list[N_count])
            M_tot += m_list[N_count]
            N_count += 1
        else:
            x_coords = np.append(x_coords, np.empty((int(0.5*N_est))))
            y_coords = np.append(y_coords, np.empty((int(0.5*N_est))))
            z_coords = np.append(z_coords, np.empty((int(0.5*N_est))))
            m_list = np.append(m_list, np.empty((int(0.5*N_est))))
            r_list = np.append(r_list, np.empty((int(0.5*N_est))))
    cart_coords = np.asarray([x_coords[:N_count],
                             y_coords[:N_count], z_coords[N_count]])
    return cart_coords, r_list[:N_count], m_list[:N_count]


class setup_particle(object):
    def __init__(self, rebsim, rad=(1.0), mass=(1.0),
                 position=(0, 0, 0), vels=(0, 0, 0)):
        self.rad = rad
        self.mass = mass
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.vx = vels[0]
        self.vy = vels[1]
        self.vz = vels[2]
        self.rebsim = rebsim

    def add_to_sim_cartesian(self):
        self.rebsim.add(m=self.mass,
                        x=self.x, y=self.y, z=self.z,
                        vx=self.vx, vy=self.vy, vz=self.vz,
                        r=self.rad)


class setup_sphere(object):
    def __init__(self, rebsim, sphere_rad,
                 position=(0, 0, 0), vels=(0, 0, 0)):
        self.sphere_rad = sphere_rad
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.vx = vels[0]
        self.vy = vels[1]
        self.vz = vels[2]
        self.rebsim = rebsim

    def random_equal_sphere_N(self, N_tot, particle_rad, particle_m):
        sphere_rad = self.sphere_rad
        N_tot, sphere_coords, cart_coords = N_random_coords_shell(N_tot,
                                                                  (0.0,
                                                                   sphere_rad))
        cart_coords[0] = cart_coords[0] + self.x
        cart_coords[1] = cart_coords[1] + self.y
        cart_coords[2] = cart_coords[2] + self.z
        self.N_particles = N_tot
        for i in range(0, N_tot):
            p1 = setup_particle(rebsim=self.rebsim, rad=particle_rad,
                                mass=particle_m,
                                position=(cart_coords[0][i],
                                          cart_coords[1][i],
                                          cart_coords[2][i]),
                                vels=(self.vx, self.vy, self.vz))
            p1.add_to_sim_cartesian()


def plot_3D_simulation(rebsim, color_mass=True, color_dens=False, mode='SHOW'):
    ax = plt.subplot(111, projection='3d')
    N_tot = rebsim.N
    particle_arr = rebsim.particles
    xpos_arr = np.empty((N_tot))
    ypos_arr = np.empty_like(xpos_arr)
    zpos_arr = np.empty_like(xpos_arr)
    rad_arr = np.empty_like(xpos_arr)
    if any([color_mass, color_dens]):
        mass_arr = np.empty_like(xpos_arr)
    else:
        pass
    for i in range(0, N_tot):
        particle = particle_arr[i]
        xpos_arr[i] = particle.x
        ypos_arr[i] = particle.y
        zpos_arr[i] = particle.z
        rad_arr[i] = particle.r
        if any([color_mass, color_dens]):
            mass_arr[i] = particle.m
    if color_dens:
        dens_arr = mass_arr/(4.0*np.pi*(rad_arr**3.0))
        plot_fig = ax.scatter(x=xpos_arr, y=ypos_arr, z=zpos_arr,
                              c=dens_arr, s=(rad_arr**2.0))
    else:
        plot_fig = ax.scatter(xpos_arr, ypos_arr, zpos_arr, s=(rad_arr**3.0),
                              c=mass_arr)
    plt.colorbar(plot_fig)
    if mode == 'SHOW':
        plt.show()
    elif mode == 'RETURN':
        return fig
    else:
        savename = mode + '.pdf'
        plt.savefig(savename)

if __name__ == '__main__':
    test_sim = reb.Simulation()
    test_sim.integrator = "hermes"
    test_sim.G = 6.674e-11  # SI units
    test_sim.units = ('s', 'm', 'kg')
    test_sim.dt = 1.0
    test_sphere = setup_sphere(test_sim, 10.0)
    test_sphere.random_equal_sphere_N(1000, 2.5, 2.5)
    test_sim.add(m=30.0, r=5.0, x=100.0, y=0.0, z=0.0, vy=0.004)
    test_sim.move_to_com()
    test_sim.collision = "direct"
    test_sim.collision_resolve_keep_sorted = 1
    test_sim.track_energy_offset = 1
    t_max = 1e1
    test_sim.N_active = test_sim.N
    E0 = test_sim.calculate_energy()
    print test_sim.N
    # plot_3D_simulation(test_sim, mode='SHOW')
    test_sim.integrate(t_max)
    dE = abs(test_sim.calculate_energy() - E0)/E0
    print E0, dE
    plot_3D_simulation(test_sim, mode='SHOW')
    print test_sim.N
