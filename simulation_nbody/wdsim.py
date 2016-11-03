import rebound
import numpy as np
import matplotlib.pyplot as plt
import assignpos as pos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getstarparams(mstar, rstar):
    '''
    Gives the parameters for a star required for a rebound.Simulation

    Input Parameters
    ----------------

    mstar       : Mass of the star in Simulation units
                : float

    rstar       : Radius of the star in Simulation units
                : float

    Star Parameters
    -------------------
    mstar        : Above
                : float

    x,y,z       : Coordinates of star in Simulation units
                : float

    vx, vy, vz  : Velocity components of particle in Simulation units
                : float

    r           : Radius of particle in Simulation units

    id          : Integer ID of Star = 0
                : int
    '''

    return [mstar, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rstar, 0]

def shellpile(radius, nparticles, mpile, startpos = [0.0, 0.0, 0.0],
    startvel = [0.0, 0.0, 0.0], shellrad = np.asarray([0.0]),
    mshells = np.asarray([1.0])):
    '''
    Gives the parameters required for a collection of particles in a rubble
    pile structured as a series of concentric shells. All particles are
    assigned a fixed radius r, given by:
    r = cuberoot(1/(N/0.735)/(R**3.0)) where N = nparticles, R = radius.
    0.735 is given by Kepler's conjecture. The final number of particles
    will only approximately equal the given nparticles with the error
    increasing with the number of shells. Each shell is hexagonally packed.

    Input Parameters
    ----------------
    radius      : R, the total radius of the rubble pile in Simulation units
                : float

    nparticles  : Approximate number of total particles for the rubble pile
                : int

    mpile       : M, the approximate total mass of the rubble pile
                    in Simulation units
                : float

    startpos    :  X,Y,Z co-ordinates of center of rubble pile.
                    Default assumes rubble pile is in rest frame.
                : float numpy.ndarray

    startvel    :  X,Y,Z velocity components of center of rubble pile.
                    Default assumes rubble pile is in rest frame.
                : float numpy.ndarray

    shellrad    : An array of inner radii for each shell.
                    (units of rubble pile radii)
                    Default assumes no shells, so the inner radius is 0.0
                : float numpy.ndarray

    mshells     : An array of masses of particles within a shell.
                    (units of mean mass per particle, calculated separately)
                    Default assumes no shells, so the mass is 1.0
                : float numpy.ndarray
    '''
    mparticle = mpile/nparticles
    rparticle = (0.735*(radius**3.0)/nparticles)**(1.0/3.0)
    rinners = shellrad*radius
    if len(rinners) > 1:
        routers = np.append(rinners[1:], [radius])
    else:
        routers = [radius]

    idcount = 1 #assumes star is id = 0
    particles = np.empty((9,nparticles))
    count = 1

    for i in range(0, len(shellrad)):
        shellcoords = pos.makeshell(rinners[i], routers[i], rparticle)
        for j in range(0,len(shellcoords[0])):
            particles[:,i+j] = [mshells[i]*mparticle, shellcoords[0][j]+startpos[0],
                shellcoords[1][j]+startpos[1], shellcoords[2][j]+startpos[2],
                startvel[0], startvel[1], startvel[2], rparticle,
                int(i + j + idcount)]
            count+=1

    particles = particles[:,:count]

    '''
    Output Parameters
    -----------------
    particles   : An array of the 8 parameters required for a rebound.particle
                : numpy.ndarray

    Particle Parameters
    -------------------
    mass        : Mass of particle in Simulation units
                : float

    x,y,z       : Coordinates of particle in Simulation units
                : float

    vx, vy, vz  : Velocity components of particle in Simulation units
                : float

    r           : Radius of particle in Simulation units
                : float

    id          : Integer ID of particle, starting from 1. Assumes Star ID = 0
                : int
    '''

    return particles


def setupSimulation(mstar, rstar, rpile, nparticles, mpile, period, dt = 1.0):
    sim = rebound.Simulation()
    sim.units = {'s', 'km', 'kg'}

    starparams = getstarparams(mstar, rstar)
    sim.add(m=starparams[0], x = starparams[1], y = starparams[1],
        z = starparams[2], vx = starparams[3], vy = starparams[4],
        vz = starparams[5], r = starparams[6], id = 0)


    axis = ((sim.G*(mstar + mpile)*(period**2.0))/(4.0*(np.pi**2.0)))**(1.0/3.0)
    kepvel = np.sqrt(sim.G*(mstar + mpile)/axis)

    particles= shellpile(rpile, nparticles, mpile)
    for q in range(0,len(particles[0])):
        particle = particles[:,q]
        sim.add(m=particle[0], x = particle[1] + axis, y = particle[2],
            z = particle[3], vx = particle[4], vy = particle[5] + kepvel,
            vz = particle[6], r = particle[7], id = int(particle[8]))
    sim.move_to_com()
    sim.dt = dt
    sim.integrator = 'whfast'
    sim.integrator_whfast_safemode = 0
    return sim

def savesim(fname):
    sim.save(fname)
    sim.status()


def restoresim(fname):
    sim = rebound.Simulation.from_file(fname)
    sim.status()
    return sim

if __name__ == '__main__':
    msun = 2.0*(10.0**30.0)
    sim = setupSimulation(0.6*msun, 8547.0, 225.0, 100, 10.0**22.0, 4.4989*3600.0)
    print sim.N
    #sim = restoresim('checkpoint1000.bin')
    Noutputs = 2.0

    xs = np.empty((sim.N,Noutputs/10))
    ys = np.empty((sim.N,Noutputs/10))
    zs = np.empty((sim.N,Noutputs/10))
    allradius = np.empty((sim.N, Noutputs/10))
    count = 0
    times = np.linspace(0.0, 10000000.0*sim.dt, Noutputs)
    print times
    rparticle = (np.append([8547.0], [sim.particles[j].r for j in range(1,sim.N)]))**2.0
    for i, time in enumerate(times):
        count +=1
        #print sim.particles[0]
        print sim.particles[60].x - sim.particles[10].x
        sim.integrate(time)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    print sim.particles[0]
    ax.scatter([sim.particles[j].x for j in range(0,sim.N)],
        [sim.particles[j].y for j in range(0,sim.N)],
        [sim.particles[j].z for j in range(0,sim.N)],
        [(sim.particles[j].r)**2.0 for j in range(0,sim.N)])
    plt.savefig('longorbitplot2.pdf')
    sim = setupSimulation(0.6*msun, 8547.0, 200.0, 100, 10.0**20.0, 4.4989*3600.0)
    print sim.N
    for i, time in enumerate(times):
        count +=1
            #print sim.particles[0]
        print sim.particles[60].x - sim.particles[10].x
        sim.integrate(time)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    print sim.particles[0]
    ax.scatter([sim.particles[j].x for j in range(0,sim.N)],
        [sim.particles[j].y for j in range(0,sim.N)],
        [sim.particles[j].z for j in range(0,sim.N)],
        [(sim.particles[j].r)**2.0 for j in range(0,sim.N)])
    plt.savefig('longorbitplot.pdf')
    #savesim('checkpoint1000.bin')
