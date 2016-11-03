import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def makecube(side, radius, gap = 0.0):
    xcoords = np.empty(((10.0*side/radius)**3.0))
    ycoords = np.empty(((10.0*side/radius)**3.0))
    zcoords = np.empty(((10.0*side/radius)**3.0))
    sep = 2*radius + gap
    maxcoords = [side, side, side]
    hy = sep*np.sqrt(3.0)/2.0
    hz = sep*np.sqrt(6.0)/3.0
    ii, jj, kk = [range(0,int(maxcoords[0]/sep) + 1), range(0,int(maxcoords[1]/hy)+1),
        range(0,int(maxcoords[1]/hy)+1)]
    count = 0
    for i,j,k in itertools.product(ii,jj,kk):
        coords = (np.asarray([2*i + (j+k)%2, (np.sqrt(3.0)*(j + 1.0/3.0*(k%2))),
            (2.0*np.sqrt(6.0)/3.0*k)])*(sep/2.0))
        xcoords[count] = coords[0]
        ycoords[count] = coords[1]
        zcoords[count] = coords[2]
        count += 1
    return [xcoords[:count], ycoords[:count], zcoords[:count]]

def makesphere(sphererad, radius, gap = 0.0):
    side = 2.0*sphererad
    tempcoords = makecube(side, radius, gap = gap)
    tempx = tempcoords[0] - sphererad
    tempy = tempcoords[1] - sphererad
    tempz = tempcoords[2] - sphererad
    xcoords = np.empty(((10.0*side/radius)**3.0))
    ycoords = np.empty(((10.0*side/radius)**3.0))
    zcoords = np.empty(((10.0*side/radius)**3.0))
    dist = [tempx[j]**2.0 + tempy[j]**2.0 + tempz[j]**2.0 for j in range(0, len(tempx))]
    mask = np.where(np.sqrt(dist) < sphererad)
    #print mask
    return [tempx[mask], tempy[mask], tempz[mask]]

def makeshell(rinner, router, radius, gap = 0.0):
        side = 2.0*router
        tempcoords = makecube(side, radius, gap = gap)
        tempx = tempcoords[0] - router
        tempy = tempcoords[1] - router
        tempz = tempcoords[2] - router
        xcoords = np.empty(((10.0*side/radius)**3.0))
        ycoords = np.empty(((10.0*side/radius)**3.0))
        zcoords = np.empty(((10.0*side/radius)**3.0))
        dist = [tempx[j]**2.0 + tempy[j]**2.0 + tempz[j]**2.0 for j in range(0, len(tempx))]
        mask = np.where((np.sqrt(dist) < router) & (np.sqrt(dist) > rinner))
        #print mask
        return [tempx[mask], tempy[mask], tempz[mask]]


if __name__ == '__main__':
    allcoords = makeshell(0.0 ,7.0, 1.0)
    allcoords2 = makeshell(7.0,9.0, 1.0)
    allcoords3 = makeshell(9.0,10.0, 1.0)
    print np.shape(allcoords)
    print np.shape(allcoords2)
    print np.shape(allcoords3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(xs = allcoords[0], ys = allcoords[1], zs = allcoords[2], s = 1.0)
    plt.show()
