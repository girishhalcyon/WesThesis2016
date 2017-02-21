import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=3, pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker
        from matplotlib.offsetbox import TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, fc="none"))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx,
                                                    minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                           align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False,
                                   **kwargs)


def add_scalebar(ax, matchx=True, matchy=True, hidex=False, hidey=False,
                 **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """
    def f(axis):
        l_1 = axis.get_majorticklocs()
        return len(l_1) > 1 and (l_1[1] - l_1[0])
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return sb


def plot_2D_simulation(rebsim, color_mass=False, color_dens=True, mode='SHOW',
                       inflate=100.0, wd_status='SMALL'):
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
    if wd_status == 'HIDE':
        argloc = np.argmax(mass_arr)
        xs = np.delete(xs, argloc)
        ys = np.delete(ys, argloc)
        zs = np.delete(zs, argloc)
        rad_arr = np.delete(rad_arr, argloc)
        mass_arr = np.delete(mass_arr, argloc)
    elif wd_status == 'SMALL':
        argloc = np.argmax(mass_arr)
        mass_arr[argloc] = np.median(mass_arr)
    if color_dens:
        dens_arr = mass_arr/((4.0/3.0)*np.pi*(rad_arr**3.0))
        mappable = ax1.scatter(xs, ys, c=dens_arr, s=inflate*rad_arr)
        ax1.set_aspect('equal')
        ax1.set_title('Top-down view')
        map_2 = ax2.scatter(xs, zs, c=dens_arr, s=inflate*rad_arr)
        # ax2.set_ylim(ax2.get_xlim())
        ax2.set_title('Edge-on view')
    else:
        mappable = ax1.scatter(xs, ys, c=mass_arr, s=inflate*rad_arr)
        # ax1.set_aspect('equal')
        ax1.set_title('Top-down view')
        map_2 = ax2.scatter(xs, zs, c=mass_arr, s=inflate*rad_arr)
        # ax2.set_ylim(ax2.get_xlim())
        ax2.set_title('Edge-on view')
    add_scalebar(ax1)
    add_scalebar(ax2)
    plt.colorbar(mappable, ax=ax1)
    plt.colorbar(map_2, ax=ax2)
    if mode == 'SHOW':
        plt.show()
    elif mode == 'RETURN':
        return fig
    else:
        savename = mode + '.pdf'
        plt.savefig(savename)
    plt.clf()


def plot_light_curve(time, flux, title, fmt, mode='SHOW', savetitle=None):
    plt.plot(time, flux, fmt)
    plt.title(title)
    if time[0] > 10.0:
        x_title = 'Time - ' + str(np.min(time)) + ' [s]'
        time = time - np.min(time)
    else:
        x_title = 'Time'
    x_range = np.max(time) - np.min(time)
    y_range = np.max(flux) - np.min(flux)
    plt.xlim((0.0, np.max(time) + 0.1*x_range))
    plt.ylim((np.min(flux) - 0.1*y_range, np.max(flux) + 0.1*y_range))
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Flux')
    if mode == 'SHOW':
        plt.show()
    elif mode == 'SAVE':
        save_name = savetitle + '.pdf'
        plt.savefig(save_name)
    else:
        plt.show()
    plt.close()
