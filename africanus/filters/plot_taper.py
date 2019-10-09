#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import logging

from africanus.filters import convolution_filter, taper
from africanus.util.cmdline import parse_python_assigns
from africanus.util.requirements import requires_optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D  # noqa
except ImportError as e:
    mpl_ie = e
else:
    mpl_ie = None


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("filter", choices=['kaiser-bessel', 'sinc'],
                   default='kaiser-bessel')
    p.add_argument("-ny", default=1024, type=int)
    p.add_argument("-nx", default=1024, type=int)
    p.add_argument("-hs", "--half-support", default=3, type=int)
    p.add_argument("-os", "--oversample", default=63, type=int)
    p.add_argument("-n", "--normalise", dest="normalise", action="store_true",
                   help="Normalise filter by it's volume")
    p.add_argument("--no-normalise", dest="normalise", action="store_false",
                   help="Do not normalise the filter by it's volume")
    p.add_argument("-k", "--kwargs", default="", type=parse_python_assigns,
                   help="Extra keywords arguments used to create the filter. "
                        "For example 'beta=2.3' to specify a beta shape "
                        "parameter for the Kaiser Bessel")

    return p


@requires_optional('matplotlib.pyplot', 'mpl_toolkits.mplot3d', mpl_ie)
def _plot_taper(data, ny, nx, beta=None):
    hy = ny // 2
    hx = nx // 2
    X, Y = np.mgrid[-hy:hy:1j*data.shape[0], -hx:hx:1j*data.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    xmin = -data.shape[0] // 2
    xmax = data.shape[0] // 2

    ymin = -data.shape[1] // 2
    ymax = data.shape[1] // 2

    zmax = data.max()

    ax.plot([0, 0], [0, 0], zs=[0, 1.2*zmax], color='black')
    ax.plot([0, 0], [ymin, ymax], zs=[zmax, zmax], color='black')
    ax.plot([xmin, xmax], [0, 0], zs=[zmax, zmax], color='black')

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def main():
    args = create_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Creating %s filter with half support of %d "
                 "and %d oversampling" %
                 (args.filter, args.half_support, args.oversample))

    if len(args.kwargs) > 0:
        logging.info("Extra keywords %s" % args.kwargs)

    conv_filter = convolution_filter(args.half_support,
                                     args.oversample,
                                     args.filter,
                                     normalise=args.normalise,
                                     **args.kwargs)

    print(args.kwargs)

    data = taper("kaiser-bessel", args.ny, args.nx, conv_filter, **args.kwargs)

    _plot_taper(data, args.ny, args.nx)
