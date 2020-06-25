import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
import argparse
from africanus.gridding.nifty.dask import wgridder


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str)
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str)
    p.add_argument("--model_column", default="MODEL_DATA", type=str)
    p.add_argument("--field", type=int, default=0)
    p.add_argument("--ddid", type=int, default=0)
    p.add_argument("--pol_products", type=str, default='I')
    p.add_argument("--row_chunks", type=int, default=-1)
    p.add_argument("--ncpu", type=int, default=0)
    p.add_argument("--fov", default=1.0)
    p.add_argument("--srf", default=1.2)
    p.add_argument("--nband", default=None, type=int)
    return p

if __name__=="__main__":
    args = create_parser().parse_args()

    if not args.ncpu:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()
        

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    # get max uv coords over all fields
    uvw = []
    freq = None
    for ims in args.ms:
        xds = xds_from_ms(ims, chunks={'row':-1})
        
        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
        spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
        pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

        # Get subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws = dask.compute(spws)[0]
        pols = dask.compute(pols)[0]

        for ds in xds:
            if ds.FIELD_ID != args.field and ds.DATA_DESC_ID != args.ddid:
                continue

            uvw.append(ds.UVW.data.compute())
            ddid = ddids[ds.DATA_DESC_ID]
            spw = spws[ddid.SPECTRAL_WINDOW_ID.values[0]]
            if freq is None:
                freq = spw.CHAN_FREQ.data
            else:
                assert_array_equal(freq, spw.CHAN_FREQ.data)
    uvw = np.concatenate(uvw)
    
    # set cell size
    from africanus.constants import c as lightspeed
    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)

    cell_rad = cell_N/args.srf
    args.cell_size = cell_rad*60*60*180/np.pi
    print("Cell size set to %5.5e arcseconds" % args.cell_size)
    
    # set number of pixels
    fov = args.fov*3600  # deg2asec
    nx = int(fov/args.cell_size)
    from scipy.fftpack import next_fast_len
    args.nx = next_fast_len(nx)
    args.ny = next_fast_len(nx)

    if args.nband is None:
        args.nband = freq.size

    print("Image size set to (%i, %i, %i)"%(args.nband, args.nx, args.ny))

    R = wgridder(args.ms, args.nx, args.ny, args.cell_size, nband=args.nband, nthreads=args.ncpu)

    
    x = np.zeros((args.nband, args.nx, args.ny), dtype=np.float64)
    x[:, args.nx//2, args.ny//2] = 1.0
    R.dot(x, column='MODEL_DATA')

    # dirty = R.hdot()
    # import matplotlib as mpl
    # mpl.use('TkAgg')
    # import matplotlib.pyplot as plt
    # for i in range(args.nband):
    #     plt.imshow(dirty[i])
    #     plt.colorbar()
    #     plt.show()

    