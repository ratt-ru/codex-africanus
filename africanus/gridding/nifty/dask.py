# -*- coding: utf-8 -*-


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy as np

try:
    import dask
    import dask.array as da
    from dask.base import normalize_token
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    import_error = e
else:
    import_error = None

try:
    import nifty_gridder as ng
except ImportError:
    nifty_import_err = ImportError("Please manually install nifty_gridder "
                                   "from https://gitlab.mpcdf.mpg.de/ift/"
                                   "nifty_gridder.git")
else:
    nifty_import_err = None

from africanus.util.requirements import requires_optional
from ducc0.wgridder import ms2dirty, dirty2ms
from daskms import xds_from_ms, xds_from_table
from numpy.testing import assert_array_equal
from scipy.fft import next_fast_len


def _dot_wrapper(uvw, freq, model, wgt, freq_bin_idx, freq_bin_counts,
                 cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    return _dot_internal(uvw[0], freq, model[0][0][0], wgt,
                         freq_bin_idx[0], freq_bin_counts[0],
                         cellx, celly, nu, nv, epsilon, nthreads,
                         do_wstacking)

def _dot_internal(uvw, freq, model, wgt, freq_bin_idx, freq_bin_counts,
                  cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    nrow, nchan = wgt.shape
    vis = np.zeros((nrow, nchan), dtype=np.complex128)
    for i in range(nband):
        I = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        vis[:, I] = dirty2ms(uvw=uvw, freq=freq[I], dirty=model[i], wgt=wgt[:, I], 
                             pixsize_x=cellx, pixsize_y=celly, nu=nu, nv=nv,
                             epsilon=epsilon, nthreads=nthreads, 
                             do_wstacking=do_wstacking)
    return dirty


def _hdot_wrapper(uvw, freq, ms, wgt, freq_bin_idx, freq_bin_counts,
                  nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    return _hdot_internal(uvw[0][0], freq[0], ms[0][0], wgt[0][0],
                              freq_bin_idx, freq_bin_counts, nx, ny,
                              cellx, celly, nu, nv, epsilon, nthreads,
                              do_wstacking)

def _hdot_internal(uvw, freq, ms, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    dirty = np.zeros((nband, nx, ny))
    for i in range(nband):
        I = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        dirty[i] = ms2dirty(uvw=uvw, freq=freq[I], ms=ms[:, I], wgt=wgt[:, I], 
                            npix_x=nx, npix_y=ny, pixsize_x=cellx, pixsize_y=celly,
                            nu=nu, nv=nv, epsilon=epsilon, nthreads=nthreads, 
                            do_wstacking=do_wstacking)
    return dirty


class wgridder(object):
    """
    Uses dask-ms to provide a chunked up interface to the measurement for the
    wgridder in ducc. For simplicity it is assumed that the channel mapping
    defines the chunking along the frequency axis where the channel mapping
    is defined by the frequency resolution of the measurement and the number
    of required imaging bands. By default it is assumed that all rows for a
    specific imaging band fits into memory (i.e. row_chunks=-1). If this is
    not the case row_chunks should be set explicitly (large chunks preferred).
    
    A list of measurement sets is supported but currently only a single phase
    direction and frequency range with fixed number of channels are allowed. 
    
    Only even pixel sizes are currently supported. 

    If a weighting scheme other than natural weighting is required they need
    to be pre-computed and written to the measurement set in advance.
    """
    def __init__(self, ms, nx, ny, cell_size, cell_size_y=None, nband=None, field=0,
                 ddid=0, precision=1e-7, nthreads=0, do_wstacking=1, row_chunks=-1,
                 data_column='DATA', weight_column='WEIGHT_SPECTRUM',
                 model_column="MODEL_DATA", flag_column=None, 
                 weight_norm='backward', padding=2.0):
        """
        Parameters
        ----------
        ms : list
            List of measurement sets
        nx : int
            Number of pixels along l direction.
        ny : int
            Number of pixels along m dimension.
        cell_size : float
            Cell size in arcseconds.  
        cell_size_y : float, optional
            Cell size in m direction in arcseconds. 
            If None the pixels will be square.  
        nband : int, optional
            Number of imaging bands. 
            If None then the image cube will have the same
            frequency resolution as the measurement set.
        field : int, optional
            Which field to image. Default is field 0 for all
            measurement sets.
            All fields currently need to have the same phase direction.
        ddid : int, optional
            Which DDID to image. Defaults to 0 for all
            measurement sets.
        precision : float, optional
            Gridder error tolerance. Defaults to 1e-7.
        nthreads : int, optional.
            The default of 0 means use all theads.
        do_wstacking : bool, optional
            Whether to perform w-stacking or not. 
            Defaults to True.
        row_chunks : int, optional
            Row chunking per imaging band.
            Default assumes all rows for an
            imaging band fits into memory.
        data_column : string, optional
            Which measurement set column to image. 
            Defaults to 'DATA'.
        weight_column : string, optional
            Defaults to 'WEIGHT_SPECTRUM' but will use 'WEIGHT' if it
            does not exist. Custom columns need to have the same
            dimensions as data. 
        model_column : string, optional
            Which column to write model visibilities to. 
            The default column is 'MODEL_DATA' but the user needs to provide
            a switch in the self.dot method to activate this. 
            This is to avoid accidentally over writing an existing column.
        flag_column : string, optional
            The default of None will take the union of FLAG_ROW and FLAG.
            For different behaviour point to a specific row. 
            Note that flags are implemented by zeroing the corresponding
            weights.
        weight_norm : string, 'backward' or 'ortho'
            This determines how the weights are applied. 
            The default of 'backward' only applies the weights in the
            backward (i.e. hdot) transform. This is the classical way
            to apply the weights but it results in the forward and backward
            transforms not being self adjoint. The 'ortho' mode applies the
            square root of the weights in both directions and therefore
            results in consistent operators. Note that in this mode the
            data is also whitened and care has to be taken when
            writing model visibilities to the measurement set.  
        padding : float, optional
            How much to pad by during gridding.
            Defaults to 2

        Methods
        -------

        dot : 
            The forward transform from image to visibilities (wraps dirty2ms)

        hdot : 
            The backward transform from visibilities to image (wraps ms2dirty).
            Can be used to make the dirty image
            
        make_residual : 
            Compute residual image in place given a model image.

        make_psf :
            Computes an over-sized psf

        """
        if precision > 1e-6:
            self.real_type = np.float32
            self.complex_type = np.complex64
        else:
            self.real_type = np.float64
            self.complex_type=np.complex128

        self.nx = nx
        assert self.nx % 2 == 0
        self.ny = ny
        assert self.ny % 2 == 0
        nu = int(padding * self.nx)
        self.nu = next_fast_len(nu)
        nv = int(padding * self.ny)
        self.nv = next_fast_len(nv)
        self.cell = cell_size * np.pi/60/60/180
        if cell_size_y is not None:
            self.celly = cell_size_y * np.pi/60/60/180
        self.field = field
        self.ddid = ddid
        self.precision = precision
        self.nthreads = nthreads
        self.do_wstacking = do_wstacking
        if weight_norm != 'backward' or weight_norm != 'ortho':
            raise ValueError("Unknown weight_norm supplied")
        else:
            self.weight_norm = weight_norm

        if isinstance(ms, list):
            self.ms = ms
        else:
            self.ms = [ms]
        
        # first we check that phase centres, DDID freqs and pols are the same
        freq = None
        ptype = None
        radec = None
        for ims in self.ms:
            xds = xds_from_ms(ims, columns=('UVW'), chunks={'row':-1})

            # Get subtable data
            ddids = xds_from_table(ms + "::DATA_DESCRIPTION")[0][self.DDID]
            fields = xds_from_table(ms + "::FIELD", group_cols="__row__")[0][self.field]
            spws = xds_from_table(ms + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ms + "::POLARIZATION", group_cols="__row__")

            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]

            for ds in xds:
                if ds.FIELD_ID != self.field and ds.DATA_DESC_ID != self.ddid:
                    continue

                # check correct phase direction
                field = fields[self.field]
                if radec is None:
                    radec = field.PHASE_DIR.data.squeeze()
                else: 
                    assert_array_equal(radec, field.PHASE_DIR.data.squeeze())

                ddid = ddids[ds.DATA_DESC_ID]

                # same polarisation type
                pol = pols[ddid.POLARIZATION_ID.values[0]]
                corr_type_set = set(pol.CORR_TYPE.data.squeeze())
                if corr_type_set.issubset(set([9, 10, 11, 12])):
                    pol_type = 'linear'
                elif corr_type_set.issubset(set([5, 6, 7, 8])):
                    pol_type = 'circular'
                else:
                    raise ValueError("Cannot determine polarisation type "
                                    "from correlations %s. Constructing "
                                    "a feed rotation matrix will not be "
                                    "possible." % (corr_type_set,))

                if ptype is None:
                    ptype = pol_type
                else:
                    assert ptype == pol_type

                # same frequencies            
                spw = spws[ddid.SPECTRAL_WINDOW_ID.values[0]]
                if freq is None:
                    freq = spw.CHAN_FREQ.data
                else:
                    assert_array_equal(freq, spw.CHAN_FREQ.data)

        self.freq = freq
        self.ptype = ptype

        # compute freq mapping and channel chunks
        self.nchan = freq.size
        if nband is None:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])
        self.chan_chunks = list(self.freq_mapping[1::] - self.freq_mapping[0:-1])
        self.freq_da = da.from_array(self.freq, chunks=self.chan_chunks)
        self.freq_bin_idx = da.from_array(self.freq_mapping[0:-1], chunks=1)
        self.freq_bin_counts = da.from_array(self.freq_mapping[1::] - self.freq_mapping[0:-1], chunks=1)

        ######################################################################################################################
        # meta info for xds_from_table
        self.data_column = data_column
        self.weight_column = weight_column
        self.model_column = model_column
        self.ms = ms
        self.row_chunks = row_chunks
        self.schema = {
            data_column: {'dims': ('chan',)},
            weight_column: {'dims': ('chan', )},
            "UVW": {'dims': ('uvw',)},
        }

    def dot(self, x, write=False, column=None):
        """
        Implements forward transform i.e.

        V = Rx

        where R is the interferometric response.
        This is not recommended if the result does not
        fit in memory. For this reason there is an option
        to write the result to a column in the MS.
        If write=True then it will attempt to write the
        visibilities to column. If column is None then the
        default of MODEL_DATA is used.  
        """
        if write and column is None:
            column = self.model_column
        model = da.from_array(x, chunks=(1, self.nx, self.ny))
        for ims in self.ms:
            xds = xds_from_table(ims, chunks={"row":self.row_chunks, "chan": self.chan_chunks},
                                 table_schema=self.schema)
            out_data = []
            for ds in xds:
                if ds.FIELD_ID != self.field and ds.DATA_DESC_ID != self.ddid:
                    continue
                weights = getattr(ds, self.weight_column).data
                uvw = ds.UVW.data

                vis = da.blockwise(_dot_wrapper, ('row', 'chan'),
                                   uvw, ('row', 'three'), 
                                   self.freq_da, ('chan',),
                                   model, ('band', 'nx', 'ny'),
                                   weights, ('row', 'chan'),
                                   self.freq_bin_idx, ('band',),
                                   self.freq_bin_counts, ('band'),
                                   self.cell, None, 
                                   self.celly, None,
                                   self.nu, None, 
                                   self.nv, None,
                                   self.precision, None,
                                   self.do_wstacking, None,
                                   new_axes={"nx": self.nx, "ny": self.ny},
                                   dtype=self.complex_type)
                
                if write:
                    data_vars = {column:(('row', 'chan'), vis)}
                    out_ds = Dataset(data_vars)
                    out_data.append(out_ds)

            if write:
                xds_to_table(out_data, ims, columns="ALL")


        return dirty.compute()

    def hdot(self, model_data=None):
        """
        Implements the backward transform

        I^D = R.H V

        It is assumed that the result always fits in memory.
        If model_data is not passed in then it produces the
        dirty image.
        """
        if model_data is None:
            model_data = self.data_column
        for ims in self.ms:
            xds = xds_from_table(ims, chunks={"row":self.row_chunks, "chan": self.chan_chunks},
                                 table_schema=self.schema)
            for ds in xds:
                if ds.FIELD_ID != self.field and ds.DATA_DESC_ID != self.ddid:
                    continue
                data = getattr(ds, model_data)
                weights = getattr(ds, self.weight_column)
                uvw = ds.UVW.data

                dirty += da.blockwise(_hdot_wrapper, ('band', 'nx', 'ny'),
                                      uvw, ('row', 'three'), 
                                      self.freq_da, ('chan',),
                                      data, ('row', 'chan'),
                                      weights, ('row', 'chan'),
                                      self.freq_bin_idx, ('band',),
                                      self.freq_bin_counts, ('band'),
                                      self.nx, None,
                                      self.ny, None, 
                                      self.cell, None, 
                                      self.celly, None,
                                      self.nu, None, 
                                      self.nv, None,
                                      self.precision, None,
                                      self.do_wstacking, None,
                                      new_axes={"nx": self.nx, "ny": self.ny},
                                      dtype=self.real_type)

        return dirty.compute()


    def make_residual(self, x, v_dof=None):
        print("Making residual")
        residual = np.zeros(x.shape, dtype=x.dtype)
        xds = xds_from_ms(self.ms, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = getattr(ds, self.data_column).data
            weights = getattr(ds, self.weight_column).data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field

                # get residual vis
                if weighti.any():
                    residual_vis = weighti * datai - ng.dirty2ms(uvw=uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=weighti,
                                                                pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                                nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)

                    # make residual image
                    residual[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=residual_vis, wgt=weighti,
                                            npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                            epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return residual

    

    def make_psf(self):
        print("Making PSF")
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            weights = getattr(ds, self.weight_column).data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)

                if weighti.any():
                    psf_array[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti.astype(self.complex_type), wgt=weighti,
                                                npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                                epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array


class GridderConfigWrapper(object):
    """
    Wraps a nifty GridderConfiguration for pickling purposes.
    """

    def __init__(self, nx=1024, ny=1024, eps=2e-13,
                 cell_size_x=2.0, cell_size_y=2.0):
        self.nx = nx
        self.ny = ny
        self.csx = cell_size_x
        self.csy = cell_size_y
        self.eps = eps
        self.grid_config = ng.GridderConfig(nx, ny, eps,
                                            cell_size_x,
                                            cell_size_y)

    @property
    def object(self):
        return self.grid_config

    def __reduce__(self):
        return (GridderConfigWrapper,
                (self.nx, self.ny, self.eps, self.csx, self.csy))


if import_error is None:
    @normalize_token.register(GridderConfigWrapper)
    def normalize_gridder_config_wrapper(gc):
        return normalize_token((gc.nx, gc.ny, gc.csx, gc.csy, gc.eps))


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def grid_config(nx=1024, ny=1024, eps=2e-13, cell_size_x=2.0, cell_size_y=2.0):
    """
    Returns a wrapper around a NIFTY GridderConfiguration object.


    Parameters
    ----------
    nx : int, optional
        Number of X pixels in the grid. Defaults to 1024.
    ny : int, optional
        Number of Y pixels in the grid. Defaults to 1024.
    cell_size_x : float, optional
        Cell size of the X pixel in arcseconds. Defaults to 2.0.
    cell_size_y : float, optional
        Cell size of the Y pixel in arcseconds. Defaults to 2.0.
    eps : float
        Gridder accuracy error. Defaults to 2e-13


    Returns
    -------
    grid_config : :class:`GridderConfigWrapper`
        The NIFTY Gridder Configuration
    """
    return GridderConfigWrapper(nx, ny, eps, cell_size_x, cell_size_y)


def _nifty_baselines(uvw, chan_freq):
    """ Wrapper function for creating baseline mappings per row chunk """
    assert len(chan_freq) == 1, "Handle multiple channel chunks"
    return ng.Baselines(uvw[0], chan_freq[0])


def _nifty_indices(baselines, grid_config, flag,
                   chan_begin, chan_end, wmin, wmax):
    """ Wrapper function for creating indices per row chunk """
    return ng.getIndices(baselines, grid_config, flag[0],
                         chan_begin, chan_end, wmin, wmax)


def _nifty_grid(baselines, grid_config, indices, vis, weights):
    """ Wrapper function for creating a grid of visibilities per row chunk """
    assert len(vis) == 1 and type(vis) == list
    return ng.ms2grid_c(baselines, grid_config, indices,
                        vis[0], None, weights[0])[None, :, :]


def _nifty_grid_streams(baselines, grid_config, indices,
                        vis, weights, grid_in=None):
    """ Wrapper function for creating a grid of visibilities per row chunk """
    return ng.ms2grid_c(baselines, grid_config, indices,
                        vis, grid_in, weights)


class GridStreamReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first accessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph serially summing coherencies in
    ``stream`` parallel streams.
    """

    def __init__(self, baselines, indices, gc,
                 corr_vis, corr_weights,
                 corr, streams):
        token = dask.base.tokenize(baselines, indices, gc,
                                   corr_vis, corr_weights,
                                   corr, streams)
        self.name = "-".join(("nifty-grid-stream", str(corr), token))
        self.bl_name = baselines.name
        self.idx_name = indices.name
        self.cvis_name = corr_vis.name
        self.wgt_name = corr_weights.name
        self.gc = gc
        self.corr = corr

        self.row_blocks = indices.numblocks[0]
        self.streams = streams

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Extract dimension blocks
        return self.row_blocks

    def _create_dict(self):
        # Graph dictionary
        layers = {}

        name = self.name
        gc = self.gc
        row_blocks = self.row_blocks
        streams = self.streams
        indices_name = self.idx_name
        baselines_name = self.bl_name
        corr_vis_name = self.cvis_name
        corr_wgt_name = self.wgt_name

        cb = 0  # Assume single channel

        # Split our row blocks by the number of streams
        # For all blocks in a stream, we'll grid those
        # blocks serially, passing one grid into the other
        row_block_chunks = (row_blocks + streams - 1) // streams

        for rb_start in range(0, row_blocks, row_block_chunks):
            rb_end = min(rb_start + row_block_chunks, row_blocks)
            last_key = None

            for rb in range(rb_start, rb_end):
                fn = (_nifty_grid_streams,
                      (baselines_name, rb),
                      gc,
                      (indices_name, rb),
                      (corr_vis_name, rb, cb),
                      (corr_wgt_name, rb, cb),
                      # Re-use grid from last operation if present
                      last_key)

                key = (name, rb, cb)
                layers[key] = fn
                last_key = key

        return layers


class FinalGridReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first accessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph serially summing coherencies in
    ``stream`` parallel streams.
    """

    def __init__(self, grid_stream_reduction):
        self.in_name = grid_stream_reduction.name
        token = dask.base.tokenize(grid_stream_reduction)
        self.name = "grid-stream-reduction-" + token
        self.row_blocks = grid_stream_reduction.row_blocks
        self.streams = grid_stream_reduction.streams

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Always returns a single block, corresponding to one image
        return 1

    def _create_dict(self):
        # Graph dictionary
        layers = {}

        row_blocks = self.row_blocks
        streams = self.streams
        cb = 0  # Assume single channel

        # Split our row blocks by the number of streams
        # For all blocks in a stream, we'll grid those
        # blocks serially, passing one grid into the other
        row_block_chunks = (row_blocks + streams - 1) // streams
        last_keys = []

        for ob, rb_start in enumerate(range(0, row_blocks, row_block_chunks)):
            rb_end = min(rb_start + row_block_chunks, row_blocks)
            last_keys.append((self.in_name, rb_end - 1, cb))

        key = (self.name, 0, 0)
        task = (sum, last_keys)
        layers[key] = task

        return layers


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def grid(vis, uvw, flags, weights, frequencies, grid_config,
         wmin=-1e30, wmax=1e30, streams=None):
    """
    Grids the supplied visibilities in parallel. Note that
    a grid is create for each visibility chunk.

    Parameters
    ----------
    vis : :class:`dask.array.Array`
        visibilities of shape :code:`(row, chan, corr)`
    uvw : :class:`dask.array.Array`
        uvw coordinates of shape :code:`(row, 3)`
    flags : :class:`dask.array.Array`
        flags of shape :code:`(row, chan, corr)`
    weights : :class:`dask.array.Array`
        weights of shape :code:`(row, chan, corr)`.
    frequencies : :class:`dask.array.Array`
        frequencies of shape :code:`(chan,)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding Configuration
    wmin : float
        Minimum W coordinate to grid. Defaults to -1e30.
    wmax : float
        Maximum W coordinate to grid. Default to 1e30.
    streams : int, optional
        Number of parallel gridding operations. Default to None,
        in which case as many grids as visibility chunks will
        be created.

    Returns
    -------
    grid : :class:`dask.array.Array`
        grid of shape :code:`(ny, nx, corr)`
    """
    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel currently unsupported")

    # Create a baseline object per row chunk
    baselines = da.blockwise(_nifty_baselines, ("row",),
                             uvw, ("row", "uvw"),
                             frequencies, ("chan",),
                             dtype=np.object)

    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel unsupported")

    gc = grid_config.object
    grids = []

    for corr in range(vis.shape[2]):
        corr_flags = flags[:, :, corr]
        corr_vis = vis[:, :, corr]
        corr_weights = weights[:, :, corr]

        indices = da.blockwise(_nifty_indices, ("row",),
                               baselines, ("row",),
                               gc, None,
                               corr_flags, ("row", "chan"),
                               -1, None,  # channel begin
                               -1, None,  # channel end
                               wmin, None,
                               wmax, None,
                               dtype=np.int32)

        if streams is None:
            # Standard parallel reduction, possibly memory hungry
            # if many threads (and thus grids) are gridding
            # parallel
            grid = da.blockwise(_nifty_grid, ("row", "nu", "nv"),
                                baselines, ("row",),
                                gc, None,
                                indices, ("row",),
                                corr_vis, ("row", "chan"),
                                corr_weights, ("row", "chan"),
                                new_axes={"nu": gc.Nu(), "nv": gc.Nv()},
                                adjust_chunks={"row": 1},
                                dtype=np.complex128)

            grids.append(grid.sum(axis=0))
        else:
            # Stream reduction
            layers = GridStreamReduction(baselines, indices, gc,
                                         corr_vis, corr_weights,
                                         corr, streams)
            deps = [baselines, indices, corr_vis, corr_weights]
            graph = HighLevelGraph.from_collections(layers.name, layers, deps)
            chunks = corr_vis.chunks
            grid_stream_red = da.Array(graph, layers.name, chunks, vis.dtype)

            layers = FinalGridReduction(layers)
            deps = [grid_stream_red]
            graph = HighLevelGraph.from_collections(layers.name, layers, deps)
            chunks = ((gc.Nu(),), (gc.Nv(),))
            corr_grid = da.Array(graph, layers.name, chunks, vis.dtype)

            grids.append(corr_grid)

    return da.stack(grids, axis=2)


def _nifty_dirty(grid, grid_config):
    """ Wrapper function for creating a dirty image """
    grids = [grid_config.grid2dirty_c(grid[:, :, c]).real
             for c in range(grid.shape[2])]

    return np.stack(grids, axis=2)


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def dirty(grid, grid_config):
    """
    Computes the dirty image from gridded visibilities and the
    gridding configuration.

    Parameters
    ----------
    grid : :class:`dask.array.Array`
        Gridded visibilities of shape :code:`(nv, nu, ncorr)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding configuration

    Returns
    -------
    dirty : :class:`dask.array.Array`
        dirty image of shape :code:`(ny, nx, corr)`
    """

    gc = grid_config.object
    nx = gc.Nxdirty()
    ny = gc.Nydirty()

    return da.blockwise(_nifty_dirty, ("nx", "ny", "corr"),
                        grid, ("nx", "ny", "corr"),
                        gc, None,
                        adjust_chunks={"nx": nx, "ny": ny},
                        dtype=grid.real.dtype)


def _nifty_model(image, grid_config):
    """ Wrapper function for creating a dirty image """
    images = [grid_config.dirty2grid_c(image[:, :, c])
              for c in range(image.shape[2])]

    return np.stack(images, axis=2)


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def model(image, grid_config):
    """
    Computes model visibilities from an image
    and a gridding configuration.

    Parameters
    ----------
    image : :class:`dask.array.Array`
        Image of shape :code:`(ny, nx, corr)`.
    grid_config : :class:`GridderConfigWrapper`
        nifty gridding configuration object

    Returns
    -------
    model_vis : :class:`dask.array.Array`
        Model visibilities of shape :code:`(nu, nv, corr)`.
    """

    gc = grid_config.object
    nu = gc.Nu()
    nv = gc.Nv()

    return da.blockwise(_nifty_model, ("nu", "nv", "corr"),
                        image, ("nu", "nv", "corr"),
                        gc, None,
                        adjust_chunks={"nu": nu, "nv": nv},
                        dtype=image.dtype)


def _nifty_degrid(grid, baselines, indices, grid_config):
    assert len(grid) == 1 and len(grid[0]) == 1
    return ng.grid2ms_c(baselines, grid_config.object, indices, grid[0][0])


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def degrid(grid, uvw, flags, weights, frequencies,
           grid_config, wmin=-1e30, wmax=1e30):
    """
    Degrids the visibilities from the supplied grid in parallel.

    Parameters
    ----------
    grid : :class:`dask.array.Array`
        gridded visibilities of shape :code:`(ny, nx, corr)`
    uvw : :class:`dask.array.Array`
        uvw coordinates of shape :code:`(row, 3)`
    flags : :class:`dask.array.Array`
        flags of shape :code:`(row, chan, corr)`
    weights : :class:`dask.array.Array`
        weights of shape :code:`(row, chan, corr)`.
        Currently unsupported and ignored.
    frequencies : :class:`dask.array.Array`
        frequencies of shape :code:`(chan,)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding Configuration
    wmin : float
        Minimum W coordinate to grid. Defaults to -1e30.
    wmax : float
        Maximum W coordinate to grid. Default to 1e30.

    Returns
    -------
    grid : :class:`dask.array.Array`
        grid of shape :code:`(ny, nx, corr)`
    """

    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel currently unsupported")

    # Create a baseline object per row chunk
    baselines = da.blockwise(_nifty_baselines, ("row",),
                             uvw, ("row", "uvw"),
                             frequencies, ("chan",),
                             dtype=np.object)

    gc = grid_config.object
    vis_chunks = []

    for corr in range(grid.shape[2]):
        corr_flags = flags[:, :, corr].map_blocks(np.require, requirements="C")
        corr_grid = grid[:, :, corr].map_blocks(np.require, requirements="C")

        indices = da.blockwise(_nifty_indices, ("row",),
                               baselines, ("row",),
                               gc, None,
                               corr_flags, ("row", "chan"),
                               -1, None,  # channel begin
                               -1, None,  # channel end
                               wmin, None,
                               wmax, None,
                               dtype=np.int32)

        vis = da.blockwise(_nifty_degrid, ("row", "chan"),
                           corr_grid, ("ny", "nx"),
                           baselines, ("row",),
                           indices, ("row",),
                           grid_config, None,
                           new_axes={"chan": frequencies.shape[0]},
                           dtype=grid.dtype)

        vis_chunks.append(vis)

    return da.stack(vis_chunks, axis=2)
