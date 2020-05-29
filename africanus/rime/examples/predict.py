#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from collections import namedtuple
from functools import lru_cache
from operator import getitem
import weakref

import numpy as np

try:
    from astropy.io import fits
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import Tigger
    from daskms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.util.beams import beam_filenames, beam_grids
from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import (phase_delay, predict_vis, parallactic_angles,
                                 beam_cube_dde, feed_rotation)
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.util.requirements import requires_optional


_einsum_corr_indices = 'ijkl'


def _brightness_schema(corrs, index):
    if corrs == 4:
        return "sf" + _einsum_corr_indices[index:index + 2], index + 1
    else:
        return "sfi", index


def _phase_delay_schema(corrs, index):
    return "srf", index


def _spi_schema(corrs, index):
    return "s", index


def _gauss_shape_schema(corrs, index):
    return "srf", index


def _bl_jones_output_schema(corrs, index):
    if corrs == 4:
        return "->srfi" + _einsum_corr_indices[index]
    else:
        return "->srfi"


_rime_term_map = {
    'brightness': _brightness_schema,
    'phase_delay': _phase_delay_schema,
    'spi': _spi_schema,
    'gauss_shape': _gauss_shape_schema,
}


def corr_schema(pol):
    """
    Parameters
    ----------
    pol : Dataset

    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    # Select the single row out
    corrs = pol.NUM_CORR.data[0]
    corr_types = pol.CORR_TYPE.data[0]

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def baseline_jones_multiply(corrs, *args):
    names = args[::2]
    arrays = args[1::2]

    input_einsum_schemas = []
    corr_index = 0

    for name, array in zip(names, arrays):
        try:
            # Obtain function for prescribing the input einsum schema
            schema_fn = _rime_term_map[name]
        except KeyError:
            raise ValueError("Unknown RIME term '%s'" % name)
        else:
            # Extract it and the next corr index
            einsum_schema, corr_index = schema_fn(corrs, corr_index)
            input_einsum_schemas.append(einsum_schema)

            if not len(einsum_schema) == array.ndim:
                raise ValueError("%s len(%s) == %d != %s.ndim"
                                 % (name, einsum_schema,
                                    len(einsum_schema), array.shape))

    output_schema = _bl_jones_output_schema(corrs, corr_index)
    schema = ",".join(input_einsum_schemas) + output_schema

    return da.einsum(schema, *arrays)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-mc", "--model-chunks", type=int, default=10)
    p.add_argument("-b", "--beam", default=None)
    p.add_argument("-l", "--l-axis", default="L")
    p.add_argument("-m", "--m-axis", default="M")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees")
    return p


@lru_cache(maxsize=16)
def load_beams(beam_file_schema, corr_types, l_axis, m_axis):

    class FITSFile(object):
        """ Exists so that fits file is closed when last ref is gc'd """

        def __init__(self, filename):
            self.hdul = hdul = fits.open(filename)
            assert len(hdul) == 1
            self.__del_ref = weakref.ref(self, lambda r: hdul.close())

    # Open files and get headers
    beam_files = []
    headers = []

    for corr, (re, im) in beam_filenames(beam_file_schema, corr_types).items():
        re_f = FITSFile(re)
        im_f = FITSFile(im)
        beam_files.append((corr, (re_f, im_f)))
        headers.append((corr, (re_f.hdul[0].header, im_f.hdul[0].header)))

    # All FITS headers should agree (apart from DATE)
    flat_headers = []

    for corr, (re_header, im_header) in headers:
        if "DATE" in re_header:
            del re_header["DATE"]
        if "DATE" in im_header:
            del im_header["DATE"]
        flat_headers.append(re_header)
        flat_headers.append(im_header)

    if not all(flat_headers[0] == h for h in flat_headers[1:]):
        raise ValueError("BEAM FITS Header Files differ")

    #  Map FITS header type to NumPy type
    BITPIX_MAP = {8: np.dtype('uint8').type, 16: np.dtype('int16').type,
                  32: np.dtype('int32').type, -32: np.dtype('float32').type,
                  -64: np.dtype('float64').type}

    header = flat_headers[0]
    bitpix = header['BITPIX']

    try:
        dtype = BITPIX_MAP[bitpix]
    except KeyError:
        raise ValueError("No mapping from BITPIX %s to a numpy type" % bitpix)
    else:
        dtype = np.result_type(dtype, np.complex64)

    if not header['NAXIS'] == 3:
        raise ValueError("FITS must have exactly three axes. "
                         "L or X, M or Y and FREQ. NAXIS != 3")

    (l_ax, l_grid), (m_ax, m_grid), (nu_ax, nu_grid) = beam_grids(header,
                                                                  l_axis,
                                                                  m_axis)

    # Shape of each correlation
    shape = (l_grid.shape[0], m_grid.shape[0], nu_grid.shape[0])

    # Axis tranpose, FITS is FORTRAN ordered
    ax = (nu_ax - 1, m_ax - 1, l_ax - 1)

    def _load_correlation(re, im, ax):
        # Read real and imaginary for each correlation
        return (re.hdul[0].data.transpose(ax) +
                im.hdul[0].data.transpose(ax)*1j)

    # Create delayed loads of the beam
    beam_loader = dask.delayed(_load_correlation)

    beam_corrs = [beam_loader(re, im, ax)
                  for c, (corr, (re, im)) in enumerate(beam_files)]
    beam_corrs = [da.from_delayed(bc, shape=shape, dtype=dtype)
                  for bc in beam_corrs]

    # Stack correlations and rechunk to one great big block
    beam = da.stack(beam_corrs, axis=3)
    beam = beam.rechunk(shape + (len(corr_types),))

    # Dask arrays for the beam extents and beam frequency grid
    beam_lm_ext = np.array([[l_grid[0], l_grid[-1]], [m_grid[0], m_grid[-1]]])
    beam_lm_ext = da.from_array(beam_lm_ext, chunks=beam_lm_ext.shape)
    beam_freq_grid = da.from_array(nu_grid, chunks=nu_grid.shape)

    return beam, beam_lm_ext, beam_freq_grid


def parse_sky_model(filename, chunks):
    """
    Parses a Tigger sky model

    Parameters
    ----------
    filename : str
        Sky Model filename
    chunks : tuple of ints or int
        Source chunking strategy

    Returns
    -------
    source_data : dict
        Dictionary of source data,
        :code:`{'point': (...), 'gauss': (...) }`
    """
    sky_model = Tigger.load(filename, verbose=False)

    _empty_spectrum = object()

    point_radec = []
    point_stokes = []
    point_spi = []
    point_ref_freq = []

    gauss_radec = []
    gauss_stokes = []
    gauss_spi = []
    gauss_ref_freq = []
    gauss_shape = []

    for source in sky_model.sources:
        ra = source.pos.ra
        dec = source.pos.dec
        typecode = source.typecode.lower()

        I = source.flux.I  # noqa
        Q = source.flux.Q
        U = source.flux.U
        V = source.flux.V

        spectrum = (getattr(source, "spectrum", _empty_spectrum)
                    or _empty_spectrum)

        try:
            # Extract reference frequency
            ref_freq = spectrum.freq0
        except AttributeError:
            ref_freq = sky_model.freq0

        try:
            # Extract SPI for I.
            # Zero Q, U and V to get 1 on the exponential
            spi = [[spectrum.spi]*4]
        except AttributeError:
            # Default I SPI to -0.7
            spi = [[0, 0, 0, 0]]

        if typecode == "gau":
            emaj = source.shape.ex
            emin = source.shape.ey
            pa = source.shape.pa

            gauss_radec.append([ra, dec])
            gauss_stokes.append([I, Q, U, V])
            gauss_spi.append(spi)
            gauss_ref_freq.append(ref_freq)
            gauss_shape.append([emaj, emin, pa])

        elif typecode == "pnt":
            point_radec.append([ra, dec])
            point_stokes.append([I, Q, U, V])
            point_spi.append(spi)
            point_ref_freq.append(ref_freq)
        else:
            raise ValueError("Unknown source morphology %s" % typecode)

    Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
    Gauss = namedtuple("Gauss", ["radec", "stokes", "spi", "ref_freq",
                                 "shape"])

    source_data = {}

    if len(point_radec) > 0:
        source_data['point'] = Point(
                    da.from_array(point_radec, chunks=(chunks, -1)),
                    da.from_array(point_stokes, chunks=(chunks, -1)),
                    da.from_array(point_spi, chunks=(chunks, 1, -1)),
                    da.from_array(point_ref_freq, chunks=chunks))

    if len(gauss_radec) > 0:
        source_data['gauss'] = Gauss(
                    da.from_array(gauss_radec, chunks=(chunks, -1)),
                    da.from_array(gauss_stokes, chunks=(chunks, -1)),
                    da.from_array(gauss_spi, chunks=(chunks, 1, -1)),
                    da.from_array(gauss_ref_freq, chunks=chunks),
                    da.from_array(gauss_shape, chunks=(chunks, -1)))

    return source_data


def support_tables(args):
    """
    Parameters
    ----------
    args : object
        Script argument objects

    Returns
    -------
    table_map : dict of Dataset
        {name: dataset}
    """

    n = {k: '::'.join((args.ms, k)) for k
         in ("ANTENNA", "DATA_DESCRIPTION", "FIELD",
             "SPECTRAL_WINDOW", "POLARIZATION")}

    # All rows at once
    lazy_tables = {"ANTENNA": xds_from_table(n["ANTENNA"])}

    compute_tables = {
        # Fixed shape rows
        "DATA_DESCRIPTION": xds_from_table(n["DATA_DESCRIPTION"]),
        # Variably shaped, need a dataset per row
        "FIELD": xds_from_table(n["FIELD"],
                                group_cols="__row__"),
        "SPECTRAL_WINDOW": xds_from_table(n["SPECTRAL_WINDOW"],
                                          group_cols="__row__"),
        "POLARIZATION": xds_from_table(n["POLARIZATION"],
                                       group_cols="__row__"),
    }

    lazy_tables.update(dask.compute(compute_tables)[0])
    return lazy_tables


def _zero_pes(parangles, frequency, dtype_):
    """ Create zeroed pointing errors """
    ntime, na = parangles.shape
    nchan = frequency.shape[0]
    return np.zeros((ntime, na, nchan, 2), dtype=dtype_)


def _unity_ant_scales(parangles, frequency, dtype_):
    """ Create zeroed antenna scalings """
    _, na = parangles[0].shape
    nchan = frequency.shape[0]
    return np.ones((na, nchan, 2), dtype=dtype_)


def dde_factory(args, ms, ant, field, pol, lm, utime, frequency):
    if args.beam is None:
        return None

    # Beam is requested
    corr_type = tuple(pol.CORR_TYPE.data[0])

    if not len(corr_type) == 4:
        raise ValueError("Need four correlations for DDEs")

    parangles = parallactic_angles(utime, ant.POSITION.data,
                                   field.PHASE_DIR.data[0][0])

    corr_type_set = set(corr_type)

    if corr_type_set.issubset(set([9, 10, 11, 12])):
        pol_type = 'linear'
    elif corr_type_set.issubset(set([5, 6, 7, 8])):
        pol_type = 'circular'
    else:
        raise ValueError("Cannot determine polarisation type "
                         "from correlations %s. Constructing "
                         "a feed rotation matrix will not be "
                         "possible." % (corr_type,))

    # Construct feed rotation
    feed_rot = feed_rotation(parangles, pol_type)

    dtype = np.result_type(parangles, frequency)

    # Create zeroed pointing errors
    zpe = da.blockwise(_zero_pes, ("time", "ant", "chan", "comp"),
                       parangles, ("time", "ant"),
                       frequency, ("chan",),
                       dtype, None,
                       new_axes={"comp": 2},
                       dtype=dtype)

    # Created zeroed antenna scaling factors
    zas = da.blockwise(_unity_ant_scales, ("ant", "chan", "comp"),
                       parangles, ("time", "ant"),
                       frequency, ("chan",),
                       dtype, None,
                       new_axes={"comp": 2},
                       dtype=dtype)

    # Load the beam information
    beam, lm_ext, freq_map = load_beams(args.beam, corr_type,
                                        args.l_axis, args.m_axis)

    # Introduce the correlation axis
    beam = beam.reshape(beam.shape[:3] + (2, 2))

    beam_dde = beam_cube_dde(beam, lm_ext, freq_map, lm, parangles,
                             zpe, zas,
                             frequency)

    # Multiply the beam by the feed rotation to form the DDE term
    return da.einsum("stafij,tajk->stafik", beam_dde, feed_rot)


def vis_factory(args, source_type, sky_model,
                ms, ant, field, spw, pol):
    try:
        source = sky_model[source_type]
    except KeyError:
        raise ValueError("Source type '%s' unsupported" % source_type)

    # Select single dataset rows
    corrs = pol.NUM_CORR.data[0]
    frequency = spw.CHAN_FREQ.data[0]
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly

    lm = radec_to_lm(source.radec, phase_dir)
    uvw = -ms.UVW.data if args.invert_uvw else ms.UVW.data

    # (source, row, frequency)
    phase = phase_delay(lm, uvw, frequency)

    # (source, spi, corrs)
    # Apply spectral mode to stokes parameters
    stokes = spectral_model(source.stokes,
                            source.spi,
                            source.ref_freq,
                            frequency,
                            base=0)

    brightness = convert(stokes, ["I", "Q", "U", "V"],
                         corr_schema(pol))

    bl_jones_args = ["phase_delay", phase]

    # Add any visibility amplitude terms
    if source_type == "gauss":
        bl_jones_args.append("gauss_shape")
        bl_jones_args.append(gaussian_shape(uvw, frequency, source.shape))

    bl_jones_args.extend(["brightness", brightness])

    # Unique times and time index for each row chunk
    # The index is not global
    meta = np.empty((0,), dtype=tuple)
    utime_inv = ms.TIME.data.map_blocks(np.unique, return_inverse=True,
                                        meta=meta, dtype=tuple)

    # Need unique times for parallactic angles
    nan_chunks = (tuple(np.nan for _ in utime_inv.chunks[0]),)
    utime = utime_inv.map_blocks(getitem, 0,
                                 chunks=nan_chunks,
                                 dtype=ms.TIME.dtype)

    time_idx = utime_inv.map_blocks(getitem, 1, dtype=np.int32)

    jones = baseline_jones_multiply(corrs, *bl_jones_args)
    dde = dde_factory(args, ms, ant, field, pol, lm, utime, frequency)

    return predict_vis(time_idx, ms.ANTENNA1.data, ms.ANTENNA2.data,
                       dde, jones, dde, None, None, None)


@requires_optional("dask.array", "Tigger",
                   "daskms", opt_import_error)
def predict(args):
    # Convert source data into dask arrays
    sky_model = parse_sky_model(args.sky_model, args.model_chunks)

    # Get the support tables
    tables = support_tables(args)

    ant_ds = tables["ANTENNA"]
    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    # List of write operations
    writes = []

    # Construct a graph for each DATA_DESC_ID
    for xds in xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": args.row_chunks}):

        # Perform subtable joins
        ant = ant_ds[0]
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]

        # Select single dataset row out
        corrs = pol.NUM_CORR.data[0]

        # Generate visibility expressions for each source type
        source_vis = [vis_factory(args, stype, sky_model,
                                  xds, ant, field, spw, pol)
                      for stype in sky_model.keys()]

        # Sum visibilities together
        vis = sum(source_vis)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        xds = xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))
        # Create a write to the table
        write = xds_to_table(xds, args.ms, ['MODEL_DATA'])
        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    with ProgressBar():
        dask.compute(writes)


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
