#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from collections import namedtuple
from operator import getitem

import numpy as np

try:
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import Tigger
    from daskms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import (
    predict_vis,
    parallactic_angles,
    zernike_dde,
)
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.model.shape.dask import shapelet as shapelet_fn
from africanus.util.requirements import requires_optional

import packratt


_einsum_corr_indices = "ijkl"


def _brightness_schema(corrs, index):
    if corrs == 4:
        return "sf" + _einsum_corr_indices[index: index + 2], index + 1
    else:
        return "sfi", index


def _phase_delay_schema(corrs, index):
    return "srf", index


def _spi_schema(corrs, index):
    return "s", index


def _gauss_shape_schema(corrs, index):
    return "srf", index


def _shapelet_schema(corrs, index):
    return "rfs", index


def _bl_jones_output_schema(corrs, index):
    if corrs == 4:
        return "->srfi" + _einsum_corr_indices[index]
    else:
        return "->srfi"


_rime_term_map = {
    "brightness": _brightness_schema,
    "phase_delay": _phase_delay_schema,
    "spi": _spi_schema,
    "gauss_shape": _gauss_shape_schema,
    "shapelet_shape": _shapelet_schema,
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
        return [
            [corr_types[0], corr_types[1]],
            [corr_types[2], corr_types[3]],
        ]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]  # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]  # (1, ) shape
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
                raise ValueError(
                    "%s len(%s) == %d != %s.ndim"
                    % (name, einsum_schema, len(einsum_schema), array.shape)
                )

    output_schema = _bl_jones_output_schema(corrs, corr_index)
    schema = ",".join(input_einsum_schemas) + output_schema

    return da.einsum(schema, *arrays)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-sm", "--sky-model", default="sky-model-shapelets.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-mc", "--model-chunks", type=int, default=10)
    p.add_argument("-b", "--beam", default=None)
    p.add_argument(
        "-iuvw",
        "--invert-uvw",
        action="store_true",
        help="Invert UVW coordinates. Useful if we want "
        "compare our visibilities against MeqTrees",
    )
    p.add_argument("-z", "--zernike", action="store_true")
    p.add_argument("-dc", "--data-column", type=str, default="MODEL_DATA")
    return p


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

    shapelet_radec = []
    shapelet_stokes = []
    shapelet_spi = []
    shapelet_ref_freq = []
    shapelet_beta = []
    shapelet_coeffs = []

    for source in sky_model.sources:
        ra = source.pos.ra
        dec = source.pos.dec
        typecode = source.typecode.lower()

        I = source.flux.I  # noqa
        Q = source.flux.Q
        U = source.flux.U
        V = source.flux.V

        spectrum = (
            getattr(source, "spectrum", _empty_spectrum) or _empty_spectrum
        )
        try:
            # Extract reference frequency
            ref_freq = spectrum.freq0
        except AttributeError:
            ref_freq = sky_model.freq0

        try:
            # Extract SPI for I.
            # Zero Q, U and V to get 1 on the exponential
            spi = [[spectrum.spi, 0, 0, 0]]
        except AttributeError:
            # Default I SPI to -0.7
            spi = [[-0.7, 0, 0, 0]]

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

        elif typecode == "sha":
            beta_l = source.shape.sbetal
            beta_m = source.shape.sbetam
            coeffs = source.shape.shapelet_coeffs

            shapelet_radec.append([ra, dec])
            shapelet_stokes.append([I, Q, U, V])
            shapelet_spi.append(spi)
            shapelet_ref_freq.append(ref_freq)
            shapelet_beta.append([beta_l, beta_m])
            shapelet_coeffs.append(np.array(coeffs))
        else:
            raise ValueError("Unknown source morphology %s" % typecode)

    Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
    Gauss = namedtuple(
        "Gauss", ["radec", "stokes", "spi", "ref_freq", "shape"]
    )
    Shapelet = namedtuple(
        "Shapelet", ["radec", "stokes", "spi", "ref_freq", "beta", "coeffs"]
    )

    source_data = {}

    if len(point_radec) > 0:
        source_data["point"] = Point(
            da.from_array(point_radec, chunks=(chunks, -1)),
            da.from_array(point_stokes, chunks=(chunks, -1)),
            da.from_array(point_spi, chunks=(chunks, 1, -1)),
            da.from_array(point_ref_freq, chunks=chunks),
        )
    if len(gauss_radec) > 0:
        source_data["gauss"] = Gauss(
            da.from_array(gauss_radec, chunks=(chunks, -1)),
            da.from_array(gauss_stokes, chunks=(chunks, -1)),
            da.from_array(gauss_spi, chunks=(chunks, 1, -1)),
            da.from_array(gauss_ref_freq, chunks=chunks),
            da.from_array(gauss_shape, chunks=(chunks, -1)),
        )
    if len(shapelet_radec) > 0:
        source_data["shapelet"] = Shapelet(
            da.from_array(shapelet_radec, chunks=(chunks, -1)),
            da.from_array(shapelet_stokes, chunks=(chunks, -1)),
            da.from_array(shapelet_spi, chunks=(chunks, 1, -1)),
            da.from_array(shapelet_ref_freq, chunks=(chunks)),
            da.from_array(shapelet_beta, chunks=(chunks, -1)),
            da.from_array(shapelet_coeffs, chunks=(chunks, 1, -1)),
        )

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

    n = {
        k: "::".join((args.ms, k))
        for k in (
            "ANTENNA",
            "DATA_DESCRIPTION",
            "FIELD",
            "SPECTRAL_WINDOW",
            "POLARIZATION",
        )
    }

    # All rows at once
    lazy_tables = {"ANTENNA": xds_from_table(n["ANTENNA"])}

    compute_tables = {
        # Fixed shape rows
        "DATA_DESCRIPTION": xds_from_table(n["DATA_DESCRIPTION"]),
        # Variably shaped, need a dataset per row
        "FIELD": xds_from_table(n["FIELD"], group_cols="__row__"),
        "SPECTRAL_WINDOW": xds_from_table(
            n["SPECTRAL_WINDOW"], group_cols="__row__"
        ),
        "POLARIZATION": xds_from_table(
            n["POLARIZATION"], group_cols="__row__"
        ),
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


def zernike_factory(
    args, ms, ant, field, pol, lm, utime, frequency, jon, nrow=None
):
    """ Generate a primary beam DDE using Zernike polynomials """
    if not args.zernike:
        return None

    # Extract coefficient lengths for beam
    nsrc = lm.shape[0]
    utime = utime.compute()
    ntime = len(utime)
    utime = utime[:ntime]
    na = ant.coords.get("ROWID").data.shape[0]  # np.max(ant['row'].data) +1
    nbl = na * (na - 1) / 2
    ntime = int(nrow // nbl)
    nchan = len(frequency)
    npoly = 20

    # Make sure row_chunks and time_chunks are compatible
    n_row_chunks = len(jon)
    time_chunk_size = ntime // n_row_chunks
    time_chunks = (time_chunk_size,) * (n_row_chunks - 1)
    time_chunks = time_chunks + (ntime - sum(time_chunks),)

    # Create inputs to Zernike call
    zernike_coords = np.empty((3, nsrc, ntime, na, nchan))
    coeffs_r = np.empty((na, nchan, 2, 2, npoly))
    coeffs_i = np.empty((na, nchan, 2, 2, npoly))
    noll_index_r = np.empty((na, nchan, 2, 2, npoly))
    noll_index_i = np.empty((na, nchan, 2, 2, npoly))
    frequency_scaling = da.from_array(np.ones((nchan,)), chunks=(nchan,))
    pointing_errors = da.from_array(
        np.zeros((ntime, na, nchan, 2)), chunks=(time_chunks, na, nchan, 2)
    )
    antenna_scaling = da.from_array(
        np.ones((na, nchan, 2)), chunks=(na, nchan, 2)
    )
    parangles = da.from_array(
        parallactic_angles(
            np.array(utime)[:ntime],
            ant.POSITION.data,
            field.PHASE_DIR.data[0][0],
        ).compute(),
        chunks=(time_chunks, na),
    )

    # Convert coordinates to match a beam with a diameter of 10 degrees
    for src in range(nsrc):
        (
            zernike_coords[0, src, :, :, :],
            zernike_coords[1, src, :, :, :],
            zernike_coords[2, src, :, :, :],
        ) = (lm[src, 1] * 180 / np.pi / 5, lm[src, 0] * 180 / np.pi / 5, 0)

    # Load in Zernike coefficients for MeerKAT at L-Band
    packratt.get(
        "/beams/meerkat/meerkat_zernike_coeffs/meerkat/zernike_coeffs.tar.gz",
        "./",
    )
    params = np.load("./meerkat/params.npy", allow_pickle=True)

    # Assign coefficients
    for ant in range(na):
        for chan in range(nchan):
            coeffs_r[ant, chan, :, :, :] = params[chan, 0][0, :, :, :]
            coeffs_i[ant, chan, :, :, :] = params[chan, 0][1, :, :, :]
            noll_index_r[ant, chan, :, :, :] = params[chan, 1][0, :, :, :]
            noll_index_i[ant, chan, :, :, :] = params[chan, 1][1, :, :, :]

    # Call Zernike_dde
    dde_r = zernike_dde(
        da.from_array(
            zernike_coords, chunks=(3, nsrc, time_chunks, na, nchan)
        ),
        da.from_array(coeffs_r, chunks=coeffs_r.shape),
        da.from_array(noll_index_r, chunks=noll_index_r.shape),
        parangles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )
    dde_i = zernike_dde(
        da.from_array(
            zernike_coords, chunks=(3, nsrc, time_chunks, na, nchan)
        ),
        da.from_array(coeffs_i, chunks=coeffs_i.shape),
        da.from_array(noll_index_i, chunks=noll_index_i.shape),
        parangles,
        frequency_scaling,
        antenna_scaling,
        pointing_errors,
    )
    return dde_r + 1j * dde_i


def vis_factory(args, source_type, sky_model, ms, ant, field, spw, pol):
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

    # (source, spi, corrs)
    # Apply spectral mode to stokes parameters
    stokes = spectral_model(
        source.stokes,
        source.spi,
        source.ref_freq,
        frequency,
        base=[1, 0, 0, 0],
    )

    brightness = convert(stokes, ["I", "Q", "U", "V"], corr_schema(pol))
    bl_jones_args = []
    # Add any visibility amplitude terms
    if source_type == "gauss":
        bl_jones_args.append("gauss_shape")
        bl_jones_args.append(gaussian_shape(uvw, frequency, source.shape))
    elif source_type == "shapelet":
        bl_jones_args.append("shapelet_shape")
        s_fn = shapelet_fn(uvw, frequency, source.coeffs, source.beta)
        bl_jones_args.append(s_fn)

    bl_jones_args.extend(["brightness", brightness])

    # Unique times and time index for each row chunk
    # The index is not global
    meta = np.empty((0,), dtype=tuple)
    utime_inv = ms.TIME.data.map_blocks(
        np.unique, return_inverse=True, meta=meta, dtype=tuple
    )

    # Need unique times for parallactic angles
    nan_chunks = (tuple(np.nan for _ in utime_inv.chunks[0]),)
    utime = utime_inv.map_blocks(
        getitem, 0, chunks=nan_chunks, dtype=ms.TIME.dtype
    )

    time_idx = utime_inv.map_blocks(getitem, 1, dtype=np.int32)

    jones = baseline_jones_multiply(corrs, *bl_jones_args)

    dde = zernike_factory(
        args,
        ms,
        ant,
        field,
        pol,
        lm,
        utime,
        frequency,
        jones.chunks[1],
        nrow=time_idx.shape[0],
    )

    return predict_vis(
        time_idx,
        ms.ANTENNA1.data,
        ms.ANTENNA2.data,
        dde,
        jones,
        dde,
        None,
        None,
        None,
    )

    dde = zernike_factory(
        args,
        ms,
        ant,
        field,
        pol,
        lm,
        utime,
        frequency,
        jones.chunks[1],
        nrow=time_idx.shape[0],
    )

    return predict_vis(
        time_idx,
        ms.ANTENNA1.data,
        ms.ANTENNA2.data,
        dde,
        jones,
        dde,
        None,
        None,
        None,
    )


@requires_optional("dask.array", "Tigger", "daskms", opt_import_error)
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
    for xds in xds_from_ms(
        args.ms,
        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": args.row_chunks},
    ):

        # Perform subtable joins
        ant = ant_ds[0]
        field = field_ds[xds.attrs["FIELD_ID"]]
        ddid = ddid_ds[xds.attrs["DATA_DESC_ID"]]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]

        # Select single dataset row out
        corrs = pol.NUM_CORR.data[0]

        # Generate visibility expressions for each source type
        source_vis = [
            vis_factory(args, stype, sky_model, xds, ant, field, spw, pol)
            for stype in sky_model.keys()
        ]

        # Sum visibilities together
        vis = sum(source_vis)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        xds = (
            xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))
            if args.data_column == "MODEL_DATA"
            else xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), vis))
        )

        # Create a write to the table
        write = xds_to_table(xds, args.ms, [args.data_column])

        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    with ProgressBar():
        da.compute(writes)


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
