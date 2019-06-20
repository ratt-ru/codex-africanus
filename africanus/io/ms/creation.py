# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numpy as np

try:
    import pyrap.tables as pt
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.util.requirements import requires_optional

# Map Measurement Set string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128
}


def _dm_spec(coldesc, tile_mem_limit=4*1024*1024):
    """
    Create data manager spec for a given column description,
    by adding a DEFAULTTILESHAPE that fits within the memory limit
    """

    # Get the reversed column shape. DEFAULTTILESHAPE is deep in
    # casacore and its necessary to specify their ordering here
    # ntilerows is the dim that will change least quickly
    rev_shape = list(reversed(coldesc["shape"]))

    ntilerows = 1
    np_dtype = MS_TO_NP_TYPE_MAP[coldesc["valueType"].upper()]
    nbytes = np.dtype(np_dtype).itemsize

    # Try bump up the number of rows in our tiles while they're
    # below the memory limit for the tile
    while reduce(mul, rev_shape + [2*ntilerows], nbytes) < tile_mem_limit:
        ntilerows *= 2

    return {"DEFAULTTILESHAPE": np.int32(rev_shape + [ntilerows])}


def _ms_desc_and_dm_info(nchan, ncorr, add_imaging_cols=False):
    """
    Creates Table Description and Data Manager Information objects that
    describe a MeasurementSet.

    Creates additional DATA, IMAGING_WEIGHT and possibly
    MODEL_DATA and CORRECTED_DATA columns.

    Columns are given fixed shapes defined by the arguments to this function.

    Parameters
    ----------
    nchan : int
        Nimber of channels
    ncorr : int
        Number of correlations
    add_imaging_cols : bool, optional
        Add imaging columns. Defaults to False.

    Returns
    -------
    table_spec : dict
        Table specification dictionary
    dm_info : dict
        Data Manager Information dictionary
    """

    # Columns that will be modified.
    # We want to keep things like their keywords,
    # but modify their shapes, dimensions, options and data managers
    modify_columns = {"WEIGHT", "SIGMA", "FLAG", "FLAG_CATEGORY",
                      "UVW", "ANTENNA1", "ANTENNA2"}

    # Get the required table descriptor for an MS
    table_desc = pt.required_ms_desc("MAIN")

    # Take columns we wish to modify
    extra_table_desc = {c: d for c, d in table_desc.items()
                        if c in modify_columns}

    # Used to set the SPEC for each Data Manager Group
    dmgroup_spec = {}

    # Update existing columns with shape and data manager information
    dm_group = 'UVW'
    shape = [3]
    extra_table_desc["UVW"].update(options=0, shape=shape, ndim=len(shape),
                                   dataManagerGroup=dm_group,
                                   dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(extra_table_desc["UVW"])

    dm_group = 'Weight'
    shape = [ncorr]
    extra_table_desc["WEIGHT"].update(options=4, shape=shape, ndim=len(shape),
                                      dataManagerGroup=dm_group,
                                      dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(extra_table_desc["WEIGHT"])

    dm_group = 'Sigma'
    shape = [ncorr]
    extra_table_desc["SIGMA"].update(options=4, shape=shape, ndim=len(shape),
                                     dataManagerGroup=dm_group,
                                     dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(extra_table_desc["SIGMA"])

    dm_group = 'Flag'
    shape = [nchan, ncorr]
    extra_table_desc["FLAG"].update(options=4, shape=shape, ndim=len(shape),
                                    dataManagerGroup=dm_group,
                                    dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(extra_table_desc["FLAG"])

    dm_group = 'FlagCategory'
    shape = [1, nchan, ncorr]
    extra_table_desc["FLAG_CATEGORY"].update(
                                    options=4, keywords={},
                                    shape=shape, ndim=len(shape),
                                    dataManagerGroup=dm_group,
                                    dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(extra_table_desc["FLAG_CATEGORY"])

    # Create new columns for integration into the MS
    additional_columns = []

    dm_group = 'Data'
    shape = [nchan, ncorr]
    desc = pt.tablecreatearraycoldesc(
        "DATA", 0+0j, comment="The Visibility DATA Column",
        options=4, valuetype='complex', keywords={"UNIT": "Jy"},
        shape=shape, ndim=len(shape), datamanagergroup=dm_group,
        datamanagertype='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(desc["desc"])
    additional_columns.append(desc)

    dm_group = 'WeightSpectrum'
    shape = [nchan, ncorr]
    desc = pt.tablecreatearraycoldesc(
        "WEIGHT_SPECTRUM", 1.0, comment="Per-channel weights",
        options=4, valuetype='float', shape=shape, ndim=len(shape),
        datamanagergroup=dm_group, datamanagertype='TiledColumnStMan')
    dmgroup_spec[dm_group] = _dm_spec(desc["desc"])
    additional_columns.append(desc)

    # Add Imaging Columns, if requested
    if add_imaging_cols:
        dm_group = 'ImagingWeight'
        shape = [nchan]
        desc = pt.tablecreatearraycoldesc(
            "IMAGING_WEIGHT", 0,
            comment="Weight set by imaging task (e.g. uniform weighting)",
            options=4, valuetype='float', shape=shape, ndim=len(shape),
            datamanagergroup=dm_group, datamanagertype='TiledColumnStMan')
        dmgroup_spec[dm_group] = _dm_spec(desc["desc"])
        additional_columns.append(desc)

        dm_group = 'ModelData'
        shape = [nchan, ncorr]
        desc = pt.tablecreatearraycoldesc(
            "MODEL_DATA", 0+0j, comment="The Visibility MODEL_DATA Column",
            options=4, valuetype='complex', keywords={"UNIT": "Jy"},
            shape=shape, ndim=len(shape), datamanagergroup=dm_group,
            datamanagertype='TiledColumnStMan')
        dmgroup_spec[dm_group] = _dm_spec(desc["desc"])
        additional_columns.append(desc)

        dm_group = 'CorrectedData'
        shape = [nchan, ncorr]
        desc = pt.tablecreatearraycoldesc(
            "CORRECTED_DATA", 0+0j,
            comment="The Visibility CORRECTED_DATA Column",
            options=4, valuetype='complex', keywords={"UNIT": "Jy"},
            shape=shape, ndim=len(shape), datamanagergroup=dm_group,
            datamanagertype='TiledColumnStMan')
        dmgroup_spec[dm_group] = _dm_spec(desc["desc"])
        additional_columns.append(desc)

    # Update extra table description with additional columns
    extra_table_desc.update(pt.maketabdesc(additional_columns))

    # Update the original table descriptor with modifications/additions
    # Need this to construct a complete Data Manager specification
    # that includes the original columns
    table_desc.update(extra_table_desc)

    # Construct DataManager Specification
    dminfo = pt.makedminfo(table_desc, dmgroup_spec)

    return extra_table_desc, dminfo


@requires_optional("pyrap.tables", opt_import_error)
def empty_ms(ms_name, nchan, ncorr, add_imaging_cols):
    """
    Creates an empty Measurement Set with Fixed Column Shapes.
    Unlikely to work with multiple SPECTRAL_WINDOW's with
    different shapes, or multiple POLARIZATIONS
    with different correlations.

    Interface likely to change to somehow support this in future.

    Parameters
    ----------
    ms_name : str
        Measurement Set filename
    nchan : int
        Number of channels
    ncorr : int
        Number of correlations
    add_imaging_cols : bool
        True if ``MODEL_DATA``, ``CORRECTED_DATA`` and ``IMAGING_WEIGHTS``
        columns should be added.
    """

    table_desc, dm_info = _ms_desc_and_dm_info(nchan, ncorr, add_imaging_cols)

    with pt.default_ms(ms_name, table_desc, dm_info):
        pass
