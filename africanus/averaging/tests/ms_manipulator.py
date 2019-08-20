import sys
import os
import os.path
from copy import deepcopy

import numpy as np

# Look for a casacore library binding that will provide Table tools
try:
    # Try to use the casapy table tool first
    import casac
    tb = casac.homefinder.find_home_by_name('tableHome').create()
    casacore_binding = 'casapy'
except:
    try:
        # Otherwise fall back to pyrap
        from pyrap import tables
        casacore_binding = 'pyrap'
    except ImportError:
        casacore_binding = ''
    else:
        # Perform python-casacore version checks
        from pkg_resources import parse_version
        import casacore

        pyc_ver = parse_version(casacore.__version__)
        req_ver = parse_version("2.2.1")

        if not pyc_ver >= req_ver:
            raise ImportError("python-casacore %s is required, "
                              "but the current version is %s. "
                              "Note that python-casacore %s "
                              "requires at least casacore 2.3.0."
                                      % (req_ver, pyc_ver, req_ver))


def std_scalar(comment, valueType='integer', option=0, **kwargs):
    """Description for standard scalar column."""
    return dict(comment=comment, valueType=valueType, dataManagerType='StandardStMan',
                dataManagerGroup='StandardStMan', option=option, maxlen=0, **kwargs)


def std_array(comment, valueType, ndim, **kwargs):
    """Description for standard array column with variable shape (used for smaller arrays)."""
    return dict(comment=comment, valueType=valueType, ndim=ndim, dataManagerType='StandardStMan',
                dataManagerGroup='StandardStMan', _c_order=True, option=0, maxlen=0, **kwargs)


def fixed_array(comment, valueType, shape, **kwargs):
    """Description for direct array column with fixed shape (used for smaller arrays)."""
    return dict(comment=comment, valueType=valueType, shape=np.asarray(shape, dtype=np.int32), ndim=len(shape),
                dataManagerType='StandardStMan', dataManagerGroup='StandardStMan',
                _c_order=True, option=5, maxlen=0, **kwargs)


def tiled_array(comment, valueType, ndim, dataManagerGroup, **kwargs):
    """Description for array column with tiled storage manager (used for bigger arrays)."""
    return dict(comment=comment, valueType=valueType, ndim=ndim, dataManagerType='TiledShapeStMan',
                dataManagerGroup=dataManagerGroup, _c_order=True, option=0, maxlen=0, **kwargs)


def define_hypercolumn(desc):
    """Add hypercolumn definitions to table description."""
    desc['_define_hypercolumn_'] = dict([(v['dataManagerGroup'],
                                          dict(HCdatanames=[k], HCndim=v['ndim'] + 1))
                                         for k, v in desc.iteritems() if v['dataManagerType'] == 'TiledShapeStMan'])


# Map Measurement Set string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128
}


# Create a description of an array column
def makearrcoldesc(columnname, value, ndim=0,
                   shape=[], datamanagertype='',
                   datamanagergroup='',
                   options=0, maxlen=0, comment='',
                   valuetype='', keywords={}):
    """Create description of an array column.
    A description for a scalar column can be created from a name for
    the column and a data value, which is used only to determine the
    type of the column. Note that a dict value is also possible.
    It is possible to create the column description in more detail
    by giving the dimensionality, shape, data manager name, group, option,
    and comment as well.
    The data manager type tells which data manager (storage manager)
    is used to store the columns. The data manager type and group are
    explained in more detail in the `casacore Tables
    <../../casacore/doc/html/group__Tables__module.html>`_ documentation.
    It returns a dict with fields `name` and `desc` which can thereafter be used
    to build a table description using function :func:`maketabdesc`.
    `name`
      The name of the column.
    `value`
      A data value, which is only used to determine the data type of the column.
      It is only used if argument `valuetype` is not given.
    `ndim`
      Optionally the number of dimensions. A value > 0 means that all
      arrays in the column must have that dimensionality. Note that the
      arrays can still differ in shape unless the shape vector is also given.
    `shape`
      An optional sequence of integers giving the shape of the array in each
      cell. If given, it forces option FixedShape (see below) and sets the
      number of dimensions (if not given). All arrays in the column get the
      given shape and the array is created as soon as a row is added.
      Note that the shape vector gives the shape in each table cell; the
      number of rows in the table should NOT be part of it.
    `datamanagertype`
      Type of data manager which can be one of StandardStMan (default),
      IncrementalStMan, TiledColumnStMan, TiledCellStMan, or TiledShapeStMan.
      The tiled storage managers are usually used for bigger data arrays.
    `datamanagergroup`
      Data manager group. Only for the expert user.
    `options`
      Optionally numeric array options which can be added to combine them.
      `1` means Direct.
          It tells that the data are directly stored in the table. Direct
          forces option FixedShape. If not given, the array is indirect, which
          means that the data will be stored in a separate file.
      `4` means FixedShape.
          This option does not need to be given, because it is enforced if
          the shape is given. FixedShape means that the shape of the array must
          be the same in each cell of the column. Otherwise the array shapes may
          be different in each column cell and is it possible that a cell does
          not contain an array at all.
          Note that when given (or implicitly by option Direct), the
          shape argument must be given as well.
      Default is 0, thus indirect and variable shaped.
    `maxlen`
      Maximum length of string values in a column.
      Default 0 means unlimited.
    `comment`
      Comment: informational for user.
    `valuetype`
      A string giving the column's data type. Possible data types are
      bool (or boolean), uchar (or byte), short, int (or integer), uint,
      float, double, complex, dcomplex, and string.
    'keywords'
      A dict defining initial keywords for the column.
    For example::
      acd1= makescacoldesc("arr1", 1., 0, [2,3,4])
      td = maketabdesc(acd1)
    This creates a table description consisting of an array column `arr1`
    containing 3-dim arrays of doubles with shape [2,3,4].
    """
    vtype = valuetype
    if vtype == '':
        vtype = _value_type_name(value)
    if len(shape) > 0:
        if ndim <= 0:
            ndim = len(shape)
    rec2 = {'valueType': vtype,
            'dataManagerType': datamanagertype,
            'dataManagerGroup': datamanagergroup,
            'ndim': ndim,
            'shape': shape,
            '_c_order': True,
            'option': options,
            'maxlen': maxlen,
            'comment': comment,
            'keywords': keywords}
    return {'name': columnname,
            'desc': rec2}
def kat_ms_desc_and_dminfo(nbl, model_data=False):
    """
    Creates Table Description and Data Manager Information objecs that
    describe a MeasurementSet suitable for holding MeerKAT data.
    Creates additional DATA, IMAGING_WEIGHT and possibly
    MODEL_DATA and CORRECTED_DATA columns.
    Columns are given fixed shapes defined by the arguments to this function.
    :param nbl: Number of baselines.
    :param nchan: Number of channels.
    :param ncorr: Number of correlations.
    :param model_data: Boolean indicated whether MODEL_DATA and CORRECTED_DATA
                        should be added to the Measurement Set.
    :return: Returns a tuple containing a table description describing
            the extra columns and hypercolumns, as well as a Data Manager
            description.
    """

    if not casacore_binding == 'pyrap':
        raise ValueError("kat_ms_desc_and_dminfo requires the "
                        "casacore binding to operate")

    # Columns that will be modified.
    # We want to keep things like their
    # keywords, dims and shapes
    modify_columns = {"UVW", "ANTENNA1", "ANTENNA2"}

    # Get the required table descriptor for an MS
    table_desc = tables.required_ms_desc("MAIN")

    # Take columns we wish to modify
    extra_table_desc = {c: d for c, d in table_desc.iteritems()
                                        if c in modify_columns}

    # Used to set the SPEC for each Data Manager Group
    dmgroup_spec = {}

    def dmspec(coldesc, tile_mem_limit=None):
        """
        Create data manager spec for a given column description,
        mostly by adding a DEFAULTTILESHAPE that fits into the
        supplied memory limit.
        """

        # Choose 4MB if none given
        if tile_mem_limit is None:
            tile_mem_limit = 4*1024*1024

        # Get the reversed column shape. DEFAULTTILESHAPE is deep in
        # casacore and its necessary to specify their ordering here
        # ntilerows is the dim that will change least quickly
        rev_shape = list(reversed(coldesc["shape"]))

        ntilerows = 1
        np_dtype = MS_TO_NP_TYPE_MAP[coldesc["valueType"].upper()]
        nbytes = np.dtype(np_dtype).itemsize

        # Try bump up the number of rows in our tiles while they're
        # below the memory limit for the tile
        while np.product(rev_shape + [2*ntilerows])*nbytes < tile_mem_limit:
            ntilerows *= 2

        return {"DEFAULTTILESHAPE": np.int32(rev_shape + [ntilerows])}

    # Update existing columns with shape and data manager information
    dm_group = 'UVW'
    shape = [3]
    extra_table_desc["UVW"].update(options=0,
        shape=shape, ndim=len(shape),
        dataManagerGroup=dm_group,
        dataManagerType='TiledColumnStMan')
    dmgroup_spec[dm_group] = dmspec(extra_table_desc["UVW"])

    # Create new columns for integration into the MS
    additional_columns = []

    dm_group = 'Weight'
    shape = []
    desc = tables.makearrcoldesc("WEIGHT", 0.,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='float',
            comment="The weight Column")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    dm_group = 'Sigma'
    shape = []
    desc = tables.makearrcoldesc("SIGMA", 0.,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='float',
            comment="The sigma Column")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    dm_group = 'Flag'
    shape = []
    desc = tables.makearrcoldesc("FLAG", 1,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='bool',
            comment="The flag Column")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    dm_group = 'FlagCategory'
    shape = []
    desc = tables.makearrcoldesc("FLAG_CATEGORY", 1,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='bool',
            comment="The flag_category Column")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    dm_group = 'Data'
    shape = []
    desc = tables.makearrcoldesc("DATA", 0.+0j,
	    ndim=len(shape),
	    shape=shape,
	    options=0,
	    maxlen=0,
	    datamanagertype='StandardStMan',
	    datamanagergroup=dm_group,
	    valuetype='complex',
	    keywords={"UNIT": "Jy"},
	    comment="The Visibility DATA Column")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    dm_group = 'ImagingWeight'
    shape = []
    desc = tables.makearrcoldesc("IMAGING_WEIGHT", 0.,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='float',
            comment="Weight set by imaging task (e.g. uniform weighting)")
    #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
    additional_columns.append(desc)

    # Add MODEL_DATA and CORRECTED_DATA if requested
    if model_data == True:
        dm_group = 'ModelData'
        shape = []
        desc = tables.makearrcoldesc("MODEL_DATA", 0.+0j,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='complex',
            keywords={"UNIT": "Jy"},
            comment="The Visibility MODEL_DATA Column")
        #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
        additional_columns.append(desc)

        dm_group = 'CorrectedData'
        shape = []
        desc = tables.makearrcoldesc("CORRECTED_DATA", 0.+0j,
            ndim=len(shape),
            shape=shape,
            options=0,
            maxlen=0,
            datamanagertype='StandardStMan',
            datamanagergroup=dm_group,
            valuetype='complex',
            keywords={"UNIT": "Jy"},
            comment="The Visibility CORRECTED_DATA Column")
        #dmgroup_spec[dm_group] = dmspec(desc1["desc"])
        additional_columns.append(desc)

    # Update extra table description with additional columns
    extra_table_desc.update(tables.maketabdesc(additional_columns))

    # Update the original table descriptor with modifications/additions
    # Need this to construct a complete Data Manager specification
    # that includes the original columns
    table_desc.update(extra_table_desc)

    # Construct DataManager Specification
    dminfo = tables.makedminfo(table_desc, dmgroup_spec)

    return extra_table_desc, dminfo


caltable_desc = {}
caltable_desc['TIME'] = std_scalar('Timestamp of solution', 'double', option=5)
caltable_desc['FIELD_ID'] = std_scalar(
    'Unique id for this pointing', 'integer', option=5)
caltable_desc['SPECTRAL_WINDOW_ID'] = std_scalar(
    'Spectral window', 'integer', option=5)
caltable_desc['ANTENNA1'] = std_scalar(
    'ID of first antenna in interferometer', 'integer', option=5)
caltable_desc['ANTENNA2'] = std_scalar(
    'ID of second antenna in interferometer', 'integer', option=5)
caltable_desc['INTERVAL'] = std_scalar(
    'The effective integration time', 'double', option=5)
caltable_desc['SCAN_NUMBER'] = std_scalar('Scan number', 'integer', option=5)
caltable_desc['OBSERVATION_ID'] = std_scalar(
    'Observation id (index in OBSERVATION table)', 'integer', option=5)
caltable_desc['PARAMERR'] = std_array('Parameter error', 'float', -1)
caltable_desc['FLAG'] = std_array('Solution values', 'boolean', -1)
caltable_desc['SNR'] = std_array('Signal to noise ratio', 'float', -1)
caltable_desc['WEIGHT'] = std_array('Weight', 'float', -1)
# float version of caltable
caltable_desc_float = deepcopy(caltable_desc)
caltable_desc_float['FPARAM'] = std_array('Solution values', 'float', -1)
define_hypercolumn(caltable_desc_float)
# complex version of caltable
caltable_desc_complex = deepcopy(caltable_desc)
caltable_desc_complex['CPARAM'] = std_array('Solution values', 'complex', -1)
define_hypercolumn(caltable_desc_complex)

# Define the appropriate way to open a table using the selected binding
if casacore_binding == 'casapy':
    def open_table(filename, readonly=False, ack=False, **kwargs):
        success = tb.open(filename, nomodify=readonly, **kwargs)
        return tb if success else None

    def create_ms(filename, table_desc=None, dm_info=None):
        raise NotImplementedError("create_ms not implemented for casapy")

elif casacore_binding == 'pyrap':
    def open_table(filename, readonly=False, ack=False, **kwargs):
        t = tables.table(filename, readonly=readonly, ack=ack, **kwargs)

        return t if type(t) == tables.table else None

    def create_ms(filename, table_desc=None, dm_info=None):
        with tables.default_ms(filename, table_desc, dm_info) as T:
            # Add the SOURCE subtable
            source_filename = os.path.join(os.getcwd(), filename, "SOURCE")
            tables.default_ms_subtable("SOURCE", source_filename)
            T.putkeyword("SOURCE", "Table: %s" % source_filename)

else:
    def open_table(filename, readonly=False):
        raise NotImplementedError("Cannot open MS '%s', as neither "
                                    "casapy nor pyrap were found" % (filename,))

    def create_ms(filename, table_desc=None, dm_info=None):
        raise NotImplementedError("Cannot create MS '%s', as neither "
                                    "casapy nor pyrap were found" % (filename,))


# -------- Routines that create MS data structures in dictionaries -----------

def populate_main_dict_time(ms_name, scan, desc, uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid, vis_data, weights, verbose=False):
    """
    dexplain before distribution

    """
    main_dict = {}

    main_dict = deep_copy_sub_tables(ms_name, bda_freq=False, verbose=verbose)
    nbr_vis, nbr_freqs, nbr_corr = vis_data.shape;
    # timestamps = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
    sub_dict = {}
    # ID of first antenna in interferometer (integer)
    sub_dict['ANTENNA1'] = A0
    # ID of second antenna in interferometer (integer)
    sub_dict['ANTENNA2'] = A1
    # The data column (complex, 3-dim)
    sub_dict['DATA'] = vis_data
    # The data description table index (integer)
    sub_dict['DATA_DESC_ID'] = desc
    # The effective integration time (double)
    sub_dict['EXPOSURE'] = exposure
    # The data flags, array of bools with same shape as data
    sub_dict['FLAG'] = flag
    # Row flag - flag all data in this row if True (boolean)
    sub_dict['FLAG_ROW'] = flag_row
    # The sampling interval (double)
    sub_dict['INTERVAL'] = interval
    # The model data column (complex, 3-dim)
    # Modified Julian Dates in seconds (double)
    sub_dict['TIME'] = timestamps
    # Modified Julian Dates in seconds (double)
    sub_dict['TIME_CENTROID'] = time_centroid
    # Vector with uvw coordinates (in metres) (double, 1-dim, shape=(3,))
    sub_dict['UVW'] = uvw
    # Weight for each polarisation spectrum (float, 1-dim)
    sub_dict['WEIGHT'] = weights
    # Scan tables
    sub_dict['SCAN_NUMBER'] = scan
    main_dict['MAIN'] = sub_dict
    return main_dict


def populate_main_dict_freq(ms_name, freq_dict, verbose=False):
    """
    dexplain before distribution

    """
    main_dict = {}

    main_dict = deep_copy_sub_tables(ms_name, bda_freq=True, verbose=verbose)
    # prepare all tables into a main table
    # initialise output arrays
    sub_dict = {}
    sub_dict_visflag = {}
    sub_dict['SCAN_NUMBER'] = sub_dict['DATA_DESC_ID'] = sub_dict['ANTENNA1'] = sub_dict['ANTENNA2'] = np.array([], dtype=np.int32)

    sub_dict['FLAG_ROW'] = flag = np.array([], dtype=np.bool)
    sub_dict['UVW'] = sub_dict['INTERVAL'] = sub_dict['EXPOSURE'] = sub_dict['TIME'] = sub_dict['TIME_CENTROID'] = sub_dict['WEIGHT'] = np.array([
    ])
    # prepare scan indexes
    nrow_scan = 0
    data = []
    flag = []
    for sc, sub_freq_dict in freq_dict.iteritems():
        spw = 0;
        for nbr_bda_vis in sub_freq_dict:
        # ID of first antenna in interferometer (integer)

            sub_dict['ANTENNA1'] = np.append(sub_dict['ANTENNA1'], sub_freq_dict[nbr_bda_vis]['A0'], axis=0)
            sub_dict['ANTENNA2'] = np.append(sub_dict['ANTENNA2'], sub_freq_dict[nbr_bda_vis]['A1'], axis=0)
            sub_dict['FLAG_ROW'] = np.append(sub_dict['FLAG_ROW'], sub_freq_dict[nbr_bda_vis]['FLAG_ROW'], axis=0)
            sub_dict['UVW'] = np.append(sub_dict['UVW'], sub_freq_dict[nbr_bda_vis]['UVW'], axis=0) if sub_dict['UVW'].size else sub_freq_dict[nbr_bda_vis]['UVW']
            sub_dict['INTERVAL'] = np.append(sub_dict['INTERVAL'], sub_freq_dict[nbr_bda_vis]['INTERVAL'], axis=0)
            sub_dict['EXPOSURE'] = np.append(sub_dict['EXPOSURE'], sub_freq_dict[nbr_bda_vis]['EXPOSURE'], axis=    0)
            sub_dict['TIME'] = np.append(sub_dict['TIME'], sub_freq_dict[nbr_bda_vis]['TIME'], axis=0)
            sub_dict['TIME_CENTROID'] = np.append(sub_dict['TIME_CENTROID'], sub_freq_dict[nbr_bda_vis]['TIME_CENTROID'], axis=0)
            sub_dict['WEIGHT'] = np.append(sub_dict['WEIGHT'], sub_freq_dict[nbr_bda_vis]['WEIGHT'], axis=0) if sub_dict['WEIGHT'].size else sub_freq_dict[nbr_bda_vis]['WEIGHT']
            sub_dict['DATA_DESC_ID'] = np.append(sub_dict['DATA_DESC_ID'], np.ones_like(sub_freq_dict[nbr_bda_vis]['A0'])*spw,  axis=0)
            spw = spw +1
            # keep variable frequency shape array in dictionary
            data.append(sub_freq_dict[nbr_bda_vis]['DATA'])
            flag.append(sub_freq_dict[nbr_bda_vis]['FLAG'])
        sub_dict['SCAN_NUMBER'] = np.append(sub_dict['SCAN_NUMBER'], np.ones(len(sub_dict['ANTENNA1'])-nrow_scan)*sc, axis=0)
        nrow_scan = len(sub_dict['ANTENNA1'])
    sub_dict['DATA'] = data
    sub_dict['FLAG'] = flag
    main_dict['MAIN'] = sub_dict;
    return main_dict;


def deep_copy_sub_tables(ms_name,  bda_freq=False, verbose=False):
    """ explain this piece of code before distribution

    """
    t = open_main(ms_name, verbose)
    sub_tables = t.keywordnames()
    #if bda_freq:
    #	sub_tables.remove('SPECTRAL_WINDOW')
    #	sub_tables.remove('DATA_DESCRIPTION')
    sub_dict = {}
    for sub_table_name in sub_tables:
        try:
            t2 = open_table(t.getkeyword(sub_table_name), ack=verbose);
            cols = t2.colnames()
            cols_dict = {}
            for col_name in cols:
                try:
                    cols_dict[col_name] = t2.getcol(col_name)
                except RuntimeError, err:
                    print "  error writing column '%s' (%s)" %(col_name, err)
            sub_dict[sub_table_name] = cols_dict
        except TypeError, err:
            print "  error creating sub table '%s'  (%s)" % (sub_table_name, err)

    return sub_dict


#def populate_main_dict_bd_freq(output_bd_freq):
#    """
#    """


	     #subtab_dict['CHAN_WIDTH'] =  np.append(
             #               bd_freq_dict[cf]['CHAN_WIDTH'], [chan_width], axis=0)
             #subtab_dict['NUM_CHAN'] =  bd_freq_dict[cf]['NUM_CHAN']+vis_data.shape[1]
	     #subtab_dict['TOTAL_BANDWIDTH'] =  bd_freq_dict[cf]['TOTAL_BANDWIDTH']+chan_width*vis_data.shape[1]
	     #subtab_dict['RESOLUTION'] =  np.append(
             #               bd_freq_dict[cf]['RESOLUTION'], [chan_width], axis=0)
             #subtab_dict['EFFECTIVE_BW'] =  np.append(
             #               bd_freq_dict[cf]['EFFECTIVE_BW'], [chan_width], axis=0)
	     #subtab_dict['CHAN_FREQ'] =  np.append(
             #               bd_freq_dict[cf]['CHAN_FREQ'], [bd_freq_dict[cf]['CHAN_FREQ'][-1]+chan_width], axis=0)



	      #subtab_dict['CHAN_WIDTH'] = np.ones(1, dtype=np.float64)*chan_width
	      #subtab_dict['NUM_CHAN'] = vis_data.shape[1]
	      #subtab_dict['TOTAL_BANDWIDTH'] = chan_width*vis_data.shape[1]
	      #subtab_dict['RESOLUTION'] = np.ones(1, dtype=np.float64)*chan_width
	      #subtab_dict['EFFECTIVE_BW'] = np.ones(1, dtype=np.float64)*chan_width
	      #subtab_dict['CHAN_FREQ'] = np.ones(1, dtype=np.float64)*(freq_obs+chan_width/2.)


#    return bd_freq_dict

#def populate_ms_dict_bd_time(ms_name, scan, desc, uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid,
 #                      				vis_data, weights, verbose=False):
    #"""
    #explain before distribution
    #"""

    #ms_dict = deep_copy_sub_tables(ms_name, bda_freq=False, verbose=verbose)
    #ms_dict['MAIN'] = populate_main_dict(scan, desc, uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid,
#                       				vis_data, weights)

    #return ms_dict

#def populate_ms_dict_bd_freq(ms_name, freq_obs, output_bd_freq, verbose=False):
#    """
#    """
#    ms_dict = deep_copy_sub_tables(ms_name, bda_freq=True, verbose=verbose)
#    ms_dict['MAIN'] = populate_main_dict_bd_freq(output_bd_freq)

#    return ms_dict


# ----------------- Write completed dictionary to MS file --------------------


def open_main(ms_name, verbose=True):
    t = open_table(ms_name, ack=verbose)
    if t is None:
        print "Failed to open main table for writing."
        sys.exit(1)
    return t


def write_rows_bd_time(t, row_dict, bda_freq=True, verbose=True):
    # must not be empty before trying to add
    if row_dict.values()[0] is not None:
    	num_rows = np.array(row_dict.values()[0]).shape[0]
    	# Append rows to the table by starting after the last row in table
    	startrow = t.nrows()
    	# Add the space required for this group of rows
    	t.addrows(num_rows)
    	if verbose:
            print "  added %d rows" % (num_rows,)
    	for col_name, col_data in row_dict.iteritems():
             if col_name in t.colnames():
                 if bda_freq:
		    if col_name=="DATA" or col_name=="FLAG":
		       startrow_col = 0;
                       for row_data in col_data:
			   nrow_col = row_data.shape[0]
			   t.putcol(columnname=col_name, value=row_data.T if casacore_binding == 'casapy' else row_data, startrow=startrow_col, nrow=nrow_col)
			   startrow_col = startrow_col + nrow_col;
			   if verbose:
                              print "  wrote column '%s' with shape %s" % (col_name, row_data.shape)
                    else:
			try:
                            t.putcol(col_name, col_data.T if casacore_binding == 'casapy' else col_data, startrow)
                            if verbose:
                               print "  wrote column '%s' with shape %s" % (col_name, np.array(col_data).shape)
                        except RuntimeError, err:
                             print " here  error writing column '%s' with shape %s (%s)" % (col_name, np.array(col_data).shape, err)

                 else:
             	     try:
                         t.putcol(col_name, col_data.T if casacore_binding == 'casapy' else col_data, startrow)
                         if verbose:
                            print "  wrote column '%s' with shape %s" % (col_name, np.array(col_data).shape)
             	     except RuntimeError, err:
                          print " hereXX error writing column '%s' with shape %s (%s)" % (col_name, np.array(col_data).shape, err)
             else:
                if verbose:
                     print "  column '%s' not in table" % (col_name,)


#def write_rows_bd_freq(ms_dict, ms_name, verbose=True):
#    """
#    """
#    # iterate over scans
#    for scan_id, sub_dict in ms_dict.iteritems():
#	if isinstance(sub_dict, dict):
#            sub_dict = [sub_dict]




def write_dict(ms_dict, ms_name, bda_freq=True, verbose=True):
    # Iterate through subtables
    for sub_table_name, sub_dict in ms_dict.iteritems():
        # Allow parsing of single dict and array of dicts in the same fashion
        if isinstance(sub_dict, dict):
            sub_dict = [sub_dict]
        # Iterate through row groups that are separate dicts within the sub_dict array
        for row_dict in sub_dict:
            if verbose:
                print "Table %s:" % (sub_table_name,)
            # Open table using whichever casacore library was found
            t = open_table(ms_name, ack=verbose) if sub_table_name == 'MAIN' else \
                open_table(os.path.join(ms_name, sub_table_name))
            if verbose and t is not None:
                print "  opened successfully"
            if t is None:
                print "  could not open table!"
                break
            write_rows_bd_time(t, row_dict, bda_freq, verbose)
            t.close()
            if verbose:
                print "  closed successfully"


#def write_rows_bd_freq(t, row_dict, verbose=True):
#    """
#    """
