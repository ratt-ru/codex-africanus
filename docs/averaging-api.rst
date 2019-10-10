---------
Averaging
---------

Routines for averaging visibility data.

Time and Channel Averaging
--------------------------

The routines in this section average row-based samples by:

1. Averaging samples of consecutive **time** values into bins defined
   by an period of :code:`time_bin_secs` seconds.
2. Averaging channel data into equally sized bins of :code:`chan_bin_size`.

In order to achieve this, a **baseline x time** ordering is established
over the input data where **baseline** corresponds to the
unique **(ANTENNA1, ANTENNA2)** pairs and **time** corresponds
to the unique, monotonically increasing **TIME** values
associated with the rows of a Measurement Set.

======== === === === === ===
Baseline T0  T1  T2  T3  T4
======== === === === === ===
(0, 0)   0.1 0.2 0.3 0.4 0.5
(0, 1)   0.1 0.2 0.3 0.4 0.5
(0, 2)   0.1 0.2  X  0.4 0.5
(1, 1)   0.1 0.2 0.3 0.4 0.5
(1, 2)   0.1 0.2 0.3 0.4 0.5
(2, 2)   0.1 0.2 0.3 0.4 0.5
======== === === === === ===

It is possible for times or baselines to be missing. In the above
example, T2 is missing for baseline (0, 2).

.. warning::

  The above requires unique lexicographical
  combinations of (TIME, ANTENNA1, ANTENNA2). This can usually
  be achieved by suitably partitioning input data on indexing rows,
  DATA_DESC_ID and SCAN_NUMBER in particular.


For each baseline, adjacent time's are assigned to a bin
if :math:`h_c - h_e/2 - (l_c - l_e/2) <` :code:`time_bin_secs`, where
:math:`h_c` and :math:`l_c` are the upper and lower time and
:math:`h_e` and :math:`l_e` are the upper and lower intervals,
taken from the **INTERVAL** column.
Note that no distinction is made between flagged and unflagged data
when establishing the endpoints in the bin.

The reason for this is that the `Measurement Set v2.0 Specification
<https://casa.nrao.edu/Memos/229.html>`_ specifies that
**TIME** and **INTERVAL** columns
are defined as containing the *nominal*
time and period at which the visibility was sampled.
This means that their values includie valid, flagged and missing data.
Thus, averaging a
regular high-resolution **baseline x htime** grid should produce
a regular low-resolution  **baseline x ltime** grid (**htime > ltime**)
in the presence of bad data

By contrast, other columns such as **TIME_CENTROID**
and **EXPOSURE** contain the *effective* time and period as
they exclude missing and bad data. Their increased accuracy,
and therefore variability means that they are unsuitable for
establishing a grid over the data.

To summarise, the averaged times in each bin establish a map:

- from possibly unordered input rows.
- to a reduced set of output rows ordered by
  averaged :code:`(TIME, ANTENNA1, ANTENNA2)`.

Flagged Data Handling
~~~~~~~~~~~~~~~~~~~~~

Both **FLAG_ROW** and **FLAG** columns may be supplied to the averager,
but they should be consistent with each other. The averager will throw
an exception if this is not the case, rather than making an assumption as
to which is correct.

When provided with flags, the averager will output averages
for bins that are completely flagged.

Part of the reason for this is that the  specifies that
the **TIME** and **INTERVAL** columns represent the *nominal* time and interval
values.
This means that they should represent valid as well as flagged or missing data
in their computation.

By contrast, most other columns such as **TIME_CENTROID** and **EXPOSURE**,
contain the *effective* values and should only include valid, unflagged data.

To support this:

1. **TIME** and **INTERVAL** are averaged using both flagged and
   unflagged samples.
2. Other columns, such as **TIME_CENTROID** are handled as follows:

   1. If the bin contains some unflagged data, only this data
      is used to calculate average.
   2. If the bin is completely flagged, the average of all samples
      (which are all flagged) will be used.

3. In both cases, a completely flagged bin will have it's flag set.
4. To support the two cases, twice the memory of the output array
   is required to track both averages, but only one array of merged
   values is returned.

Guarantees
~~~~~~~~~~

1. Averaged output data will be lexicographically ordered by
   :code:`(TIME, ANTENNA1, ANTENNA2)`
2. **TIME** and **INTERVAL** columns always contain the
   *nominal* average and sum and therefore contain both and missing
   or unflagged data.
3. Other columns will contain the *effective*
   average and will contain only valid data *except* when
   all data in the bin is flagged.
4. Completely flagged bins will be set as flagged in
   both the *nominal* and *effective* case.
5. Certain columns are averaged, while others are summed,
   or simply assigned to the last value in the bin in the case
   of antenna indices.
6. **Visibility data** is averaged by multiplying and dividing
   by **WEIGHT_SPECTRUM** or **WEIGHT** or natural weighting,
   in order of priority.

  .. math::

    \frac{\sum v_i w_i}{\sum w_i}

7. **SIGMA_SPECTRUM** is averaged by multiplying and dividing
   by **WEIGHT_SPECTRUM** or **WEIGHT** or natural weighting,
   in order of priority and availability.

   **SIGMA** is only averaged with **WEIGHT** or natural weighting.

  .. math::

    \sqrt{\frac{\sum w_i^2 \sigma_i^2}{(\sum w_i)^2}}


The following table summarizes the handling of each
column in the main Measurement Set table:

=============== ================= ============================ ===========
Column          Unflagged/Flagged Aggregation Method           Required
                sample handling
=============== ================= ============================ ===========
TIME            Nominal           Mean                         Yes
INTERVAL        Nominal           Sum                          Yes
ANTENNA1        Nominal           Assigned to Last Input       Yes
ANTENNA2        Nominal           Assigned to Last Input       Yes
TIME_CENTROID   Effective         Mean                         No
EXPOSURE        Effective         Sum                          No
FLAG_ROW        Effective         Set if All Inputs Flagged    No
UVW             Effective         Mean                         No
WEIGHT          Effective         Sum                          No
SIGMA           Effective         Weighted Mean                No
DATA (vis)      Effective         Weighted Mean                No
FLAG            Effective         Set if All Inputs Flagged    No
WEIGHT_SPECTRUM Effective         Sum                          No
SIGMA_SPECTRUM  Effective         Weighted Mean                No
=============== ================= ============================ ===========

The following SPECTRAL_WINDOW sub-table columns are averaged as follows:

=============== ============================
Column          Aggregation Method
=============== ============================
CHAN_FREQ       Mean
CHAN_WIDTH      Sum
EFFECTIVE_BW    Sum
RESOLUTION      Sum
=============== ============================

Dask Implementation
~~~~~~~~~~~~~~~~~~~

The dask implementation chunks data up by row and channel and
averages each chunk independently of values in other chunks. This should
be kept in mind if one wishes to maintain a particular ordering
in the output dask arrays.

Typically, Measurement Set data is monotonically ordered in time. To
maintain this guarantee in output dask arrays,
the chunks will need to be separated by distinct time values.
Practically speaking this means that the first and second chunk
should not both contain value time 0.1, for example.

Numpy
~~~~~

.. currentmodule:: africanus.averaging

.. autosummary::
    time_and_channel

.. autofunction:: time_and_channel


Dask
~~~~

.. currentmodule:: africanus.averaging.dask

.. autosummary::
    time_and_channel

.. autofunction:: time_and_channel

