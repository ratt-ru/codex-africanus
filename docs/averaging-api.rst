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

For each baseline, adjacent time's are assigned to a bin
if :math:`h_c - h_e/2 - (l_c - l_e/2) <` :code:`time_bin_secs`, where
:math:`h_c` and :math:`l_c` are the upper and lower time and
:math:`h_e` and :math:`l_e` are the upper and lower intervals.
Note that no distinction is made between flagged and unflagged data
when establishing the endpoints in the bin.

To achieve the above, the Measurement Set **TIME** and **INTERVAL**
columns are used to sort and average input rows into output rows,
because they establish a time-based grid
on Measurement Set data. Additionally, the
`Measurement Set v2.0 Specification
<https://casa.nrao.edu/Memos/229.html>`_ specifies that both
flagged and unflagged data should be included in the computation
of **TIME** and **INTERVAL** values.
This means that averaging a regular high-resolution
**baseline x htime** grid should exactly produce a regular
low-resolution  **baseline x ltime** grid (**htime > ltime**)
even if some values are flagged. For this reason, flagged values
are retained during the averaging process instead of being
discarded from the output.

To summarise, the averaged times in each bin establish a map:

- from possibly unordered input rows.
- to a reduced set of output rows ordered by
  averaged :code:`(TIME, ANTENNA1, ANTENNA2)`.

Flagged Data Handling
~~~~~~~~~~~~~~~~~~~~~

The averager will output averages for bins that are completely flagged.

The `Measurement Set v2.0 Specification
<https://casa.nrao.edu/Memos/229.html>`_ specifies that both
flagged and unflagged data should be included in the computation
of **TIME** and **INTERVAL** values.

By contrast, most other columns
such as **TIME_CENTROID** and **EXPOSURE**,
should only include unflagged data or valid data.


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
2. **TIME** and **INTERVAL** columns always contain the average of **Both**
   flagged and unflagged data.
3. In the case of other columns, if a bin contains unflagged data,
   the bin will be set to the average of this data. However, if the bin
   is completely flagged, it will contain the average of all data.
   In other words the bin is **Exclusively** unflagged or flagged.
4. Completely flagged bins will be set as flagged in either case.
5. Certain columns are averaged, while others are summed,
   or simply assigned to the last value in the bin in the case
   of antenna indices.
6. In particular, **visibility data** is averaged by a
   `Mean of Circular Quantities
   <https://en.wikipedia.org/wiki/Mean_of_circular_quantities>`_
   and this means that visibility amplitudes are normalised.

=============== ================= ============================ ===========
Column          Unflagged/Flagged Aggregation Method           Required
                sample handling
=============== ================= ============================ ===========
TIME            Both              Mean                         No
INTERVAL        Both              Sum                          No
ANTENNA1        Both              Assigned to Last Input       Yes
ANTENNA2        Both              Assigned to Last Input       Yes
TIME_CENTROID   Exclusive         Mean                         Yes
EXPOSURE        Exclusive         Sum                          Yes
FLAG_ROW        Exclusive         Set if All Inputs Flagged    No
UVW             Exclusive         Mean                         No
WEIGHT          Exclusive         Mean                         No
SIGMA           Exclusive         Mean                         No
DATA (vis)      Exclusive         Mean of Circular Quantities  No
FLAG            Exclusive         Set if All Inputs Flagged    No
WEIGHT_SPECTRUM Exclusive         Mean                         No
SIGMA_SPECTRUM  Exclusive         Mean                         No
=============== ================= ============================ ===========

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

