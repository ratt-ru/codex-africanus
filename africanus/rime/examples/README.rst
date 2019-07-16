Predict Script
==============

Predicts 5 sources in a cross pattern around (0, 60) degrees.
Assumes the phase centre in the supplied
Measurement Set is (0, 60) degrees.

.. code-block:: bash

    $ python predict.py -sm sky-model.txt OBSERVATION.MS
    $ wsclean -weight uniform -name obs -size 512 512 -scale 1.5asec -make-psf -data-column MODEL_DATA OBSERVATION.MS
