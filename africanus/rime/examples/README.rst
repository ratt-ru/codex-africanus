Predict Script
==============

Predicts from sources.txt.

.. code-block:: bash

    $ python predict.py -sm sky-model.txt OBSERVATION.MS
    $ wsclean -weight uniform -name obs -size 512 512 -scale 1.5asec -make-psf -data-column MODEL_DATA OBSERVATION.MS
