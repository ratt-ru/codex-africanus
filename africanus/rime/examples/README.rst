Predict Script
==============

predict.py: Predicts from sky-model.txt.

.. code-block:: bash

    $ python predict.py -sm sky-model.txt OBSERVATION.MS
    $ wsclean -weight uniform -name obs -size 512 512 -scale 1.5asec -make-psf -data-column MODEL_DATA OBSERVATION.MS


predict_shapelet.py: Predicts from N6251-sky-model.txt.

.. code-block:: bash

    $ python predict_shapelet.py -sm N6251-sky-model.txt OBSERVATION.MS
    $ wsclean -weight uniform -name obs -size 512 512 -scale 1.5asec -make-psf -data-column MODEL_DATA OBSERVATION.MS
