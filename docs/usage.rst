Usage
=====

Nominally, this should be used via the Python API.

There are a few convenience endpoint defitions in `cli.py`.


For instance, the Kevnet training can be called with 


.. code-block:: bash

  train_kevnet --steps=100 --epochs=1000 --history=stage1.json --output=stage1.h5 dl_data/v04_00_00/*.h5

The other endpoints can be called similarly