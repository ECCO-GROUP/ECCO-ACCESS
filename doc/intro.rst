######################################################################
*ecco_access*: Python access utilities for the ECCO state estimate
######################################################################

Why use *ecco_access*?
=======================

``ecco_access`` makes it easy for any Python user to access ECCO output and open it in their workspace as an ``xarray`` dataset. It can also be used to access other datasets made available through NASA Earthdata (https://search.earthdata.nasa.gov/), if the ShortName of the dataset is known.

Some features that ``ecco_access`` supports, specific to ECCO output, include:

* Querying of the variables in ECCO datasets
* Spatial subsetting of ECCO output in the 13-tile Lat-Lon-Cap (LLC) native grid
* Access on the AWS Cloud to ECCO release(s) not yet available through NASA Earthdata

What versions of ECCO are currently supported?
==============================================

Currently access to ECCO version 4, releases 4 and 5 are supported. Additional versions will be made available as they are released through `NASA Earthdata`_ by the Physical Oceanography Distributed Active Archive Center (PO.DAAC), or made available on the ECCO S3 bucket in the AWS Cloud (s3://ecco-model-granules).

.. _NASA Earthdata : https://search.earthdata.nasa.gov

What is the ECCO state estimate?
================================

(courtesy of Ian Fenty)

The Estimating the Circulation and Climate of the Ocean (ECCO) Central Production state estimate is a reconstruction of the three-dimensional time-varying ocean and sea-ice state. Version 4 is provided on an approximately 1-degree horizontal grid (cells range from ca. 20 to 110 km in length and 50 vertical levels of varying thickness. ECCO version 4 currently starts in 1992 and extends until 2017 (release 4) or 2019 (release 5).

The ECCO CP state estimate has two defining features: (1) it reproduces a large set of remote sensing and in-situ observational data within their prior quantified uncertainties and (2) the dynamical evolution of its ocean circulation, hydrography, and sea-ice through time perfectly satisfies the laws of physics and thermodynamics.  The state estimate is the solution of a free-running ocean and sea-ice general circulation model and consequently can be used to assess budgets of quantities like heat, salt and vorticity.

For more details see `this summary`_.

.. _this summary : https://ecco-v4-python-tutorial.readthedocs.io/intro.html

Additional packages
===================

Once ECCO data are in your workspace, the `ecco_v4_py`_ package can be used to perform common computations, regridding, and plotting.

.. _ecco_v4_py : https://ecco-v4-python-tutorial.readthedocs.io

The `xmitgcm`_ package helps open ECCO output files in the binary MDS format that are typically used prior to formal releases.

.. _xmitgcm : https://xmitgcm.readthedocs.io/en/latest/index.html

The `xgcm`_ Python package facilitates computational operations such as interpolation and differentiation on ECCO's native LLC grid. The `gcmfaces`_ package has similar functionality for Matlab/Octave.

.. _xgcm : https://xgcm.readthedocs.io/en/latest/
.. _gcmfaces : https://gcmfaces.readthedocs.io/en/latest/

The `earthaccess`_ Python package can be also be used to query and access datasets on NASA Earthdata. It is a useful interface for querying NASA Earthdata metadata, though without some of the ECCO-specific features in ``ecco_access``.

.. _earthaccess : https://earthaccess.readthedocs.io/en/latest/
