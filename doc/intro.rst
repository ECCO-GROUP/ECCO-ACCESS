######################################################################
ecco_access: Python access utilities for the ECCO state estimate
######################################################################

Why use `ecco_access`_?
=======================

`ecco_access`_ makes it easy for any Python user to access ECCO output and open it in their workspace as an `xarray` dataset. It can also be used to access other datasets made available through NASA Earthdata (https://search.earthdata.nasa.gov/), if the ShortName of the dataset is known.

Additionally, `ecco_access`_ has some ECCO-specific features that enable:

- querying of the variables in ECCO datasets
- spatial subsetting of ECCO output in the 13-tile Lat-Lon-Cap (LLC) native grid
- access on the AWS Cloud to ECCO release(s) not yet available through NASA Earthdata

What versions of ECCO are currently supported?
==============================================

Currently access to ECCO version 4, releases 4 and 5 are supported. Additional versions will be made available as they are released through the Physical Oceanography Distributed Active Archive Center (PO.DAAC), or made available on the ECCO S3 bucket in the AWS Cloud (s3://ecco-model-granules).

What is the ECCO state estimate?
================================

(courtesy of Ian Fenty)
The Estimating the Circulation and Climate of the Ocean (ECCO) Central Production state estimate is a reconstruction of the three-dimensional time-varying ocean and sea-ice state. Version 4 is provided on an approximately 1-degree horizontal grid (cells range from ca. 20 to 110 km in length and 50 vertical levels of varying thickness. ECCO version 4 currently starts in 1992 and extends until 2017 (release 4) or 2019 (release 5).

The ECCO CP state estimate has two defining features: (1) it reproduces a large set of remote sensing and in-situ observational data within their prior quantified uncertainties and (2) the dynamical evolution of its ocean circulation, hydrography, and sea-ice through time perfectly satisfies the laws of physics and thermodynamics.  The state estimate is the solution of a free-running ocean and sea-ice general circulation model and consequently can be used to assess budgets of quantities like heat, salt and vorticity.

Fore more details see [this summary](https://ecco-v4-python-tutorial.readthedocs.io/intro.html).

Additional packages
===================

Once ECCO data are in your workspace, the `ecco_v4_py`_ package can be used to perform common computations, regridding, and plotting.

.. _ecco_v4_py : https://ecco-v4-python-tutorial.readthedocs.io

The `xmitgcm`_ package helps open ECCO output files in the binary MDS format that are typically used prior to formal releases.

.. _xmitgcm : https://xmitgcm.readthedocs.io/en/latest/index.html

The following packages are also useful for computational operations on ECCO's native LLC grid:

.. _xgcm (Python) : https://xgcm.readthedocs.io/en/latest/
.. _gcmfaces (Matlab/Octave) : https://gcmfaces.readthedocs.io/en/latest/

The `earthaccess`_ Python package can be also be used to query and access datasets on NASA Earthdata, though without the ECCO-specific features in `ecco_access`_.

.. _earthaccess : https://earthaccess.readthedocs.io/en/latest/
