**************************
Installing *ecco_access*
**************************

Preliminaries
-------------

The ``ecco_access`` package runs Python code, so you will first need a `Python <https://www.python.org/>`_ setup on your machine. Python is an open-source programming language, so no commercial software or licenses are necessary to run it. `Miniforge <https://conda-forge.org/download/>`_ is preferred for many users as it is a no-frills conda/mamba installation that defaults to the ``conda-forge`` channel of Python packages, where ``ecco_access`` is distributed. ``ecco_access`` can also be downloaded by users of other Python installers such as `Anaconda <https://www.anaconda.com/download/>`_ or `PyPI <https://pypi.org>`_, see below.

.. note::
   Currently ``ecco_access`` requires Python >= 3.11 to run correctly; 
   this is due to the requirements of Zarr 3 which ``ecco_access`` depends on.
   If this becomes an impediment to users, we will add compatibility 
   with earlier versions of Python.

Installing with conda/mamba
---------------------------

``ecco_access`` can be downloaded with the *conda* package management system, which is very helpful because *conda* typically will install all required dependencies and handle any package incompatibility issues. *mamba* has very similar functionality and syntax to *conda*, but is coded in C++ and is often faster than *conda*.

To install using *conda* (requires a Miniforge/Miniconda/Anaconda installation):

.. code-block:: bash
	
    conda install ecco_access

Or equivalently in *mamba* (available to Miniforge users):

.. code-block:: bash
    
    mamba install ecco_access
    
    
If for some reason the above command returns an error (especially if using Anaconda), include the ``-c`` option to point to the channel where the package is found (*conda-forge*).

.. code-block:: bash

    conda install -c conda-forge ecco_access


Installing with PyPI
--------------------

If you do not have a conda/mamba installation or prefer to work with the Python Package Index (PyPI), ``ecco_access`` can be installed with the ``pip`` command.

.. code-block:: bash
	
    pip install ecco-access


Cloning the GitHub repository
-----------------------------

If you might like to contribute new features or code improvements to ``ecco_access``, you can clone the code repository from GitHub. You will need `git <https://git-scm.com/install/>`_ installed and a `GitHub <https://github.com>`_ account. It is also recommended for security purposes to set up an `HTTPS Personal Access Token`_ or `SSH key`_.

.. _HTTPS Personal Access Token : https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
.. _SSH key : https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

If you are planning to contribute to ``ecco_access`` development, please create your own fork by navigating to the `repository <https://github.com/ECCO-GROUP/ECCO-ACCESS>`_. Then while logged in to your account on GitHub, click the "Fork" button on the upper-right part of the page, and create a fork with you as owner; the repository name can remain the same. The resulting fork will be *{your_username}/ECCO-ACCESS*.

Then to clone your fork of the ``ecco_access`` code repository to your local machine:

.. code-block:: bash

    git clone https://github.com/{your_username}/ECCO-ACCESS.git

or using an SSH key:

.. code-block:: bash
    
    git clone git@github.com:{your_username}/ECCO-ACCESS.git
