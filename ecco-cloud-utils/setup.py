import setuptools

from distutils.core import setup

setup(
  name = 'ecco-cloud-utils',
  packages = ['ecco-cloud-utils'], # this must be the same as the name above
  version = '0.0.1',
  description = 'Estimating the Circulation and Climate of the Ocean (ECCO) - Cloud Package',
  author = 'Ian Fenty',
  author_email = 'ian.fenty@jpl.nasa.gov',
  url = 'https://github.com/ECCO-GROUP/ECCO-CLOUD/ecco-cloud-utils',
  keywords = ['ecco','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
 # data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
  install_requires=['datetime', 'python-dateutil', 'numpy'],
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research', 
      'License :: OSI Approved :: MIT License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Physics'
  ]
)
