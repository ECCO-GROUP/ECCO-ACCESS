from setuptools import setup

def README():
    with open('README.md') as f:
        return f.read()

setup(
  name = 'ecco_access',
  packages = ['ecco_access'], # this must be the same as the name above
  version = '0.1.0',
  description = 'Access utilities for ECCO state estimate output hosted on PO.DAAC and in the AWS cloud',
  author = 'Andrew Delman, Ian Fenty, Jack McNelis, Marie Zahn, and others',
  author_email = 'andrewdelman@ucla.edu',
  url = 'https://github.com/ECCO-GROUP/ECCO-ACCESS',
  keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
  python_requires = '>=3.7',
  install_requires=[
  'numpy',
  'future',
  'dask[complete]',
  'fsspec >= 2024.12.0',
  'netCDF4',
  'pandas',
  'python-dateutil',
  'requests',
  's3fs >= 2024.12.0',
  'tqdm',
  'xarray',
  'zarr >= 3.0.7'],
  tests_require=['pytest','coverage'],
  license='MIT',
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11',
      'Programming Language :: Python :: 3.12',
      'Programming Language :: Python :: 3.13',
      'Topic :: Scientific/Engineering :: Physics'
  ],
  long_description=README(),
  long_description_content_type='text/markdown'
)
