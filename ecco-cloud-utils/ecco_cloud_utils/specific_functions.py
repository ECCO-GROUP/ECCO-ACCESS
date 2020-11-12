# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:11:54 2020

@author: duncan
"""
import numpy as np

# Pre-transformation (on Datasets only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def RDEFT4_remove_negative_values(ds):
    for field in ds.data_vars:
        if field in ['lat', 'lon']:
            continue
        ds[field].values = np.where(
            ds[field].values < 0, np.nan, ds[field].values)
    return ds

# Post-transformations (on DataArrays only)
# -----------------------------------------------------------------------------------------------------------------------------------------------


def avhrr_sst_kelvin_to_celsius(da, field_name):
    if field_name == 'analysed_sst':
        da.attrs['units'] = 'Celsius'
        da.values -= 273.15
    return da


def seaice_concentration_to_fraction(da, field_name):
    if field_name == 'ice_conc':
        da.attrs['units'] = '1'
        da.values /= 100.
    return da
