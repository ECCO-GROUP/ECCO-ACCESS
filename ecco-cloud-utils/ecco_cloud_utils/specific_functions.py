# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:11:54 2020

@author: duncan
"""
import numpy as np


def avhrr_sst_kelvin_to_celcius(da):
    if 'analysed_sst' in da.name:
        da.attrs['units'] = 'celcius'
        old_values = da.values
        new_values = [x-273.15 for x in old_values]
        da.values = new_values
    return da

# Pre-transformation


def RDEFT4_remove_negative_values(ds):
    for field in ds.data_vars:
        if field in ['lat', 'lon']:
            continue
        ds[field].values = np.where(
            ds[field].values < 0, np.nan, ds[field].values)
    return ds
