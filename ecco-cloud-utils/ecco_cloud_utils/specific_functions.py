# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:11:54 2020

@author: duncan
"""

def avhrr_sst_kelvin_to_celcius(da):
    old_values = da.values
    new_values = [x-273.15 for x in old_values]
    da.values = new_values
    return da
