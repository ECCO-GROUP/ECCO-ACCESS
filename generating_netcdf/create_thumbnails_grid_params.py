#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:06:49 2020

@author: ifenty
"""

import sys
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCOv4-py')

import ecco_v4_py as ecco
import ecco_cloud_utils as ea
import copy as copy
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import netCDF4 as nc4
import xarray as xr
import datetime
from pprint import pprint
import pyresample as pr
import uuid
import pickle
from collections import OrderedDict
from pandas import read_csv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean


#%%

thumbnail_width_to_height_ratio = 1
thumbnail_height = 9.11


#%%
# NATIVE
plt.close('all')
netcdf_grid_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4_20210309')
ecco_grid_native = xr.open_dataset(netcdf_grid_dir / 'GRID_GEOMETRY_ECCO_v4r4_native_llc0090.nc')

cmap = copy.copy(cmocean.cm.deep)
cmap.set_bad(color='dimgray')

tmp = ecco_grid_native['Depth']
tmp = tmp.where(ecco_grid_native.maskC.isel(k=0))

cmin = tmp.min()
cmax = tmp.max()

tb = ecco.plot_tiles(tmp, cmap=cmap, fig_num=1, \
                     show_tile_labels=False,\
                     cmin=cmin, cmax=cmax)


shortname_tmp = ecco_grid_native.metadata_link.split('ShortName=')[1]
print(shortname_tmp)

tb_filename = shortname_tmp + '.jpg'

fig_ref = tb[0]
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

fname = netcdf_grid_dir / tb_filename

print (fname)
plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)


#%%
#LATLON


plt.close('all')
latlon_grid_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4_20210309')
ecco_grid_latlon = xr.open_dataset(netcdf_grid_dir / 'GRID_GEOMETRY_ECCO_v4r4_latlon_0p50deg.nc')

cmap = copy.copy(cmocean.cm.deep)
#cmap.set_bad(color='dimgray')

tmp = ecco_grid_latlon['Depth']
tmp = tmp.where(ecco_grid_latlon.maskC.isel(Z=0))

cmin = tmp.min()
cmax = tmp.max()

#%%
shortname_tmp = ecco_grid_latlon.metadata_link.split('ShortName=')[1]
print(shortname_tmp)

tb_filename = shortname_tmp + '.jpg'

fig_ref = tb[0]
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

fname = netcdf_grid_dir / tb_filename

print (fname)
#plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)

#%%
fig_ref = plt.figure(num=2, figsize=[8,8], clear=True);
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-66.5))
ecco.plot_global(tmp.longitude,\
                tmp.latitude,\
                tmp,\
                4326, cmin, cmax, ax,\
                show_grid_labels=False, cmap=cmap)

fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

#%%
print (fname)
plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)


