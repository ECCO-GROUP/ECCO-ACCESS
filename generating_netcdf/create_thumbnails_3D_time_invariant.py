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

dataset_dir = Path('/home/ifenty/tmp/time-invariant_20210317/')
mix_llc = xr.open_dataset(dataset_dir / 'OCEAN_3D_MIXING_COEFFS_ECCO_V4r4_native_llc0090.nc')
mix_ll = xr.open_dataset(dataset_dir / 'OCEAN_3D_MIXING_COEFFS_ECCO_V4r4_latlon_0p50deg.nc')


#%%
kl=5
tmp = mix_llc['KAPGM'].isel(k=kl)
cmap = copy.copy(cmocean.cm.dense)

cmap.set_bad(color='dimgray')

cmin = tmp.min()
cmax = tmp.max()
plt.close('all')
tb = ecco.plot_tiles(tmp, cmap=cmap, fig_num=1, \
                     show_tile_labels=False,\
                     cmin=cmin, cmax=cmax)

#%%
shortname_tmp = mix_llc.metadata_link.split('ShortName=')[1]
print(shortname_tmp)

tb_filename = shortname_tmp + '.jpg'

fig_ref = tb[0]
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

fname = dataset_dir / tb_filename
#plt.suptitle(f'ECCO V4r4 Gent-McWilliams diffusivity coefficient at {np.round(mix_llc.Z.isel(k=kl).values)}m')

print (fname)
plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)


#%%
#LATLON


plt.close('all')
kl=5
tmp = mix_ll['KAPGM'].isel(Z=kl)
cmap = copy.copy(cmocean.cm.dense)

cmin = tmp.min()
cmax = tmp.max()

#%%
shortname_tmp = mix_ll.metadata_link.split('ShortName=')[1]
print(shortname_tmp)

tb_filename = shortname_tmp + '.jpg'

fig_ref = tb[0]
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

fname = dataset_dir / tb_filename

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
plt.title(f'ECCO V4r4 Gent-McWilliams diffusivity coefficient at {np.round(mix_ll.Z.isel(Z=kl).values)}m')
#%%
print (fname)
plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)


