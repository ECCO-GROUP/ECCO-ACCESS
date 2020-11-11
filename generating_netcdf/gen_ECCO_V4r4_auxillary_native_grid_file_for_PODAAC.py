#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:27:53 2020

@author: ifenty
"""

import sys
import json
import numpy as np
from importlib import reload
sys.path.append('/home/ifenty/ECCOv4-py')
import ecco_v4_py as ecco
reload(ecco)
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
import ecco_cloud_utils as ea
from pathlib import Path
import pandas as pd
import netCDF4 as nc4
import xarray as xr
import datetime
from pprint import pprint
from collections import OrderedDict
import pyresample as pr
import uuid
import pickle
import matplotlib.pyplot as plt
from pandas import read_csv
reload(ecco)

# uses the MITgcm simplegrid package by Greg Moore and Ian Fenty
# https://github.com/nasa/simplegrid
sys.path.append('/home/ifenty/git_repos_others/simplegrid')
import simplegrid as sg


## GRID DIR -- MAKE SURE USE GRID FIELDS WITHOUT BLANK TILES!!
#mds_grid_dir = Path('/Users/ifenty/tmp/no_blank_all')

## METADATA
metadata_json_dir = Path('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')

metadata_fields = ['ECCOv4r4_global_metadata_for_all_datasets',
                   'ECCOv4r4_global_metadata_for_latlon_datasets',
                   'ECCOv4r4_global_metadata_for_native_datasets',
                   'ECCOv4r4_coordinate_metadata_for_1D_datasets',
                   'ECCOv4r4_coordinate_metadata_for_latlon_datasets',
                   'ECCOv4r4_coordinate_metadata_for_native_datasets',
                   'ECCOv4r4_geometry_metadata_for_latlon_datasets',
                   'ECCOv4r4_geometry_metadata_for_native_datasets',
                   'ECCOv4r4_groupings_for_1D_datasets',
                   'ECCOv4r4_groupings_for_latlon_datasets',
                   'ECCOv4r4_groupings_for_native_datasets',
                   'ECCOv4r4_variable_metadata',
                   'ECCOv4r4_variable_metadata_for_latlon_datasets']


print('\nLOADING METADATA')
# load METADATA
metadata = dict()

for mf in metadata_fields:
    mf_e = mf + '.json'
    print(mf_e)
    with open(str(metadata_json_dir / mf_e), 'r') as fp:
        metadata[mf] = json.load(fp)


# metadata for different variables
global_metadata_for_all_datasets = metadata['ECCOv4r4_global_metadata_for_all_datasets']
global_metadata_for_latlon_datasets = metadata['ECCOv4r4_global_metadata_for_latlon_datasets']
global_metadata_for_native_datasets = metadata['ECCOv4r4_global_metadata_for_native_datasets']

coordinate_metadata_for_1D_datasets = metadata['ECCOv4r4_coordinate_metadata_for_1D_datasets']
coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']

geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
geometry_metadata_for_native_datasets = metadata['ECCOv4r4_geometry_metadata_for_native_datasets']

groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']
variable_metadata = metadata['ECCOv4r4_variable_metadata']

global_metadata = global_metadata_for_all_datasets + global_metadata_for_native_datasets

ecco_grid_dir_mds = Path('/home/ifenty/data/grids/grid_ECCOV4r4')

title='ECCO V4r4 Grid Geometry'
file_basename = 'ECCO_V4r4_llc90_grid_geometry'

mitgrid_files = list(ecco_grid_dir_mds.glob('tile*mitgrid'))
mitgrid_tile = dict()
mitgrid_tile[1] = sg.gridio.read_mitgridfile(ecco_grid_dir_mds / 'tile001.mitgrid', 90, 270)
mitgrid_tile[2] = sg.gridio.read_mitgridfile(ecco_grid_dir_mds / 'tile002.mitgrid', 90, 270)
mitgrid_tile[3] = sg.gridio.read_mitgridfile(ecco_grid_dir_mds / 'tile003.mitgrid', 90, 90)
mitgrid_tile[4] = sg.gridio.read_mitgridfile(ecco_grid_dir_mds / 'tile004.mitgrid', 270, 90)
mitgrid_tile[5] = sg.gridio.read_mitgridfile(ecco_grid_dir_mds / 'tile005.mitgrid', 270, 90)

XG_igjg = dict()
YG_igjg = dict()

for i in range(1,6):
    XG_igjg[i] = mitgrid_tile[i]['XG'].T
    YG_igjg[i] = mitgrid_tile[i]['YG'].T

XG_igjg_tiles = ecco.llc_ig_jg_faces_to_tiles(XG_igjg)
YG_igjg_tiles = ecco.llc_ig_jg_faces_to_tiles(YG_igjg)

XC_bnds = np.zeros((13,90,90,4))
YC_bnds = np.zeros((13,90,90,4))

for tile in range(13):
    XC_bnds[tile,:,:,0] = XG_igjg_tiles[tile, :-1, :-1] # --
    XC_bnds[tile,:,:,1] = XG_igjg_tiles[tile, :-1, 1:]   # -+
    XC_bnds[tile,:,:,2] = XG_igjg_tiles[tile, 1:,  1:]   # ++
    XC_bnds[tile,:,:,3] = XG_igjg_tiles[tile, 1:, :-1] # +-

    YC_bnds[tile,:,:,0] = YG_igjg_tiles[tile, :-1, :-1] # --
    YC_bnds[tile,:,:,1] = YG_igjg_tiles[tile, :-1, 1:]   # -+
    YC_bnds[tile,:,:,2] = YG_igjg_tiles[tile, 1:,  1:]   # ++
    YC_bnds[tile,:,:,3] = YG_igjg_tiles[tile, 1:, :-1] # +-

#%%
tile_coords = list(range(13))
ij_coords = list(range(90))
nb_coords = list(range(4))

XC_bnds_DA = xr.DataArray(XC_bnds, dims=["tile","j","i","nb"], coords=[tile_coords, ij_coords, ij_coords, nb_coords])
YC_bnds_DA = xr.DataArray(YC_bnds, dims=["tile","j","i","nb"], coords=[tile_coords, ij_coords, ij_coords, nb_coords])
XC_bnds_DA.name = 'XC_bnds'
YC_bnds_DA.name = 'YC_bnds'
XC_YC_bnds = xr.merge([XC_bnds_DA, YC_bnds_DA])

print(XC_bnds_DA)
#%%

output_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4/')
grid_new = \
        ecco.create_nc_grid_files_on_native_grid_from_mds(ecco_grid_dir_mds, output_dir,\
                                                          title = title, \
                                                          file_basename = file_basename,\
                                                          coordinate_metadata = coordinate_metadata_for_native_datasets, \
                                                          geometry_metadata   = geometry_metadata_for_native_datasets, \
                                                          global_metadata     = global_metadata,\
                                                          cell_bounds = XC_YC_bnds,
                                                          less_output=True, \
                                                          write_to_disk=True)


#%%


XGf = ecco.llc_tiles_to_faces(grid_new.XG)
YGf = ecco.llc_tiles_to_faces(grid_new.YG)

plt.figure(num=1, clear=True)
for i in range(1,6):
    plt.subplot(2,5,i)
    plt.imshow(XG_igjg[i], origin='lower',vmin=-180,vmax=180)
    plt.title(i)

    plt.subplot(2,5,5+i)
    plt.imshow(XGf[i], origin='lower',vmin=-180,vmax=180)
    plt.title(i)

plt.figure(num=2, clear=True)
for i in range(1,6):
    plt.subplot(2,5,i)
    plt.imshow(YG_igjg[i], origin='lower',vmin=-90,vmax=90)
    plt.title(i)

    plt.subplot(2,5,5+i)
    plt.imshow(YGf[i], origin='lower',vmin=-90,vmax=90)
    plt.title(i)


plt.figure(num=3,clear=True)
for i in range(1,14):
    plt.subplot(2,13,i)
    plt.imshow(XG_igjg_tiles[i-1], origin='lower',vmin=-180,vmax=180)
    plt.title(i)

    plt.subplot(2,13,13+i)
    plt.imshow(grid_new.XG[i-1], origin='lower',vmin=-180,vmax=180)
    plt.title(i)

plt.figure(num=4,clear=True)
for i in range(1,14):
    plt.subplot(2,13,i)
    plt.imshow(YG_igjg_tiles[i-1], origin='lower',vmin=-90,vmax=90)
    plt.title(i)

    plt.subplot(2,13,13+i)
    plt.imshow(grid_new.YG[i-1], origin='lower',vmin=-90,vmax=90)
    plt.title(i)


#%%
plt.figure(num=5,clear=True)
for i in range(1,14):
    plt.subplot(3,5,i)
    plt.plot(grid_new.XG[i-1], grid_new.YG[i-1],'ro')
    plt.plot(XG_igjg_tiles[i-1], YG_igjg_tiles[i-1],'b.')
    plt.plot(grid_new.XC[i-1], grid_new.YC[i-1],'kx')

#%%
plt.figure(num=6,clear=True)
i=20;j=20
cols = ['r','b','g','y']

import pyproj
proj_wgs84 = pyproj.Proj(init="epsg:4326")
proj_new = pyproj.Proj(init="epsg:3413")
lat = grid_new.YC.values
lon = grid_new.XC.values

lat_bnds = YC_bnds
lon_bnds = XC_bnds

x, y = pyproj.transform(proj_wgs84, proj_new, lon, lat)
x_bnds, y_bnds = pyproj.transform(proj_wgs84, proj_new, lon_bnds, lat_bnds)

n=2
for tile in range(0,13):
    plt.subplot(3,5,tile+1)
    if tile != 6:
        plt.plot(grid_new.XC[tile,i-n:i+n+1,j-n:j+n+1].values, \
                 grid_new.YC[tile,i-n:i+n+1,j-n:j+n+1].values,'ko',markersize=10)
        for c in range(4):
            plt.plot(XC_bnds[tile,i,j,c], YC_bnds[tile,i,j,c], '.',\
                     color=cols[c],markersize=20)
    else:
        plt.plot(x[tile,i-n:i+n+1,j-n:j+n+1], \
                 y[tile,i-n:i+n+1,j-n:j+n+1],'ko',markersize=10)
        for c in range(4):
            plt.plot(x_bnds[tile,i,j,c], y_bnds[tile,i,j,c], '.',\
                     color=cols[c],markersize=20)

