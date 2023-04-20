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
#sys.path.append('/home/ifenty/ECCOv4-py')
sys.path.append('/home/ifenty/git_repo_mine/ECCOv4-py/')

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



def meta_fixes(G):
    G.attrs['references'] ='ECCO Consortium, Fukumori, I., Wang, O., Fenty, I., Forget, G., Heimbach, P., & Ponte, R. M. 2020. Synopsis of the ECCO Central Production Global Ocean and Sea-Ice State Estimate (Version 4 Release 4). doi:10.5281/zenodo.3765928'
    G.attrs['source'] ='The ECCO V4r4 state estimate was produced by fitting a free-running solution of the MITgcm (checkpoint 66g) to satellite and in situ observational data in a least squares sense using the adjoint method'
    G.attrs['coordinates_comment'] = "Note: the global 'coordinates' attribute describes auxiliary coordinates."
    return G

#%%

output_array_precision = np.float32


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
                   'ECCOv4r4_variable_metadata_for_latlon_datasets',
                   'ECCOv4r4_dataset_summary']


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

global_metadata_for_native_datasets = metadata['ECCOv4r4_global_metadata_for_native_datasets']
global_metadata_for_latlon_datasets = metadata['ECCOv4r4_global_metadata_for_latlon_datasets']

global_latlon_metadata = global_metadata_for_all_datasets + global_metadata_for_latlon_datasets
coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']

global_metadata_native = global_metadata_for_all_datasets + global_metadata_for_native_datasets
coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']
geometry_metadata_for_native_datasets = metadata['ECCOv4r4_geometry_metadata_for_native_datasets']
variable_metadata = metadata['ECCOv4r4_variable_metadata']

dataset_summary = metadata['ECCOv4r4_dataset_summary']
podaac_dir = metadata_json_dir
podaac_dataset_table = read_csv(podaac_dir / 'PODAAC_datasets-revised_20210226.5.csv')

ecco_grid_dir_mds = Path('/home/ifenty/data/grids/grid_ECCOV4r4')

title='ECCO Geometry Parameters for the 0.5 degree Lat-Lon Model Grid (Version 4 Release 4'


#%%
# LOAD TILE FILES
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

XC_bnds_DA = xr.DataArray(XC_bnds, dims=["tile","j","i","nb"],\
                          coords=[tile_coords, ij_coords, ij_coords, nb_coords])
YC_bnds_DA = xr.DataArray(YC_bnds, dims=["tile","j","i","nb"],\
                          coords=[tile_coords, ij_coords, ij_coords, nb_coords])
XC_bnds_DA.name = 'XC_bnds'
YC_bnds_DA.name = 'YC_bnds'
XC_YC_bnds = xr.merge([XC_bnds_DA, YC_bnds_DA])

print(XC_bnds_DA)
#%%
file_basename = 'GRID_GEOMETRY_ECCO'

output_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4_20210316b/')
G =ecco.create_ecco_grid_geometry_from_mds(ecco_grid_dir_mds,
                                           grid_output_dir=None,
                                           title = title,
                                           file_basename = file_basename,
                                           coordinate_metadata = coordinate_metadata_for_native_datasets,
                                           geometry_metadata   = geometry_metadata_for_native_datasets,
                                           global_metadata     = global_metadata_native,
                                           cell_bounds = XC_YC_bnds,
                                           less_output=True).load()


#%%

filename ='GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'
G.attrs['product_name'] = filename

# get podaac metadata based on filename
print('\n... getting PODAAC metadata')
podaac_metadata = \
    ecco.find_podaac_metadata(podaac_dataset_table,
                              filename,
                              less_output=False)

# apply podaac metadata based on filename
print('\n... applying PODAAC metadata')
#pprint(podaac_metadata)
G = ecco.apply_podaac_metadata(G, podaac_metadata)

#%%
shortname = G.id.split('/')[1]
print(shortname)

G.attrs['summary']= dataset_summary[shortname]['summary']
G.attrs['title'] = dataset_summary[shortname]['title']
print(G.title)

G = meta_fixes(G)

#%%
# sort comments alphabetically
print('\n... sorting global attributes')
G.attrs = ecco.sort_attrs(G.attrs)

## Make encodings
netcdf_fill_value = ecco.get_netcdf_fill_val(output_array_precision)
encoding = ecco.create_dataset_netcdf_encoding(G)

# Make sure we don't have Dask arrays
G.load()

print('\n... creating output_dir', output_dir)
if not output_dir.exists():
    try:
        output_dir.mkdir(parents=True,exist_ok=True)
    except:
        print ('cannot make %s ' % output_dir)

# create full pathname for netcdf file
netcdf_output_filename = output_dir / filename

del G.attrs['original_mds_grid_dir']
del G.attrs['original_mds_var_dir']

#%%
G.to_netcdf(netcdf_output_filename, encoding=encoding)
G.close()


#%%

make_latlon_grid_geometry = True;

#%%
if make_latlon_grid_geometry:

    #%%

    ecco_grid = G.copy(deep=True)
    # land masks
    ecco_land_mask_c_nan  = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask_c_nan.values = np.where(ecco_land_mask_c_nan==True,1,np.nan)
    ecco_land_mask_w_nan  = ecco_grid.maskW.copy(deep=True)
    ecco_land_mask_w_nan.values = np.where(ecco_land_mask_w_nan==True,1,np.nan)
    ecco_land_mask_s_nan  = ecco_grid.maskS.copy(deep=True)
    ecco_land_mask_s_nan.values = np.where(ecco_land_mask_s_nan==True,1,np.nan)

    mapping_factors_dir = Path('/home/ifenty/tmp/ecco-v4-podaac-mapping-factors')
    debug_mode= False
    nk = 50
    wet_pts_k = dict()
    xc_wet_k = dict()
    yc_wet_k = dict()

    # Dictionary of pyresample 'grids' for each level of the ECCO grid where
    # there are wet points.  Used for the bin-averaging.  We don't want to bin
    # average dry points.
    source_grid_k = dict()
    print('\nSwath Definitions')
    print('... making swath definitions for latlon grid levels 1..nk')
    for k in range(nk):
        wet_pts_k[k] = np.where(ecco_grid.hFacC[k,:] > 0)
        xc_wet_k[k] = ecco_grid.XC.values[wet_pts_k[k]]
        yc_wet_k[k] = ecco_grid.YC.values[wet_pts_k[k]]

        source_grid_k[k]  = \
            pr.geometry.SwathDefinition(lons=xc_wet_k[k], \
                                        lats=yc_wet_k[k])

        print(k)
    #%%
    # The pyresample 'grid' information for the 'source' (ECCO grid) defined using
    # all XC and YC points, even land.  Used to create the land mask
    source_grid_all =  pr.geometry.SwathDefinition(lons=ecco_grid.XC.values.ravel(), \
                                                   lats=ecco_grid.YC.values.ravel())

    # the largest and smallest length of grid cell size in the ECCO grid.  Used
    # to determine how big of a lookup table we need to do the bin-average interp.
    source_grid_min_L = np.min([float(ecco_grid.dyG.min().values), float(ecco_grid.dxG.min().values)])
    source_grid_max_L = np.max([float(ecco_grid.dyG.max().values), float(ecco_grid.dxG.max().values)])

    # Define the TARGET GRID -- a lat lon grid
    ## create target grid.
    product_name = ''
    product_source = ''

    data_res = 0.5
    data_max_lat = 90.0
    area_extent = [-180.0, 90.0, 180.0, -90.0]
    dims = [int(360/data_res), int(180/data_res)]

    # Grid projection information
    proj_info = {'area_id':'longlat',
                 'area_name':'Plate Carree',
                 'proj_id':'EPSG:4326',
                 'proj4_args':'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'}

    target_grid_min_L, target_grid_max_L, target_grid, \
    target_grid_lons, target_grid_lats = \
        ea.generalized_grid_product(product_name,
                                    data_res,
                                    data_max_lat,
                                    area_extent,
                                    dims,
                                    proj_info)


    # pull out just the lats and lons (1D arrays)
    target_grid_lons_1D = target_grid_lons[0,:]
    target_grid_lats_1D = target_grid_lats[:,0]

    # calculate the areas of the lat-lon grid
    ea_area = ea.area_of_latlon_grid(-180, 180, -90, 90, \
                                               data_res, data_res,\
                                               less_output=True);
    lat_lon_grid_area = ea_area['area']
    target_grid_shape = lat_lon_grid_area.shape


    # calculate effective radius of each target grid cell.  required for the bin
    # averaging
    target_grid_radius = np.sqrt(lat_lon_grid_area / np.pi).ravel()


    #%5
    ##% CALCULATE GRID-TO-GRID MAPPING FACTORS
    print('\nGrid Mappings')
    grid_mapping_fname = mapping_factors_dir / "ecco_latlon_grid_mappings.p"

    #%%
    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:

       # if 'grid_mappings_k' not in globals():

        # first check to see if you have already calculated the grid mapping factors
        if grid_mapping_fname.is_file():
            # if so, load
            print('... loading latlon_grid_mappings.p')

            [grid_mappings_all, grid_mappings_k] = \
                pickle.load(open(grid_mapping_fname, 'rb'))

        else:
            # if not, make new grid mapping factors
            print('... no mapping factors found, recalculating')

			    # find the mapping between all points of the ECCO grid and the target grid.
            grid_mappings_all = \
                ea.find_mappings_from_source_to_target(source_grid_all,\
                                                       target_grid,\
                                                       target_grid_radius, \
                                                       source_grid_min_L, \
                                                       source_grid_max_L)

            # then find the mapping factors between all wet points of the ECCO grid
            # at each vertical level and the target grid
            grid_mappings_k = dict()

            for k in range(nk):
                print(k)
                grid_mappings_k[k] = \
                    ea.find_mappings_from_source_to_target(source_grid_k[k],\
                                                           target_grid,\
                                                           target_grid_radius, \
                                                           source_grid_min_L, \
                                                           source_grid_max_L)
            if not mapping_factors_dir.exists():
                try:
                    mapping_factors_dir.mkdir()
                except:
                    print ('cannot make %s ' % mapping_factors_dir)

            try:
                pickle.dump([grid_mappings_all, grid_mappings_k], \
                            open(grid_mapping_fname, 'wb'))
            except:
                print('cannot make %s ' % mapping_factors_dir)


    print('finished with grid mappings')

    #%%
    # make a land mask in lat-lon using hfacC
    print('\nLand Mask')
    if debug_mode:
        print('...DEBUG MODE -- SKIPPING LAND MASK')
        land_mask_ll = []

    else:
        # check to see if you have already calculated the land mask
        land_mask_fname = mapping_factors_dir / "ecco_latlon_land_mask.p"

        if land_mask_fname.is_file():
            # if so, load
            print('.... loading land_mask_ll')
            land_mask_ll = pickle.load(open(land_mask_fname, 'rb'))

        else:
            # if not, recalculate.
            print('.... making new land_mask_ll')
            land_mask_ll = np.zeros((nk, target_grid_shape[0], target_grid_shape[1]))

            source_indices_within_target_radius_i, \
            num_source_indices_within_target_radius_i,\
            nearest_source_index_to_target_index_i = grid_mappings_all

            for k in range(nk):
                print(k)
                source_field = ecco_land_mask_c_nan.values[k,:].ravel()

                land_mask_ll[k,:] =  \
                    ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                num_source_indices_within_target_radius_i,
                                                nearest_source_index_to_target_index_i,
                                                source_field, target_grid_shape,\
                                                operation='nearest', allow_nearest_neighbor=True)
            if not mapping_factors_dir.exists():
                try:
                    mapping_factors_dir.mkdir()
                except:
                    print ('cannot make %s ' % mapping_factors_dir)
            try:
                pickle.dump(land_mask_ll, open(land_mask_fname, 'wb'))
            except:
                print ('cannot pickle dump %s ' % land_mask_fname)

        #%%

    ## MAKE LAT AND LON BOUNDS FOR NEW DATA ARRAYS
    lat_bounds = np.zeros((dims[1],2))
    for i in range(dims[1]):
        lat_bounds[i,0] = target_grid_lats[i,0] - data_res/2
        lat_bounds[i,1] = target_grid_lats[i,0] + data_res/2
    #
    #
    lon_bounds = np.zeros((dims[0],2))
    for i in range(dims[0]):
        lon_bounds[i,0] = target_grid_lons[0,i] - data_res/2
        lon_bounds[i,1] = target_grid_lons[0,i] + data_res/2


    #%%

    ecco_grid_2D_fields_to_remap = ['Depth']
    #2D fields

    ll_grid_2D = []
    for f in ecco_grid_2D_fields_to_remap:
        F = ecco_grid[f].values


        F_wet_native = F[wet_pts_k[0]]

        # get mapping factors for the the surface level
        source_indices_within_target_radius_i, \
        num_source_indices_within_target_radius_i,\
        nearest_source_index_to_target_index_i = grid_mappings_k[0]

        # transform to new grid
        F_ll =  \
            ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                 num_source_indices_within_target_radius_i,
                                 nearest_source_index_to_target_index_i,
                                 F_wet_native,\
                                 target_grid_shape,\
                                 operation='mean', \
                                 allow_nearest_neighbor=True)

        F_ll_masked = F_ll * land_mask_ll[0,:]

        F_DA = xr.DataArray(F_ll_masked, \
                            coords=[target_grid_lats_1D,\
                                   target_grid_lons_1D], \
                            dims=["latitude","longitude"])
        F_DA.name = f
        ll_grid_2D.append(F_DA)

    #%%
    ecco_grid_3D_fields_to_remap = ['hFacC']

    ll_grid_3D = []
    for f in ecco_grid_3D_fields_to_remap:
        F = ecco_grid[f]

        F_ll = np.zeros((nk,360,720))
        max_k = 50
        for k in range(max_k):
            print(f, k)
            F_k = F[k].values
            F_wet_native = F_k[wet_pts_k[k]]

            source_indices_within_target_radius_i, \
            num_source_indices_within_target_radius_i,\
            nearest_source_index_to_target_index_i = grid_mappings_k[k]

            F_ll[k,:] =  \
                ea.transform_to_target_grid(source_indices_within_target_radius_i,
                     num_source_indices_within_target_radius_i,
                     nearest_source_index_to_target_index_i,
                     F_wet_native, target_grid_shape,\
                     operation='mean', allow_nearest_neighbor=True)


        # multiple by land mask
        F_ll_masked = F_ll * land_mask_ll

        Z = ecco_grid.Z.values

        F_DA = xr.DataArray(F_ll_masked, \
                            coords=[Z, \
                                    target_grid_lats_1D,\
                                    target_grid_lons_1D], \
                            dims=["Z", "latitude","longitude"])

        F_DA.name = f
        ll_grid_3D.append(F_DA)

    # --- end 2D or 3D
    #%% Merge the 3D and 2D fields
    A = xr.merge(ll_grid_3D)
    B = xr.merge(ll_grid_2D)

    #%%
    # Make area data variable
    area_latlon_DA = xr.DataArray(lat_lon_grid_area, \
                        coords=[target_grid_lats_1D,\
                                target_grid_lons_1D],\
                        dims=["latitude","longitude"])
    area_latlon_DA.name = 'area'

    # Make maskC data variable
    maskC = A.hFacC > 0
    maskC.name = 'maskC'


    #%% Make drF, drC
    drF_ll_DA= xr.DataArray(ecco_grid.drF.values, \
                            dims = 'Z', coords = [ecco_grid.Z.values])
    drF_ll_DA.name = 'drF'

    # Merge
    ecco_grid_ll = xr.merge([A,B, area_latlon_DA, drF_ll_DA, maskC])


    #   assign lat and lon bounds
    ecco_grid_ll=ecco_grid_ll.assign_coords({"latitude_bnds": (("latitude","nv"), lat_bounds)})
    ecco_grid_ll=ecco_grid_ll.assign_coords({"longitude_bnds": (("longitude","nv"), lon_bounds)})
    ecco_grid_ll=ecco_grid_ll.assign_coords({"Z_bnds": (("Z","nv"), ecco_grid.Z_bnds)})

    print(ecco_grid_ll.data_vars)
    print(ecco_grid_ll.coords)

    #%%
    ecco.add_global_metadata(global_latlon_metadata,ecco_grid_ll, '3D')
    ecco.add_coordinate_metadata(coordinate_metadata_for_latlon_datasets,ecco_grid_ll, less_output=True)
    ecco.add_variable_metadata(geometry_metadata_for_latlon_datasets,ecco_grid_ll, less_output=True)
    print(ecco_grid_ll)


    filename = 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'

    # add summary attribute = description of dataset
  #  ecco_grid_ll.attrs['summary'] = ecco_grid_ll_summary

    # get podaac metadata based on filename
    print('\n... getting PODAAC metadata')
    podaac_metadata = \
        ecco.find_podaac_metadata(podaac_dataset_table,
                                  filename,
                                  less_output=False)

    # apply podaac metadata based on filename
    print('\n... applying PODAAC metadata')
    ecco_grid_ll = ecco.apply_podaac_metadata(ecco_grid_ll,
                                              podaac_metadata,
                                              less_output=False)

    # Make encodings
    netcdf_fill_value = ecco.get_netcdf_fill_val(output_array_precision)
    encoding = ecco.create_dataset_netcdf_encoding(ecco_grid_ll)

    ecco_grid_ll.attrs['product_name'] = filename
    ecco_grid_ll.attrs['date_created'] = G.attrs['date_created']
    ecco_grid_ll.attrs['date_issued'] = G.attrs['date_issued']
    ecco_grid_ll.attrs['date_metadata_modified'] = G.attrs['date_metadata_modified']



    #%%
    shortname = ecco_grid_ll.id.split('/')[1]
    print(shortname)

    ecco_grid_ll.attrs['summary']= dataset_summary[shortname]['summary']
    ecco_grid_ll.attrs['title'] = dataset_summary[shortname]['title']
    print(ecco_grid_ll.title)

    ecco_grid_ll = meta_fixes(ecco_grid_ll)

    #%%
    # sort comments alphabetically
    print('\n... sorting global attributes')
    ecco_grid_ll.attrs = ecco.sort_attrs(ecco_grid_ll.attrs)


    #%%
    # remove granule time coverage attrs
    if 'time_coverage_end' in list(ecco_grid_ll.attrs.keys()):
        ecco_grid_ll.attrs.pop('time_coverage_end')

    if 'time_coverage_start' in list(ecco_grid_ll.attrs.keys()):
        ecco_grid_ll.attrs.pop('time_coverage_start')

    print('\n... creating output_dir', output_dir)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True,exist_ok=True)
        except:
            print ('cannot make %s ' % output_dir)

    # create full pathname for netcdf file
    netcdf_output_filename = output_dir / filename

    ecco_grid_ll.attrs['uuid'] = str(uuid.uuid1())

    # add one final comment (PODAAC request)
    ecco_grid_ll.attrs["coordinates_comment"] = \
        "Note: the global 'coordinates' attribute describes auxiliary coordinates."

    # sort comments alphabetically
    print('\n... sorting global attributes')
    ecco_grid_ll.attrs = ecco.sort_attrs(ecco_grid_ll.attrs)


    print(ecco_grid_ll)

    #%%
    # Make sure we don't have Dask arrays
    ecco_grid_ll.load()
    ecco_grid_ll.to_netcdf(netcdf_output_filename, encoding=encoding)
    ecco_grid_ll.close()
    #%%


#%%


#%%
#global_latlon_metadata = global_metadata_for_all_datasets + global_metadata_for_latlon_datasets
#coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
#geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
#variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']


#%%

# #%%
# if 1 == 0:

#     XGf = ecco.llc_tiles_to_faces(grid_new.XG)
#     YGf = ecco.llc_tiles_to_faces(grid_new.YG)

#     plt.figure(num=1, clear=True)
#     for i in range(1,6):
#         plt.subplot(2,5,i)
#         plt.imshow(XG_igjg[i], origin='lower',vmin=-180,vmax=180)
#         plt.title(i)

#         plt.subplot(2,5,5+i)
#         plt.imshow(XGf[i], origin='lower',vmin=-180,vmax=180)
#         plt.title(i)

#     plt.figure(num=2, clear=True)
#     for i in range(1,6):
#         plt.subplot(2,5,i)
#         plt.imshow(YG_igjg[i], origin='lower',vmin=-90,vmax=90)
#         plt.title(i)

#         plt.subplot(2,5,5+i)
#         plt.imshow(YGf[i], origin='lower',vmin=-90,vmax=90)
#         plt.title(i)


#     plt.figure(num=3,clear=True)
#     for i in range(1,14):
#         plt.subplot(2,13,i)
#         plt.imshow(XG_igjg_tiles[i-1], origin='lower',vmin=-180,vmax=180)
#         plt.title(i)

#         plt.subplot(2,13,13+i)
#         plt.imshow(grid_new.XG[i-1], origin='lower',vmin=-180,vmax=180)
#         plt.title(i)

#     plt.figure(num=4,clear=True)
#     for i in range(1,14):
#         plt.subplot(2,13,i)
#         plt.imshow(YG_igjg_tiles[i-1], origin='lower',vmin=-90,vmax=90)
#         plt.title(i)

#         plt.subplot(2,13,13+i)
#         plt.imshow(grid_new.YG[i-1], origin='lower',vmin=-90,vmax=90)
#         plt.title(i)


#     #%%
#     plt.figure(num=5,clear=True)
#     for i in range(1,14):
#         plt.subplot(3,5,i)
#         plt.plot(grid_new.XG[i-1], grid_new.YG[i-1],'ro')
#         plt.plot(XG_igjg_tiles[i-1], YG_igjg_tiles[i-1],'b.')
#         plt.plot(grid_new.XC[i-1], grid_new.YC[i-1],'kx')

#     #%%
#     plt.figure(num=6,clear=True)
#     i=20;j=20
#     cols = ['r','b','g','y']

#     import pyproj
#     proj_wgs84 = pyproj.Proj(init="epsg:4326")
#     proj_new = pyproj.Proj(init="epsg:3413")
#     lat = grid_new.YC.values
#     lon = grid_new.XC.values

#     lat_bnds = YC_bnds
#     lon_bnds = XC_bnds

#     x, y = pyproj.transform(proj_wgs84, proj_new, lon, lat)
#     x_bnds, y_bnds = pyproj.transform(proj_wgs84, proj_new, lon_bnds, lat_bnds)

#     n=2
#     for tile in range(0,13):
#         plt.subplot(3,5,tile+1)
#         if tile != 6:
#             plt.plot(grid_new.XC[tile,i-n:i+n+1,j-n:j+n+1].values, \
#                      grid_new.YC[tile,i-n:i+n+1,j-n:j+n+1].values,'ko',markersize=10)
#             for c in range(4):
#                 plt.plot(XC_bnds[tile,i,j,c], YC_bnds[tile,i,j,c], '.',\
#                          color=cols[c],markersize=20)
#         else:
#             plt.plot(x[tile,i-n:i+n+1,j-n:j+n+1], \
#                      y[tile,i-n:i+n+1,j-n:j+n+1],'ko',markersize=10)
#             for c in range(4):
#                 plt.plot(x_bnds[tile,i,j,c], y_bnds[tile,i,j,c], '.',\
#                          color=cols[c],markersize=20)

