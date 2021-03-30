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
    G.attrs['coordinates_comment'] = "Note: the global 'coordinates' attribute describes auxillary coordinates."
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

ecco_grid_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4')

title='ECCO Geometry Parameters for the 0.5 degree Lat-Lon Model Grid (Version 4 Release 4'

mixing_dir = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/3dmixingparameters')

#%%

output_dir = Path('/home/ifenty/tmp/time-invariant_20210317/')
output_dir.mkdir(parents=True, exist_ok=True)

#%%
plt.close('all')

ecco_grid = xr.open_dataset(ecco_grid_dir / 'GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc')

#%%
xx_diffkr = ecco.read_llc_to_tiles(mixing_dir, 'xx_diffkr.effective.0000000129.data', nk=50)
xx_diffkr_DA = xr.ones_like(ecco_grid.hFacC)
xx_diffkr_DA.name = 'DIFFKR'
xx_diffkr_DA.attrs = dict()
xx_diffkr_DA.values = xx_diffkr
xx_diffkr_DA = xx_diffkr_DA.where(ecco_grid.hFacC > 0)
ecco.plot_tiles(xx_diffkr_DA.isel(k=10), layout='latlon',rotate_to_latlon=True, show_colorbar=True,
                show_tile_labels=(False))
print(xx_diffkr_DA )


#%%
xx_kapgm = ecco.read_llc_to_tiles(mixing_dir, 'xx_kapgm.effective.0000000129.data', nk=50)
xx_kapgm_DA = xr.ones_like(ecco_grid.hFacC)
xx_kapgm_DA.name = 'KAPGM'
xx_kapgm_DA.attrs = dict()
xx_kapgm_DA.values = xx_kapgm
xx_kapgm_DA = xx_kapgm_DA.where(ecco_grid.hFacC > 0)
ecco.plot_tiles(xx_kapgm_DA.isel(k=10), layout='latlon',rotate_to_latlon=True, show_colorbar=True, show_tile_labels=(False),cmap='jet')
print(xx_kapgm_DA)

#%%
xx_kapredi = ecco.read_llc_to_tiles(mixing_dir, 'xx_kapredi.effective.0000000129.data', nk=50)
xx_kapredi_DA = xr.ones_like(ecco_grid.hFacC)
xx_kapredi_DA.name = 'KAPREDI'
xx_kapredi_DA.attrs = dict()
xx_kapredi_DA.values = xx_kapredi
xx_kapredi_DA = xx_kapredi_DA.where(ecco_grid.hFacC > 0)
ecco.plot_tiles(xx_kapredi_DA.isel(k=10), layout='latlon',rotate_to_latlon=True, show_colorbar=True, show_tile_labels=(False),cmap='jet')
print(xx_kapredi_DA)

#%%
G = ecco_grid.copy(deep=True)

for dv in G.data_vars:
    G = G.drop_vars(G)

G = xr.merge([G, xx_diffkr_DA, xx_kapgm_DA, xx_kapredi_DA])



# current time and date
current_time = datetime.datetime.now().isoformat()[0:19]
G.attrs['date_created'] = current_time
G.attrs['date_modified'] = current_time
G.attrs['date_metadata_modified'] = current_time
G.attrs['date_issued'] = current_time
pprint(G)

#%%
grouping_gcmd_keywords = []
# ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
print('\n... adding metadata specific to the variable')
G, grouping_gcmd_keywords = \
    ecco.add_variable_metadata(variable_metadata, G, grouping_gcmd_keywords)

#print('\n... using 1D dataseta metadata specific to the variable')
#G, grouping_gcmd_keywords = \
#    ecco.add_variable_metadata(variable_metadata_1D, G, grouping_gcmd_keywords)

for dv in G.data_vars:
    pprint(G[dv].attrs)

    G[dv].attrs['valid_min'] = np.nanmin(G[dv].values)
    G[dv].attrs['valid_max'] = np.nanmax(G[dv].values)


#%%
pprint(G)
#%%

#%%
# ADD GLOBAL METADATA
dataset_dim='3D'
print("\n... adding global metadata for all datasets")
G = ecco.add_global_metadata(global_metadata_for_all_datasets, G,\
                        dataset_dim)
print('\n... adding global metadata for native dataset')
G = ecco.add_global_metadata(global_metadata_for_native_datasets, G,\
                                dataset_dim)



#%%
filename ='OCEAN_3D_MIXING_COEFFS_ECCO_V4r4_native_llc0090.nc'
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

pprint(G)

#%%
# current time and date
current_time = datetime.datetime.now().isoformat()[0:19]
G.attrs['date_created'] = current_time
G.attrs['date_modified'] = current_time
G.attrs['date_metadata_modified'] = current_time
G.attrs['date_issued'] = current_time
pprint(G)


#%%

# ADD GLOBAL METADATA ASSOCIATED WITH TIME AND DATE
print('\n... adding time / data global attrs')
G.attrs['time_coverage_start'] = G.attrs['product_time_coverage_start']
G.attrs['time_coverage_end']   = G.attrs['product_time_coverage_end']

#%%
# uuic
print('\n... adding uuid')
G.attrs['uuid'] = str(uuid.uuid1())

G = meta_fixes(G)

#%%
# sort comments alphabetically
print('\n... sorting global attributes')
G.attrs = ecco.sort_attrs(G.attrs)


#%%
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

#%%
G.to_netcdf(netcdf_output_filename, encoding=encoding)
G.close()


#%%

make_latlon_3D_fields = True;

#%%
if make_latlon_3D_fields:

    #%%

    ecco_ll_grid = xr.open_dataset(ecco_grid_dir / 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc')

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


    #%%
    ##% CALCULATE GRID-TO-GRID MAPPING FACTORS
    print('\nGrid Mappings')
    grid_mapping_fname = mapping_factors_dir / "ecco_latlon_grid_mappings.p"

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
    ecco_3D_fields_to_remap = ['DIFFKR','KAPGM','KAPREDI']


    ll_field_3D = []
    for f in ecco_3D_fields_to_remap:
        F = G[f]

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


        Z = ecco_grid.Z.values

        F_DA = xr.DataArray(F_ll, \
                            coords=[Z, \
                                    target_grid_lats_1D,\
                                    target_grid_lons_1D], \
                            dims=["Z", "latitude","longitude"])

        F_DA.name = f
        ll_field_3D.append(F_DA)

    # --- end 2D or 3D
    #%% Merge the 3D and 2D fields
    A = xr.merge(ll_field_3D)

    H = ecco_ll_grid.copy(deep=True)
    H = xr.merge([H, A])

    ecco.add_global_metadata(global_latlon_metadata, H, '3D')
    ecco.add_variable_metadata(variable_metadata, H, less_output=True)

    H = H.drop_vars(['area','drF','Depth','maskC','hFacC'])

    for dv in H.data_vars:
        H[dv].attrs['valid_min'] = np.nanmin(H[dv].values)
        H[dv].attrs['valid_max'] = np.nanmax(H[dv].values)

    print(H)
    for dv in H.data_vars:
        print('\n')
        pprint(H[dv].attrs)
        H[dv].values = np.where(ecco_ll_grid.hFacC.values >0, H[dv].values, np.nan)
    #%%

    filename = 'OCEAN_3D_MIXING_COEFFS_ECCO_V4r4_latlon_0p50deg.nc'
    H.attrs['product_name'] = filename

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
    H = ecco.apply_podaac_metadata(H,
                                   podaac_metadata,
                                   less_output=False)

    # current time and date

    current_time = datetime.datetime.now().isoformat()[0:19]
    H.attrs['date_created'] = current_time
    H.attrs['date_modified'] = current_time
    H.attrs['date_metadata_modified'] = current_time
    H.attrs['date_issued'] = current_time
    pprint(H)

    #%%
    shortname = H.id.split('/')[1]
    print(shortname)

    H.attrs['summary']= dataset_summary[shortname]['summary']
    H.attrs['title'] = dataset_summary[shortname]['title']
    print(H.title)


    # ADD GLOBAL METADATA ASSOCIATED WITH TIME AND DATE
    print('\n... adding time / data global attrs')
    H.attrs['time_coverage_start'] = H.attrs['product_time_coverage_start']
    H.attrs['time_coverage_end']   = H.attrs['product_time_coverage_end']

    #%%
    # uuic
    print('\n... adding uuid')
    H.attrs['uuid'] = str(uuid.uuid1())

    H = meta_fixes(H)

    # Make encodings
    netcdf_fill_value = ecco.get_netcdf_fill_val(output_array_precision)
    encoding = ecco.create_dataset_netcdf_encoding(H)

    #%%
    # sort comments alphabetically
    print('\n... sorting global attributes')
    H = ecco.sort_all_attrs(H)

    #%%

    print('\n... creating output_dir', output_dir)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True,exist_ok=True)
        except:
            print ('cannot make %s ' % output_dir)

    # create full pathname for netcdf file
    netcdf_output_filename = output_dir / filename

#    ecco_grid_ll.attrs['uuid'] = str(uuid.uuid1())

    # add one final comment (PODAAC request)
#    ecco_grid_ll.attrs["coordinates_comment"] = \
#        "Note: the global 'coordinates' attribute describes auxillary coordinates."

#    # sort comments alphabetically
#    print('\n... sorting global attributes')
#    ecco_grid_ll.attrs = ecco.sort_attrs(ecco_grid_ll.attrs)


    print(H)

    #%%
    # Make sure we don't have Dask arrays
    H.load()
    H.to_netcdf(netcdf_output_filename, encoding=encoding)
    H.close()
