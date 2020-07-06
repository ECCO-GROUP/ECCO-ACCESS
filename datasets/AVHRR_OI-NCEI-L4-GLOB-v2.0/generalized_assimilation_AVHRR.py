#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import sys
from datetime import datetime
from pathlib import Path
import pyresample as pr

# Generalized functions
generalized_path = Path('../ECCO-ACCESS/ecco-cloud-utils/ecco-cloud-utils/generalized_functions')
sys.path.append(str(generalized_path))
import generalized_functions as gf

import ecco_cloud_utils as ea

import netCDF4 as nc4

# ### For Reference: Python binary file i/o specifications
# endianness:
#     big     : '>'
#     little  : '<'
# 
# precision  
#     float32 : 'f4'
#     float64 : 'f8'

### Variable Information ###
# The following lists the variables that may change from one dataset to the next
# along with a description of each one.
# --------------------------------------------------------------------------------------
# model_grid_dir ---------------- Directory of model grid to map data to
# model_grid_filename ----------- Name of model grid
# model_grid_id ----------------- ID of model grid to use in file/folder names
# model_grid_type --------------- Type of model grid
# model_grid_search_radius_max -- Maximum search radius to use in meters
# 
# product_name ---- Name of product, used in name of output files (Ex. File name w/o date)
# product_source -- Source of product, used in naming output folder (Ex. MODIS, SMAP)
# 
# data_fields -- Contains list of dictionaries for each data field to process
#             -- Can add as many data fields as needed. Each expects:
#                -- name ----------- Name of data field to use from dataset
#                -- long_name ------ Long name of data field
#                -- standard_name -- Standard name of data field
#                -- units ---------- Units of data field
# 
# new_data_attr -- Dictionary containing dataset information for final dataset. Expects:
#                  -- original_dataset_title ------- Title of original data set
#                  -- original_dataset_url --------- URL of original dataset (Ex. PO.DAAC link)
#                  -- original_dataset_reference --- Reference for original dataset
#                  -- original_dataset_product_id -- Product ID of original dataset
# 
# data_file_suffix -- Suffix of data files (Ex. '.nc')
# data_time_scale --- Time scale for the data (Supports 'daily', 'monthly')
# date_format ------- Format of date in file name (Supports 'yyyymm', 'yyyy_mm', 'yyyyddd', 'yyyymmdd')
# data_shortest_name -- Shortest time scale of data (Supports 'DAILY', 'MONTHLY')
# 
# time_zone_included_with_time -- Tells if time zone is included with the data's time
# remove_nan_days_from_data ----- Removes days of all nans from being used in mean 
# do_monthly_aggregation -------- Aggregates data into monthly mean datasets
# skipna_in_mean ---------------- Skips nans when taking mean for monthly aggregate
# extra_information ------------- Tells what special things may need to be done. Default empty.
#                              -- transpose -------- Transposes data before mapping values
#                              -- time_bounds_var -- Uses Time_bounds[0][] for start and end times
#                              -- time_var --------- Uses Time[0] for time value
#                              -- no_time ---------- No time given in data
#                              -- sea_ice_mask ----- Mask out data where there is sea ice
#                                                 -- Assumes sea_ice_fraction, lat, and lon
#                                                 -- values exist as named in dataset
#                                                 -- Must send target_grid to process_loop()
# 
# start_year ---------- Year to start reading data from
# end_year_exclusive -- Year to stop reading data from (exclusive)
# 
# data_dir -- Directory of data
# 
# data_res ---------- Resolution of data
# data_max_lat ------ Maximum latitude reached in data
# area_extent ------- Longitude and latitude bounds in data
# dims -------------- Longitude and latitude dimensions ([cols, rows])
#
# proj_info -- Dictionary containing the information for the projection of the data.
#           -- Used when defining the new grid in gf.grid_product(). Default is: 
#              -- 'area_id':'longlat'
#              -- 'area_name':'Plate Carree'
#              -- 'proj_id':'EPSG:4326'
#              -- 'proj4_args':'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

if __name__== "__main__":
    
    #%%
    #######################################################
    ##  BEGIN SPECIFIC PARAMETERS                        ##

    # model grid file to use for interpolation
    # ------------------------------------------
    #   * model grid must be provided as a netcdf file with XC and YC fields (lon, lat)
    #   * the netcdf file must also have an attribute (metadata) field 'title'
    #   * model grid can have multiple tiles (or facets or faces)
    model_grid_dir = Path('./')
    model_grid_filename  = 'ECCO-GRID.nc'
    model_grid_id   = 'llc90'
    model_grid_type = 'llc'
    model_grid_search_radius_max = 55000.0 # m
    
    # data names
    product_name = 'NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.0'
    product_source = 'AVHRR_generalized'
    
    # output parameters
    # output folder is specified with data idenifier
    mapping_time =  datetime.now().strftime("%Y%m%dT%H%M%S")
    netcdf_output_dir = Path(f'./data_output_{product_source}/mapped_to_{model_grid_id}/{mapping_time}/netcdf')
    binary_output_dir = Path(f'./data_output_{product_source}/mapped_to_{model_grid_id}/{mapping_time}/binary')
    
    # Define precision of output files, float32 is standard
    array_precision = np.float32
    
    # Define fill values for binary and netcdf
    if array_precision == np.float32:
        binary_output_dtype = '>f4'
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif array_precision == np.float64:
        binary_output_dtype = '>f8'
        netcdf_fill_value = nc4.default_fillvals['f8']

    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999
           
    # fields to process
    data_field_0 = {'name':'analysed_sst',
                    'long_name':'analysed sea surface temperature',
                    'standard_name':'sea_surface_temperature',
                    'units':'kelvin'}
    
    data_field_1 = {'name':'analysis_error',
                    'long_name':'estimated error standard deviation of analysed_sst',
                    'standard_name':'estimated_error_standard_deviation_of_analysed_sst',
                    'units':'kelvin'}
    
    data_fields = [data_field_0, data_field_1]
    
    # setup output attributes
    new_data_attr = {'original_dataset_title':'GHRSST Level 4 AVHRR_OI Global Blended Sea Surface Temperature Analysis (GDS version 2) from NCEI',
                     'original_dataset_url':'https://podaac.jpl.nasa.gov/dataset/AVHRR_OI-NCEI-L4-GLOB-v2.0',
                     'original_dataset_reference':'AVHRR_OI-NCEI-L4-GLOB-v2.0',
                     'original_dataset_product_id':'10.5067/GHAAO-4BC02'}
    
    # get filename format information
    data_file_suffix = '.nc'
    data_time_scale = 'daily'
    date_format = 'yyyymmdd'
    data_shortest_name = 'DAILY'
    
    # data booleans
    time_zone_included_with_time = True
    remove_nan_days_from_data = False
    do_monthly_aggregation = True
    skipna_in_mean = True
    extra_information = ['no_time_dashes', 'sea_ice_mask']
 
    # Years to process
    start_year = 1999
    end_year_exclusive = 2000
    years = np.arange(start_year, end_year_exclusive)

    # Location of original net (easier if all fields are softlinked to one spot)
    data_dir = Path('../Data/AVHRR/1')
    
    # data grid information
    data_res = 0.25
    data_max_lat = 90
    area_extent = [-180, 90, 180, -90] #90 and -90 switched since resulting data was flipped
    dims = [1440, 720]
    
    # Grid projection information
    proj_info = {'area_id':'longlat',
                 'area_name':'Plate Carree',
                 'proj_id':'EPSG:4326',
                 'proj4_args':'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'}    
    
    ##  END SPECIFIC PARAMETERS                          ##
    #######################################################
    #%%
    
    #%%
    #######################################################
    ## BEGIN GRID PRODUCT                                ##

    source_grid_min_L, source_grid_max_L, source_grid = gf.generalized_grid_product(product_name,
                                                                        data_res,
                                                                        data_max_lat,
                                                                        area_extent,
                                                                        dims,
                                                                        proj_info)
    
    ## END GRID PRODUCT                                  ##
    #######################################################
    #%%

    #%%
    #######################################################
    ## BEGIN MAPPING                                     ## 
   
    # make output directories
    netcdf_output_dir.mkdir(exist_ok=True, parents=True)
    binary_output_dir.mkdir(exist_ok=True, parents=True)
    
    # load the model grid
    model_grid = xr.open_dataset(model_grid_dir / model_grid_filename)
    model_grid  = model_grid.reset_coords()
  
    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the model grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    target_grid  = \
        pr.geometry.SwathDefinition(lons=model_grid.XC.values.ravel(), 
                                    lats=model_grid.YC.values.ravel())
   
    target_grid_radius = 0.5*np.sqrt(model_grid.rA.values.ravel())
    
    # Compute the mapping between the data and model grid
    source_indices_within_target_radius_i,\
    num_source_indices_within_target_radius_i,\
    nearest_source_index_to_target_index_i = \
        ea.find_mappings_from_source_to_target(source_grid,\
                                               target_grid,\
                                               target_grid_radius, \
                                               source_grid_min_L, \
                                               source_grid_max_L)
            
    ## END MAPPING                                       ##
    ####################################################### 
    #%%                         

    #%%    
    #######################################################
    ## BEGIN PROCESSING DATA                             ##
    
     # process model fields one at a time.
    for data_field_i, data_field_info in enumerate(data_fields) : #Unique
        # loop through different fields that are present in the 
        # netcdf files and that are requested
        # they have different name fields and metadata
        data_field = data_field_info['name']
     
        # loop through requested years
        for year in years:
            # get dates and filenames for this year
            iso_dates_for_year, files = \
                gf.generalized_get_data_filepaths_for_year(year, data_dir, data_file_suffix,
                                               data_time_scale, date_format)
    
            data_shortest_DA_year_merged = gf.generalized_process_loop_local(data_field_info,
                                                         data_dir,
                                                         iso_dates_for_year,
                                                         files,
                                                         source_indices_within_target_radius_i,
                                                         num_source_indices_within_target_radius_i,
                                                         nearest_source_index_to_target_index_i,
                                                         model_grid, model_grid_type,
                                                         array_precision,
                                                         time_zone_included_with_time,
                                                         remove_nan_days_from_data,
                                                         extra_information,
                                                         target_grid = target_grid)
            
            shortest_filename = f'{product_name}_{data_field}_{model_grid_id}_{data_shortest_name}_{year}'
            
            monthly_filename = f'{product_name}_{data_field}_{model_grid_id}_MONTHLY_{year}'
                               
            filenames = {'shortest':shortest_filename,
                         'monthly':monthly_filename}
            fill_values = {'binary':binary_fill_value,
                           'netcdf':netcdf_fill_value}
            output_dirs = {'binary':binary_output_dir,
                           'netcdf':netcdf_output_dir}

            new_data_attr['interpolated_grid_id'] = model_grid_id
            new_data_attr['new_name'] = f'{data_field}_interpolated_to_{model_grid_id}'

            gf.generalized_aggregate_and_save(data_shortest_DA_year_merged,
                                  new_data_attr,
                                  do_monthly_aggregation,
                                  year,
                                  skipna_in_mean,
                                  filenames,
                                  fill_values,
                                  output_dirs,
                                  binary_output_dtype,
                                  model_grid_type)
    
    ## END PROCESSING DATA                               ##
    #######################################################
    #%%