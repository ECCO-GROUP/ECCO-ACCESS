"""
Created on Fri May 10 17:27:09 2019

@author: ifenty"""

import sys
import json
import numpy as np
import importlib
sys.path.append('/home/ifenty/ECCOv4-py')
import ecco_v4_py as ecco
sys.path.append('/home/ifenty/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
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


#%%

def determine_metadata_modifiers(filename: str):
    """Return revised file metadata based on an input ECCO `filename`.
    
    This should consistently parse a filename that conforms to the 
    ECCO filename conventions and match it to a row in my metadata 
    table.
    
    """
    
    # Use filename components to find metadata row from __datasets__.
    if "_snap_" in filename:
        head, tail = filename.split("_snap_")
        head = f"{head}_snap"
    elif "_day_mean_" in filename:
        head, tail = filename.split("_day_mean_")
        head = f"{head}_day_mean"
    elif "_mon_mean_" in filename:
        head, tail = filename.split("_mon_mean_")
        head = f"{head}_mon_mean"
    else:
        raise Exception("Error: filename may not conform to ECCO V4r4 convention.")
#    
    
    print (head, tail)
    tail = tail.split("_ECCO_V4r4_")[1]
    
    # Get the filenames column from my table as a list of strings.
    names = __datasets__['DATASET.FILENAME']
    
    # Find the index of the row with a filename with matching head, tail chars.
    index = names.apply(lambda x: all([x.startswith(head), x.endswith(tail)]))
    
    # Select that row from __datasets__ table and make a copy of it.
    metadata = __datasets__[index].iloc[0].to_dict()
    
    gcmd_keywords = []
    # Select the gcmd keywords to match this dataset ShortName.
    for prefix, keywords in __keywords__.items():
        if metadata['DATASET.SHORT_NAME'].startswith(prefix):
            gcmd_keywords = ", ".join([" > ".join(kw.values()) for kw in keywords])
            break
    
    print(gcmd_keywords)
    
    modifiers = {
        'id': metadata['DATASET.PERSISTENT_ID'].replace("PODAAC-","10.5067/"),
        'metadata_link': f"https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName={metadata['DATASET.SHORT_NAME']}",
        'title': metadata['DATASET.LONG_NAME'],
    }
  
    return modifiers

#%%
def apply_metadata_modifiers(xrds, modifiers):
    """Apply attributes modifiers to ECCO dataset and its variables.
    
    Attributes that are commented `#` are retained with no modifications.
    Attributes that are assigned `None` are dropped.
    New attributes added to dataset.
    
    """
    # REPLACE GLOBAL ATTRIBUTES WITH NEW/UPDATED DICTIONARY.
    atts = xrds.attrs
    for name, modifier in modifiers.items():
        if callable(modifier):
            atts.update({name: modifier(x=atts[name])})
        elif modifier is None:
            if name in atts:
                del atts[name]
        else:
            atts.update({name: modifier})
    
    xrds.attrs = atts
    
    # MODIFY VARIABLE ATTRIBUTES IN A LOOP.
    for v in xrds.variables:
        if "gmcd_keywords" in xrds.variables[v].attrs:
            del xrds.variables[v].attrs['gmcd_keywords']
        if v=="latitude_bnds":
            xrds.variables[v].attrs['units'] = "degrees_north"
        if v=="longitude_bnds":
            xrds.variables[v].attrs['units'] = "degrees_east"
    return xrds  # Return the updated xarray Dataset.
    #return apply_modifiers  # Return the function.



#%%
def get_coordinate_attribute_to_data_vars(G):
    coord_attr = dict()
    
    dvs = list(G.data_vars)
    for dv in dvs:
        coord_attr[dv] = ' '.join([str(elem) for elem in G[dv].coords])
        
    coord_G = ' '.join([str(elem) for elem in G.coords])
    
    return coord_attr, coord_G

        
def find_metadata_in_json_dictionary(var, key, metadata, print_output=False):
    for m in metadata:        
        if m[key] == var:
            if print_output:
                print(m)
            return m
    return []


def sort_attrs(attrs):
    od = OrderedDict()
    
    keys = sorted(list(attrs.keys()),key=str.casefold)
    
    for k in keys:
        od[k] = attrs[k]

    return od


def sort_all_attrs(G, print_output=False):
    for coord in list(G.coords):
        if print_output:
            print(coord)
        new_attrs = sort_attrs(G[coord].attrs)
        G[coord].attrs = new_attrs
        
    for dv in list(G.data_vars):
        if print_output:
            print(dv)
        new_attrs = sort_attrs(G[dv].attrs)
        G[dv].attrs = new_attrs
        
    new_attrs = sort_attrs(G.attrs)
    G.attrs = new_attrs
        

#%%
# Define precision of output files, float32 is standard
# ------------------------------------------------------
array_precision = np.float32


# ECCO always uses -9999 for missing data.
binary_fill_value = -9999



ecco_start_time = np.datetime64('1992-01-01T12:00:00')

# hack to ensure that time bounds and time use same 'encoding' in xarray
time_encoding_start = 'hours since 1992-01-01 12:00:00'


## OUTPUT DIRECTORY
output_dir = Path('/home/ifenty/tmp/v4r4_nc_output_20201020')


## ECCO FIELD INPUT DIRECTORY 
# model diagnostic output 
# subdirectories must be 
#  'diags_all/diag_mon'
#  'diags_all/diag_mon'
#  'diags_all/snap'
        
diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/diags_all')

# Define tail for dataset description (summary)
dataset_description_tail_native = ' on the native Lat-Lon-Cap 90 (LLC90) model grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'
dataset_description_tail_latlon = ' interpolated to a regular 0.5-degree grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'


## GRID DIR -- MAKE SURE USE GRID FIELDS WITHOUT BLANK TILES!!
mds_grid_dir = Path('/Users/ifenty/tmp/no_blank_all')

## METADATA
metadata_json_dir = Path('/home/ifenty/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')

metadata_fields = ['ECCOv4r4_common_metadata', 
                   'ECCOv4r4_common_metadata_for_latlon_datasets',
                   'ECCOv4r4_common_metadata_for_native_grid_datasets',
                   'ECCOv4r4_common_coordinate_metadata',
                   'ECCOv4r4_dataset_groupings_for_latlon_product',
                   'ECCOv4r4_dataset_groupings_for_native_product',
                   'ECCOv4r4_variables_on_latlon_grid_metadata',
                   'ECCOv4r4_variables_on_native_grid_metadata',
                   'ECCOv4r4_podaac_datasets']


                    ## PODAAC fields
podaac_fields = Path('/home/ifenty/git_repo_others/ecco-data-pub/_ecco_podaac_nc_write_tail')


## ECCO GRID 
ecco_grid_dir = '/home/ifenty/ian1/ifenty/ECCOv4/Version4/Release4/nctiles_grid'


product_types = ['latlon'] #, 'native']
output_freq_codes  = ['AVG_DAY', 'AVG_MON']


#%% -- -program start


# Define fill values for binary and netcdf
# ---------------------------------------------
if array_precision == np.float32:
    binary_output_dtype = '>f4'
    netcdf_fill_value = nc4.default_fillvals['f4']

elif array_precision == np.float64:
    binary_output_dtype = '>f8'
    netcdf_fill_value = nc4.default_fillvals['f8']
    
    
# make output directory

if not output_dir.exists():
    try:
        output_dir.mkdir()
    except:
        print ('cannot make %s ' % output_dir)

# load METADATA

metadata = dict()

for mf in metadata_fields:
    mf_e = mf + '.json'
    print(mf_e)
    with open(str(metadata_json_dir / mf_e), 'r') as fp:
        metadata[mf] = json.load(fp)
        
# load PODAAC fields        
from pandas import read_csv
                                     
__datasets__ = read_csv(podaac_fields / 'datasets.csv')

with open(podaac_fields / 'keywords.json', "r") as f:
    __keywords__ = json.load(f)
        
# load ECCO grid
ecco_grid = ecco.load_ecco_grid_nc(ecco_grid_dir, 'ECCO-GRID.nc')
        
 
# land masks
ecco_land_mask_c = np.where(ecco_grid.hFacC.values == 0, 0, 1)
ecco_land_mask_w = np.where(ecco_grid.hFacW.values == 0, 0, 1)
ecco_land_mask_s = np.where(ecco_grid.hFacS.values == 0, 0, 1)

ecco_land_mask_c_nan = np.where(ecco_grid.hFacC.values == 0, np.nan, 1)
ecco_land_mask_w_nan = np.where(ecco_grid.hFacW.values == 0, np.nan, 1)
ecco_land_mask_s_nan = np.where(ecco_grid.hFacS.values == 0, np.nan, 1)

#%%

for product_type in product_types:
        
    if product_type == 'native':
        dataset_description_tail = dataset_description_tail_native
        groupings = metadata['ECCOv4r4_dataset_groupings_for_native_product']
        
    
    elif product_type == 'latlon':
    
        dataset_description_tail = dataset_description_tail_latlon
        groupings = metadata['ECCOv4r4_dataset_groupings_for_latlon_product']
        
        wet_pts_k = dict()
        xc_wet_k = dict()
        yc_wet_k = dict()
        
        # Dictionary of pyresample 'grids' for each level of the ECCO grid where
        # there are wet points.  Used for the bin-averaging.  We don't want to bin
        # average dry points.
        source_grid_k = dict()
        
        print('... making swath definitions for latlon grid levels 1..50')
        for k in range(50):
            wet_pts_k[k] = np.where(ecco_land_mask_c[k,:] == 1)
            xc_wet_k[k] = ecco_grid.XC.values[wet_pts_k[k]] 
            yc_wet_k[k] = ecco_grid.YC.values[wet_pts_k[k]] 
            
            source_grid_k[k]  = \
                pr.geometry.SwathDefinition(lons=xc_wet_k[k], \
                                            lats=yc_wet_k[k])
        
    
        # The pyresample 'grid' information for the 'source' (ECCO grid) defined using 
        # all XC and YC points, even land.  Used to create the land mask
        source_grid_all =  pr.geometry.SwathDefinition(lons=ecco_grid.XC.values.ravel(), \
                                                       lats=ecco_grid.YC.values.ravel())
        
        # the largest and smallest length of grid cell size in the ECCO grid.  Used
        # to determine how big of a lookup table we need to do the bin-average interp.
        source_grid_min_L = np.min([float(ecco_grid.dyG.min().values), float(ecco_grid.dxG.min().values)])
        source_grid_max_L = np.max([float(ecco_grid.dyG.max().values), float(ecco_grid.dxG.max().values)])
        
        #print(int(source_grid_min_L))
        #print(int(source_grid_max_L))
        
        
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
        lat_lon_grid_area = ea.area_of_latlon_grid(-180, 180, -90, 90, \
                                                   data_res, data_res,\
                                                   less_output=True);
        target_grid_shape = lat_lon_grid_area.shape
        
        
        # calculate effective radius of each target grid cell.  required for the bin 
        # averaging
        target_grid_radius = np.sqrt(lat_lon_grid_area / np.pi).ravel()
        
        
        ##% CALCULATE GRID-TO-GRID MAPPING FACTORS
        
        grid_mapping_fname = output_dir / "ecco_latlon_grid_mappings.p"
        
        if 'grid_mappings_k' not in globals():
                
            # first check to see if you have already calculated the grid mapping factors
            if grid_mapping_fname.is_file():
                # if so, load
                print('.... loading latlon_grid_mappings.p')
            
                [grid_mappings_all, grid_mappings_k] = \
                    pickle.load(open(grid_mapping_fname, "rb"))
                
            else:
                # if not, make new grid mapping factors
                    
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
                
                for k in range(50):
                    print(k)
                    grid_mappings_k[k] = \
                        ea.find_mappings_from_source_to_target(source_grid_k[k],\
                                                               target_grid,\
                                                               target_grid_radius, \
                                                               source_grid_min_L, \
                                                               source_grid_max_L)
                # save to disk
                pickle.dump([grid_mappings_all, grid_mappings_k], \
                            open(grid_mapping_fname, "wb"))
        else:
            print('... grid mappings k already in memory')
            
            
        # make a land mask in lat-lon using hfacC
       
        if 'land_mask_ll' not in globals():
            # check to see if you have already calculated the land mask
            land_mask_fname = output_dir / "ecco_latlon_land_mask.p"
                    
            if land_mask_fname.is_file():
                # if so, load
                print('.... loading land_mask_ll')
                land_mask_ll = pickle.load(open(land_mask_fname, "rb"))
            
            else:
                # if not, recalculate.
                land_mask_ll = np.zeros((50, target_grid_shape[0], target_grid_shape[1]))
                
                source_indices_within_target_radius_i, \
                num_source_indices_within_target_radius_i,\
                nearest_source_index_to_target_index_i = grid_mappings_all
                
                for k in range(50):
                    print(k)
                    
                    source_field = ecco_land_mask_c_nan[k,:].ravel()
                    
                    land_mask_ll[k,:] =  \
                        ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                    num_source_indices_within_target_radius_i,
                                                    nearest_source_index_to_target_index_i,
                                                    source_field, target_grid_shape,\
                                                    operation='nearest', allow_nearest_neighbor=True)
                # save to disk
                pickle.dump(land_mask_ll, open(land_mask_fname, "wb"))
        else:
            print('... land mask already in memory')

      
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
        
    
       
    ## end product_type = 'latlon'    
        
    # Make depth bounds
    depth_bounds = np.zeros((50,2))
    tmp = np.cumsum(ecco_grid.drF.values)
    
    for k in range(50):
        if k == 0:
            depth_bounds[k,0] = 0.0
        else:
            depth_bounds[k,0] = -tmp[k-1]
        depth_bounds[k,1] = -tmp[k]
        
    #%%
        
    
    
    
    # loop through output frequencies
    
    for output_freq_code in output_freq_codes:
    
        if output_freq_code == 'AVG_DAY':
            mds_diags_root_dir = diags_root / 'diags_daily'
            time_steps_to_load = [12,36,60]
            period_suffix = 'day_mean'
            dataset_description_head = 'This dataset contains daily-averaged '
            
        elif output_freq_code == 'AVG_MON':
            mds_diags_root_dir = diags_root / 'diags_monthly'
            time_steps_to_load = [732, 1428,2172]   
            period_suffix = 'mon_mean'
            dataset_description_head = 'This dataset contains monthly-averaged '
        
        elif output_freq_code == 'SNAPSHOT':
            mds_diags_root_dir = diags_root / 'diags_inst'
            period_suffix = 'inst'
            dataset_description_head = 'This dataset contains instantaneous '
            
        ## TIME STEPS TO LOAD
        #all_avail_time_steps = ecco.get_time_steps_from_mds_files(mds_diags_root_dir / 'ETAN', 'ETAN')
        #pprint(all_avail_time_steps)
        
        #time_steps_to_load = all_avail_time_steps[:2]
        print('\nloading time steps')
        pprint(time_steps_to_load)
          
        
        # load files    
        field_paths = np.sort(list(mds_diags_root_dir.glob('*' + period_suffix + '*')))
                
        ## load variable file and directory names
        print (len(field_paths))
        all_field_names = []
        
        for f in field_paths:
            all_field_names.append(f.name)
        
        print (all_field_names)
      
        max_k = 50
        
        
        # loop through groupings
        for grouping in groupings:
        
            dataset_description  = dataset_description_head + grouping['dataset_description'] + dataset_description_tail
            
            grouping_dim = grouping['dimension']
            
            tmp = grouping['fields'].split(',')
            vars_to_load  = []
            for var in tmp:
                vars_to_load.append(var.strip())
                
            var_directories = dict()
            
            for var in vars_to_load:
                var_match =  "".join([var, "_", period_suffix])
                num_matching_dirs = 0
                for fp in field_paths:
                    if var_match in str(fp):
                        #print(var, fp) 
                        var_directories[var] = fp
                        num_matching_dirs += 1
                if num_matching_dirs == 0:
                    print('>>>>>> no match found for ', var)
                elif num_matching_dirs > 1 :
                    print('>>>>>> more than one matching dir for ', var)
            
            for var in vars_to_load:
                print(var, var_directories[var])
                
            grouping_gcmd_keywords = []
           
            # loop through time
            for cur_ts in time_steps_to_load:
                
                time_delta = np.timedelta64(cur_ts, 'h')
                        
                cur_time = ecco_start_time + time_delta
                times = [pd.to_datetime(str(cur_time))]
                
                if 'AVG' in output_freq_code:
                    tb, ct = ecco.make_time_bounds_from_ds64(np.datetime64(times[0]),output_freq_code)
                    record_start_time = tb[0]
                    record_end_time = tb[1]
                else:
                    record_start_time = np.datetime64(times)
                          
                print(cur_time, tb, ct, record_start_time)
                            
                
                F_DS_vars = []

                # loop through variables to load
                for var in vars_to_load:
        
                    mds_var_dir = var_directories[var]
                    print (var, mds_var_dir)
                            
                    mds_file = list(mds_var_dir.glob(var + '*' + str(cur_ts).zfill(10) + '*.data'))
                    
                    if len(mds_file) != 1:
                        print('invalid # of mds files')
                        print(mds_file)
                    else:
                        mds_file = mds_file[0]          
                          
                        if grouping_dim == '2D':
                            F = ecco.read_llc_to_tiles(mds_var_dir, mds_file,\
                                                       llc=90, skip=0,\
                                                       nk=1, nl=1, 
                              	      filetype = '>f', less_output = True, 
                                      use_xmitgcm=False)
                            
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
                                                     operation='mean', allow_nearest_neighbor=True)
                            
                            
                            F_ll_masked = np.expand_dims(F_ll * land_mask_ll[0,:],0)
                            
                            
            
                            F_DA = xr.DataArray(F_ll_masked, \
                                                coords=[[record_end_time],\
                                                        target_grid_lats_1D,\
                                                        target_grid_lons_1D], \
                                                dims=["time", "latitude","longitude"])     
                            
                            
                        if grouping_dim == '3D':
                        
                            F = ecco.read_llc_to_tiles(mds_var_dir, mds_file, \
                                                       llc=90, skip=0, nk=50,\
                                                       nl=1, 
                              	      filetype = '>f', less_output = True, 
                                      use_xmitgcm=False)
                      
    
                            F_ll = np.zeros((50,360,720))
                            
                            for k in range(max_k):
                                print(var,k)
                                F_k = F[k]
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
                            F_ll_masked = np.expand_dims(F_ll * land_mask_ll, 0)
            
                            Z = ecco_grid.Z.values
                            
                            F_DA = xr.DataArray(F_ll_masked, \
                                                coords=[[record_end_time], \
                                                        Z, \
                                                        target_grid_lats_1D,\
                                                        target_grid_lons_1D], \
                                                dims=["time", "Z", "latitude","longitude"])     
        
                        ### END 2D VS. 3D
                        # grouping dim
                        
                        # ADD TIME STEP COORDINATE
                        F_DA=F_DA.assign_coords({"time_step": (("time"), [np.int32(cur_ts)])})
            
                        # assign name to data array
                        F_DA.name = var
                        
                        # cast to appropriate precision
                        F_DA = F_DA.astype(array_precision)
                        
                        # replace nan with fill value
                        F_DA.values = np.where(np.isnan(F_DA.values), \
                                               netcdf_fill_value, F_DA.values)
                    
                        # valid min max
                        F_DA.attrs['valid_min'] = \
                            F_ll_masked[np.where(~np.isnan(F_ll_masked))].min()
                        F_DA.attrs['valid_max'] = \
                            F_ll_masked[np.where(~np.isnan(F_ll_masked))].max()
                        
                        # convert to dataset
                        F_DS = F_DA.to_dataset()
                        
                        # assign lat and lon bounds
                        F_DS=F_DS.assign_coords({"latitude_bnds": (("latitude","nv"), lat_bounds)})
                        F_DS=F_DS.assign_coords({"longitude_bnds": (("longitude","nv"), lon_bounds)})
                                   
                        # if 3D assign depth bounds
                        if grouping_dim == '3D':
                            F_DS = F_DS.assign_coords({"Z_bnds": (("Z","nv"), depth_bounds)})
                        
                        # add appropriate time bounds.
                        if 'AVG' in output_freq_code:
                            tb, ct = \
                                ecco.make_time_bounds_and_center_times_from_ecco_dataset(F_DS,\
                                                                                         output_freq_code)
                            tb.time.values[0] = record_start_time
                            
                            #F_DS.time.values[0] = ct
                            F_DS = xr.merge((F_DS, tb))
                            F_DS = F_DS.set_coords('time_bnds')
                            F_DS_vars.append(F_DS)
                            F_DS.time.values[0] = ct
                            
                # merge the data arrays to make one DATASET
                G = xr.merge((F_DS_vars))
                
                
                # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
                print('\n... adding metadata specific to the variable')
                metadata_variable_latlon = metadata['ECCOv4r4_variables_on_latlon_grid_metadata']
                metadata_variable_native = metadata['ECCOv4r4_variables_on_native_grid_metadata']
                
                data_vars = G.data_vars
                keys_to_exclude = ['grid_dimension','name', 'GCMD_keywords',\
                                   'comments_1', 'comments_2', 'internal_note','internal note']
                for var in data_vars:
                    print("\n# " + var)
    
                    mv = find_metadata_in_json_dictionary(var, 'name', metadata_variable_latlon)
                    
                    if len(mv) == 0:
                        print('\t--> no metadata found in the latlon variable list, searching native list')
                        mv = find_metadata_in_json_dictionary(var, 'name', metadata_variable_native)
                            
                        if len(mv) == 0:
                            print('\t !!! NO METADATA FOUND')
                            sys.exit()
                             
                    else:
                        print('\tmetadata found in the latlon variable list')
                        
                    # loop through each key, add if not on exclude list
                    for m_key in sorted(mv.keys()):
                        if m_key not in keys_to_exclude:
                            G[var].attrs[m_key] = mv[m_key]
                            print('\t',m_key, ':', mv[m_key])
                            
                    # merge the two comment fields (both *MUST* be present)
                    if mv['comments_1'][-1] == '.':
                        G[var].attrs['comment'] =  mv['comments_1'] + ' ' + mv['comments_2']
                    else:
                        G[var].attrs['comment'] =  mv['comments_1'] + '. ' + mv['comments_2']
                        
                    print('\t','comment', ':', G[var].attrs['comment'])
                    
                    # Get GCMD keywords                
                    gcmd_keywords = mv['GCMD_keywords'].split(',')
                    
                    print('\t','GCMD keywords : ', gcmd_keywords)
                    
                    for gcmd_keyword in gcmd_keywords:
                        grouping_gcmd_keywords.append(gcmd_keyword.strip())
                        
                    
                                  
                # ADD COORDINATE SPECIFIC METADATA TO COORDINATE ATTRIBUTES (DATA ARRAYS)
                print('\n... adding metadata specific to the coordinate attributes')            
                metadata_coord_latlon = metadata['ECCOv4r4_common_coordinate_metadata']
    
                # don't include these helper fields in the attributes
                keys_to_exclude = ['grid_dimension','name']
                
                for coord in G.coords:
                    print("\n# " + coord)
                    mv = find_metadata_in_json_dictionary(coord, 'name',\
                                                          metadata_coord_latlon)
                    
                    if len(mv) == 0:
                        print(coord, 'NO METADATA FOUND')
                    else:
                        for m_key in sorted(mv.keys()):
                            if m_key not in keys_to_exclude:
                                G[coord].attrs[m_key] = mv[m_key]
                                print('\t',m_key, ':', mv[m_key])
                  
                    
                print("\n... adding ECCOv4 common metadata")
                # ADD ECCOV4 COMMON METADATA TO THE DATASET ATTRS
                for mc in metadata['ECCOv4r4_common_metadata']:
                   # print(mc)
                    mname = mc['name']
                    mtype = mc['type']
                    
                    add_field = True
                    
                    if 'grid_dimension' in mc.keys():
                        gd = mc['grid_dimension']
                        
                        if grouping_dim not in gd:
                            add_field = False
                            
                    if add_field == True:
                        if mtype == 's':
                            G.attrs[mname] = mc['value']
                        elif mtype == 'f':
                            G.attrs[mname] = float(mc['value'])
                        elif mtype == 'i':
                            G.attrs[mname] = np.int32(mc['value'])
                        else:
                            print('INVALID MTYPE ! ', mtype)
                    else:
                        print('\t> not adding ', mc)
                        
                print("\n... adding common metadata for latlon datasets")
                # ADD COMMON METADATA for latlon dataset to the DATASET attrs
                for mc in metadata['ECCOv4r4_common_metadata_for_latlon_datasets']:
                   # print(mc)
                    mname = mc['name']
                    mtype = mc['type']
                    
                    add_field = True
                    
                    if 'grid_dimension' in mc.keys():
                        gd = mc['grid_dimension']
                        
                        if gd == '3D' and grouping_dim == '2D':
                            add_field = False
                            
                        elif gd == '2D' and grouping_dim == '1D' :
                            add_field = False
                            
                        elif gd == '3D' and grouping_dim == '3D' :
                            add_field = True
                            
                    if add_field == True:
                        if mtype == 's':
                            G.attrs[mname] = mc['value']
                        elif mtype == 'f':
                            G.attrs[mname] = float(mc['value'])
                        elif mtype == 'i':
                            G.attrs[mname] = np.int32(mc['value'])
                    else:
                        print('\t> not adding ', mc)
                                     
                
                print('\n... adding metadata associated with the dataset grouping')
                # ADD METADATA ASSOCIATED WITH THE DATASET GROUPING TO THE DATASET ATTRS)
                G.attrs['title'] = grouping['name']
            
                if 'AVG' in output_freq_code:
                    G.attrs['time_coverage_start'] = str(G.time_bnds.values[0][0])[0:19]
                    G.attrs['time_coverage_end'] = str(G.time_bnds.values[0][1])[0:19]
        
                else:
                    G.attrs['time_coverage_start'] = str(G.time.values[0])[0:19]
                    G.attrs['time_coverage_end'] = str(G.time.values[0])[0:19]
        
    
                G.attrs['date_created'] = datetime.datetime.now().isoformat()[0:19]
                G.attrs['date_modified'] = datetime.datetime.now().isoformat()[0:19]
                G.attrs['date_metadata_modified'] = datetime.datetime.now().isoformat()[0:19]
                G.attrs['date_issued'] = datetime.datetime.now().isoformat()[0:19]
                   
                # add coordinate attribute to the variables
                coord_attrs, coord_G= get_coordinate_attribute_to_data_vars(G)
                for coord in coord_attrs.keys():
                    print(coord)
                    coord_attrs[coord] = coord_attrs[coord].split('time_step')[0].strip()
                    
                # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
                dv_encoding = dict()
                for dv in G.data_vars:
                    dv_encoding[dv] =  {'zlib':True, \
                                        'complevel':5,\
                                        'shuffle':True,\
                                        '_FillValue':netcdf_fill_value}
                    
                    G[dv].encoding['coordinates'] = coord_attrs[dv]
                     
                    
                # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
                coord_encoding = dict()
                
                for coord in G.coords:
                    coord_encoding[coord] = {'_FillValue':None}
                    
                    if coord == 'time':
                        coord_encoding[coord] = {'_FillValue':None,\
                                      'dtype':'int32',\
                                      'units':time_encoding_start}
                    elif coord == 'time_bnds':
                        coord_encoding[coord] = {'_FillValue':None,\
                                      'dtype':'int32',\
                                      'units':time_encoding_start}
                    elif coord == 'time_step':
                        coord_encoding[coord] = {'_FillValue':None,\
                                      'dtype':'int32'}
                    
                    elif 'latitude' in coord:
                        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
        
                    elif 'longitude' in coord:
                        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                        
                    elif 'Z' in coord:
                        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                        
                # MERGE ENCODINGS
                encoding = {**dv_encoding, **coord_encoding} 
             
                # MERGE GCMD KEYWORDS
                common_gcmd_keywords = G.keywords.split(',')
                gcmd_keywords_list = set(grouping_gcmd_keywords + common_gcmd_keywords)
                
                gcmd_keyword_str = ''
                for gcmd_keyword in gcmd_keywords_list:
                    if len(gcmd_keyword_str) == 0:
                        gcmd_keyword_str = gcmd_keyword
                    else:
                        gcmd_keyword_str += ', ' + gcmd_keyword
                        
                print(gcmd_keyword_str)
                G.attrs['keywords'] = gcmd_keyword_str
                
                #
                # ADD FINISHING TOUCHES 
                
                print('\n... adding uuid')
                G.attrs['uuid'] = str(uuid.uuid1())
                
                # --- 
                print('\n... adding time coverage duration, resolution')
                
                # Add time encoding and time long name
                #G.time.encoding['units'] = time_encoding_start
                #G.time_step.encoding['units'] = 'hours since 1992-01-01 12:00:00'
                
                if 'AVG' in output_freq_code:
                    G.time.attrs['long_name'] = 'center time of averaging period'
                    #G.time_bnds.encoding['units'] = time_encoding_start
                    #G.time.values[0] = tb.time_bnds.values[0,0]
                else:
                    G.time.attrs['long_name'] = 'snapshot time'
                    
                # set averaging period duration and resolution
                if output_freq_code == 'AVG_MON':
                    G.attrs['time_coverage_duration'] = 'P1M'
                    G.attrs['time_coverage_resolution'] = 'P1M'
            
                    #F_DS.time.values[0] = tb.time_bnds.values[0,0]
                    date_str = str(np.datetime64(G.time.values[0],'M'))
                    ppp_tttt = 'mon_mean'
                    
                # --- AVG DAY
                elif output_freq_code == 'AVG_DAY':
                    G.attrs['time_coverage_duration'] = 'P1D'
                    G.attrs['time_coverage_resolution'] = 'P1D'
            
                    date_str = str(np.datetime64(G.time.values[0],'D'))
                    ppp_tttt = 'day_mean'
                    
                # --- SNAPSHOT
                elif output_freq_code == 'SNAPSHOT':
                    G.attrs['time_coverage_duration'] = 'P0S'
                    G.attrs['time_coverage_resolution'] = 'P0S'
            
                    # convert from oroginal
                    #   '1992-01-16T12:00:00.000000000'
                    # to new format 
                    # '1992-01-16T120000'
                    date_str = str(G.time.values[0])[0:19].replace(':','')
                    ppp_tttt = 'snap'

                # construct filename
                filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + \
                    '_ECCO_V4r4_latlon_0p50deg.nc'
           
                # SAVE 
                        
                G.attrs['product_name'] = filename
                G.attrs['summary'] = dataset_description + ' ' + G.attrs['summary']
                
                print('\n... creating filename ', filename)
                netcdf_output_filename = output_dir / filename

                # get podaac metadata based on filename               
                metadata_modifiers = determine_metadata_modifiers(filename)

                # apply podaac metadata based on filename
                G = apply_metadata_modifiers(G, metadata_modifiers)

                print('\n... sorting attributes')
                sort_all_attrs(G)
                G.attrs["coordinates_comment"] = "Note: the global 'coordinates' attribute descibes auxillary coordinates."

                print('\n... saving to netcdf ', netcdf_output_filename)
                G.to_netcdf(netcdf_output_filename, encoding=encoding)
                G.close()
    #%%
    #  NetCDF
    
    # Filename format: vvvvvvvv_ppp_tttt_gggggg_YYYY-MM-DD-ECCO_V4r3_RRRRdeg.nc (max 42 char)
    
    #     vvvvvvvv - ECCO parameter or group name (3-??? char)
    #     ppp - day or mon (3 char) time period of averaging (only for daily and monthly means)
    #     tttt - mean or snap (4 char)
    #     YYYY - year (4 char)
    #     MM - month (2 char)
    #     DD - day (2 char, daily only)
    #     ECCO_V4r3 - release name (9 char)
    #     gggggg - latlon or native (6 char)
    #     RRRR - resolution in format 0pXX:  0p10= 1/10th degree, 0p16 = 1/6th degree, 0p50 = 1/2 degree, 0p33 = 1/3 degree
    #     maximum possible length = XX characters
    
    # Interpolated Daily Mean Example: SIarea_day_mean_latlon_1992-01-01_ECCO_V4r3_0p50deg.nc
    
    # Native Monthly Mean Example: SIarea_mon_mean_native_1992-01_ECCO_V4r3_0p50deg.nc
    
    # Native Daily Snapshot Example: SIarea_snap_native_1992-01-01_ECCO_V4r3_0p50deg.nc
    
       
    
#%%
#    %%
#
###%%
##
###%%
##
##grid_types = []
###grid_types.append('native')
##grid_types.append('latlon')
##
##tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12]
#
## params for lat-lon files
#radius_of_influence = 120000
#
#new_grid_delta_lat = data_res
#new_grid_delta_lon = data_res
#
#new_grid_min_lat = -90+new_grid_delta_lat/2
#new_grid_max_lat = 90-new_grid_delta_lat/2
#
#new_grid_min_lon = -180+new_grid_delta_lon/2
#new_grid_max_lon = 180-new_grid_delta_lon/2
#
#
#x_c = ecco_grid.XC.values
#y_c = ecco_grid.YC.values
#
#
##%%
#
#land_mask_ll_alt = np.zeros((50, dims[1],dims[0]))
#
#for k in range(50):
#    print(k)
#    new_grid_lon, new_grid_lat, land_mask_ll_alt[k,:,:] =\
#            ecco.resample_to_latlon(x_c, \
#                                    y_c, \
#                                    land_mask[k,:],\
#                                    new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
#                                    new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
#                                    fill_value = np.NaN, \
#                                    mapping_method = 'nearest_neighbor',
#                                    radius_of_influence = 120000)   
#
#
#land_mask_ll_alt = np.where(land_mask_ll_alt ==1, 1, 0)
##%% 
#
#test_count = np.array(range(13*90*90))
#
#test1 = ea.transform_to_target_grid(source_indices_within_target_radius_i,
#                             num_source_indices_within_target_radius_i,
#                             nearest_source_index_to_target_index_i,
#                             test_count.ravel(), target_grid_shape,\
#                             operation='mean', allow_nearest_neighbor=True)
#
#
#new_grid_lon, new_grid_lat, test2 = \
#            ecco.resample_to_latlon(x_c, \
#                                    y_c, \
#                                    test_count,\
#                                    new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
#                                    new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
#                                    fill_value = np.NaN, \
#                                    mapping_method = 'nearest_neighbor',
#                                    radius_of_influence = 120000)   
#
#test1 = np.flipud(test1)
#plt.figure(10,clear=True)
#plt.subplot(221);plt.imshow(test1,origin='lower')
#plt.subplot(222);plt.imshow(test2,origin='lower')
#plt.subplot(223);plt.imshow(test2-test1,vmin=-1,vmax=1,origin='lower')
##
#testd = np.where(np.abs(test2-test1) > 0, 1, 0)
#plt.subplot(224);plt.imshow(testd, origin='lower')
#
#
##%%
#
#plt.figure(11,clear=True)
#A = np.flipud(land_mask_ll[0,:])
#B = land_mask_ll_alt[0,:]
#plt.subplot(221);plt.imshow(A, origin='lower')
#plt.subplot(222);plt.imshow(B, origin='lower')
#plt.subplot(223);plt.imshow(A-B,vmin=-1,vmax=1,origin='lower')
##
#testd = np.where(np.abs(A-B) > 0, 1, 0)
#plt.subplot(224);plt.imshow(testd,origin='lower')


#%%
    
#
#source_indices_within_target_radius_i, \
#num_source_indices_within_target_radius_i,\
#nearest_source_index_to_target_index_i = grid_mappings_k[0]
#    
#field = ecco_grid.Depth.values[wet_pts_k[0]]
#    
#test_d =  \
#        ea.transform_to_target_grid(source_indices_within_target_radius_i,
#                             num_source_indices_within_target_radius_i,
#                             nearest_source_index_to_target_index_i,
#                             field,\
#                             target_grid_shape,\
#                             operation='mean', allow_nearest_neighbor=True)
#    
#plt.figure(1,clear=True);
#plt.subplot(311);plt.imshow(test_d,origin='lower');plt.colorbar();
#plt.subplot(312);plt.imshow(test_d,origin='lower',vmin=0,vmax=1500);plt.colorbar();
#plt.xlim([370/2, 410/2]);plt.ylim([420/2,500/2])
#plt.subplot(313);plt.imshow(test_d,origin='lower',vmin=0,vmax=500);plt.colorbar();
#plt.xlim([460/2, 530/2]);plt.ylim([610/2,660/2])
#
#
##%%
#source_indices_within_target_radius_i, \
#num_source_indices_within_target_radius_i,\
#nearest_source_index_to_target_index_i = grid_mappings_all
#    
#field = ecco_grid.Depth.values.ravel()
#    
#test_c =  \
#        ea.transform_to_target_grid(source_indices_within_target_radius_i,
#                             num_source_indices_within_target_radius_i,
#                             nearest_source_index_to_target_index_i,
#                             field,\
#                             target_grid_shape,\
#                             operation='mean', allow_nearest_neighbor=True)
#    
#plt.figure(2,clear=True);
#plt.subplot(311);plt.imshow(test_c,origin='lower');plt.colorbar();
#plt.subplot(312);plt.imshow(test_c,origin='lower',vmin=0,vmax=1500);plt.colorbar();
#plt.xlim([370/2, 410/2]);plt.ylim([420/2,500/2])
#plt.subplot(313);plt.imshow(test_c,origin='lower',vmin=0,vmax=500);plt.colorbar();
#plt.xlim([460/2, 530/2]);plt.ylim([610/2,660/2])