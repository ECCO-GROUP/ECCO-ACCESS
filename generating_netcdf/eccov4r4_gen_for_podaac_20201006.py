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
import matplotlib.pyplot as plt
import pyresample as pr
import uuid


#%%

def get_coordinate_attribute_to_data_vars(G):
    coord_attr = dict()
    
    dvs = list(G.data_vars)
    for dv in dvs:
        coord_attr[dv] = ' '.join([str(elem) for elem in G[dv].coords])
        
    coord_G = ' '.join([str(elem) for elem in G.coords])
    
    return coord_attr, coord_G

        
def find_metadata_in_json_dictionary(var, key, metadata):
    for m in metadata:        
        if m[key] == var:
            print(m)
            return m
    return []


def sort_attrs(attrs):
  
    od = OrderedDict()
    
    keys = sorted(list(attrs.keys()),key=str.casefold)
    
    for k in keys:
        od[k] = attrs[k]

    return od

def sort_all_attrs(G):
    for coord in list(G.coords):
        print(coord)
        new_attrs = sort_attrs(G[coord].attrs)
        G[coord].attrs = new_attrs
        
    for dv in list(G.data_vars):
        print(dv)
        new_attrs = sort_attrs(G[dv].attrs)
        G[dv].attrs = new_attrs
        
    new_attrs = sort_attrs(G.attrs)
    G.attrs = new_attrs
        
        

#%%

# Define precision of output files, float32 is standard
# ------------------------------------------------------
array_precision = np.float32

# Define fill values for binary and netcdf
# ---------------------------------------------
if array_precision == np.float32:
    binary_output_dtype = '>f4'
    netcdf_fill_value = nc4.default_fillvals['f4']

elif array_precision == np.float64:
    binary_output_dtype = '>f8'
    netcdf_fill_value = nc4.default_fillvals['f8']


# ECCO always uses -9999 for missing data.
binary_fill_value = -9999


######################################################################
importlib.reload(ecco)

start_time = np.datetime64('1992-01-01T12:00:00')


## OUTPUT DIRECTORY
output_dir = Path('/home/ifenty/tmp/v4r4_nc_output_20201006')

if not output_dir.exists():
    try:
        output_dir.mkdir()
    except:
        print ('cannot make %s ' % output_dir)




## GRID DIR -- MAKE SURE USE GRID FIELDS WITHOUT BLANK TILES!!
mds_grid_dir = Path('/Users/ifenty/tmp/no_blank_all')

## METADATA
#metadata_json_dir = Path('/Users/ifenty/git_repos/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')
metadata_json_dir = Path('/home/ifenty/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')

metadata_fields = ['ECCOv4r4_common_metadata', 
                   'ECCOv4r4_common_metadata_for_latlon_datasets',
                   'ECCOv4r4_common_metadata_for_native_grid_datasets',
                   'ECCOv4r4_coordinate_variable_metadata_for_latlon_datasets',
                   'ECCOv4r4_dataset_groupings_for_latlon_product',
                   'ECCOv4r4_variables_on_latlon_grid_metadata',
                   'ECCOv4r4_variables_on_native_grid_metadata',
                   'ECCOv4r4_common_GCMD_keywords',
                   'ECCOv4r4_variable_GCMD_keywords',
                   'ECCOv4r4_podaac_datasets']


# load METADATA
metadata = dict()

for mf in metadata_fields:
    mf_e = mf + '.json'
    print(mf_e)
    with open(str(metadata_json_dir / mf_e), 'r') as fp:
        metadata[mf] = json.load(fp)
    
        
        
groupings = metadata['ECCOv4r4_dataset_groupings_for_latlon_product']

#%%
# grid dir
ecco_grid_dir = '/home/ifenty/data/grids/ecco_pipeline_grids'

##ecco_grid_dir = '/Users/ifenty/inSync Share/Projects/ECCOv4/Release4/nctiles_grid/'
ecco_grid_dir = '/home/ifenty/ian1/ifenty/ECCOv4/Version4/Release4/nctiles_grid'

ecco_grid = ecco.load_ecco_grid_nc(ecco_grid_dir, 'ECCO-GRID.nc')

# Define the 'swath' (in the terminology of the pyresample module)
# as the lats/lon pairs of the model grid
# The routine needs the lats and lons to be one-dimensional vectors.
source_grid  = \
    pr.geometry.SwathDefinition(lons=ecco_grid.XC.values.ravel(), 
                                lats=ecco_grid.YC.values.ravel())
    
source_grid_min_L = np.min([float(ecco_grid.dyG.min().values), float(ecco_grid.dxG.min().values)])
source_grid_max_L = np.max([float(ecco_grid.dyG.max().values), float(ecco_grid.dxG.max().values)])

print(int(source_grid_min_L))
print(int(source_grid_max_L))

#%%


#%%

# TARGET GRID
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


    
target_grid_lons_1D = target_grid_lons[0,:]
target_grid_lats_1D = target_grid_lats[:,0]


lat_lon_grid_area = ea.area_of_latlon_grid(-180, 180, -90, 90, data_res, data_res)

target_grid_radius = 0.5*np.sqrt(lat_lon_grid_area).ravel()


source_indices_within_target_radius_i,\
num_source_indices_within_target_radius_i,\
nearest_source_index_to_target_index_i = \
    ea.find_mappings_from_source_to_target(source_grid,\
                                           target_grid,\
                                           target_grid_radius, \
                                           source_grid_min_L, \
                                           source_grid_max_L)
    
#   
    
#%%
test_d =  \
        ea.transform_to_target_grid(source_indices_within_target_radius_i,
                             num_source_indices_within_target_radius_i,
                             nearest_source_index_to_target_index_i,
                             ecco_grid.Depth.values.ravel(), \
                             target_grid_lons.shape,\
                             operation='mean', allow_nearest_neighbor=True)
    
plt.figure(1,clear=True);
plt.subplot(311);plt.imshow(test_d,origin='lower');plt.colorbar();
plt.subplot(312);plt.imshow(target_grid_lats,origin='lower');plt.colorbar()
plt.subplot(313);plt.imshow(target_grid_lons,origin='lower');plt.colorbar()

#%%
    
## MAKE A 3D LAND MASK IN LAT-LON GRID   

ecco_land_mask = np.where(ecco_grid.hFacC.values == 0, 0, 1)

target_grid_shape = lat_lon_grid_area.shape

land_mask_ll = np.zeros((50, target_grid_shape[0], target_grid_shape[1]))

for k in range(50):
    print(k)
    
    source_field = ecco_land_mask[k,:].ravel()
    
    land_mask_ll[k,:] =  \
        ea.transform_to_target_grid(source_indices_within_target_radius_i,
                             num_source_indices_within_target_radius_i,
                             nearest_source_index_to_target_index_i,
                             source_field, target_grid_shape,\
                             operation='mean', allow_nearest_neighbor=True)


#%%
plt.figure(2,clear=True);
plt.subplot(311);plt.imshow(land_mask_ll[0,:],origin='lower');plt.colorbar();
plt.subplot(312);plt.imshow(land_mask_ll[25,:],origin='lower');plt.colorbar()
plt.subplot(313);plt.imshow(land_mask_ll[45,:],origin='lower');plt.colorbar()


land_mask_ll = np.where(land_mask_ll ==1,1,np.nan)

plt.figure(3,clear=True);
plt.subplot(311);plt.imshow(land_mask_ll[0,:],origin='lower');plt.colorbar();
plt.subplot(312);plt.imshow(land_mask_ll[25,:],origin='lower');plt.colorbar()
plt.subplot(313);plt.imshow(land_mask_ll[45,:],origin='lower');plt.colorbar()
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
    

depth_bounds = np.zeros((50,2))
tmp = np.cumsum(ecco_grid.drF.values)

for k in range(50):
    if k == 0:
        depth_bounds[k,0] = 0.0
    else:
        depth_bounds[k,0] = tmp[k-1]
    depth_bounds[k,1] = tmp[k]
    
#%%
###################################################################################################

avgs  = ['AVG_DAY']#,'AVG_MON']

diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/diags_all')
for avg in avgs:
    if avg == 'AVG_DAY':
        ## AVERAGING TIME PERIOD
        output_freq_code = 'AVG_DAY' ## 'AVG_DAY' or 'SNAPSHOT'
       # mds_diags_root_dir = Path('/Users/ifenty/ECCOv4/Release4/forward_output/v4r4_nc_output_20200729_2311/diags_daily')
        mds_diags_root_dir = diags_root / 'diags_daily'
    elif avg == 'AVG_MON':
        output_freq_code = 'AVG_MON' ## 'AVG_DAY' or 'SNAPSHOT'
        #mds_diags_root_dir = Path('/Users/ifenty/ECCOv4/Release4/forward_output/v4r4_nc_output_20200729_2311/diags_monthly')
        mds_diags_root_dir = diags_root / 'diags_monthly'

    
    ## TIME STEPS TO LOAD
    #all_avail_time_steps = ecco.get_time_steps_from_mds_files(mds_diags_root_dir / 'ETAN', 'ETAN')
    #pprint(all_avail_time_steps)
    
    #time_steps_to_load = all_avail_time_steps[:2]
    time_steps_to_load = [12]
    print('\nloading time steps')
    pprint(time_steps_to_load)
      
    
    #######################3
    if 'AVG_MON' in output_freq_code:
        period_suffix = 'mon_mean'
    
    elif 'AVG_DAY' in output_freq_code:
        period_suffix = 'day_mean'
    
    elif 'SNAPSHOT' in output_freq_code:
        period_suffix = 'inst'
     
    field_paths = np.sort(list(mds_diags_root_dir.glob('*' + period_suffix + '*')))
    
        
    ## load variable file and directory names
    print (len(field_paths))
    all_field_names = []
    
    for f in field_paths:
        all_field_names.append(f.name)
    
    print (all_field_names)
    
    
    for grouping in groupings[0:1]:
    
        grouping_dim = grouping['dimension']
        
        tmp = grouping['fields'].split(',')
        vars_to_load  = []
        for var in tmp:
            vars_to_load.append(var.strip())
            
       
        var_directories = dict()
        
        for var in vars_to_load:
            num_matching_dirs = 0
            for fp in field_paths:
                if var in str(fp):
                    #print(var, fp) 
                    var_directories[var] = fp
                    num_matching_dirs += 1
            if num_matching_dirs == 0:
                print('>>>>>> no match found for ', var)
            elif num_matching_dirs > 1 :
                print('>>>>>> more than one matching dir for ', var)
        
        for var in vars_to_load:
            print(var, var_directories[var])
            
        
        
        for cur_ts in time_steps_to_load:
            
            time_delta = np.timedelta64(cur_ts, 'h')
                    
            cur_time = start_time + time_delta
            times = [pd.to_datetime(str(cur_time))]
            
                
            F_DS_vars = []
            
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
                    
                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file, llc=90, skip=0, nk=1, nl=1, 
                          	      filetype = '>f', less_output = False, 
                                  use_xmitgcm=False)
                  
                        
                        F_ll =  \
                            ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                 num_source_indices_within_target_radius_i,
                                 nearest_source_index_to_target_index_i,
                                 F.ravel(), target_grid_shape,\
                                 operation='mean', allow_nearest_neighbor=True)
        
        
        
#                        new_grid_lon, new_grid_lat, F_ll =\
#                                ecco.resample_to_latlon(x_c, \
#                                                        y_c, \
#                                                        F,\
#                                                        new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
#                                                        new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
#                                                        fill_value = np.NaN, \
#                                                        mapping_method = 'nearest_neighbor',
#                                                        radius_of_influence = 120000)               
                        
                        
                        F_ll_masked = np.expand_dims(F_ll * land_mask_ll[0,:],0)
                        
        
        
                        F_DA = xr.DataArray(F_ll_masked, \
                                            coords=[times, target_grid_lats_1D,\
                                                    target_grid_lons_1D], \
                                            dims=["time", "latitude","longitude"]     )     
    
                    if grouping_dim == '3D':
                    
                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file, llc=90, skip=0, nk=50, nl=1, 
                          	      filetype = '>f', less_output = False, 
                                  use_xmitgcm=False)
                  
                        
                        F_ll = np.zeros((50,360,720))
                        
                        for k in range(50):
                            print(var,k)
                            
                            
                        
                            F_ll[k,:] =  \
                                ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                     num_source_indices_within_target_radius_i,
                                     nearest_source_index_to_target_index_i,
                                     F[k,:].ravel(), target_grid_shape,\
                                     operation='mean', allow_nearest_neighbor=True)
        
        
                            
#                            new_grid_lon, new_grid_lat, F_ll[k,:] =\
#                                    ecco.resample_to_latlon(x_c, \
#                                                            y_c, \
#                                                            F[k,:],\
#                                                            new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
#                                                            new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
#                                                            fill_value = np.NaN, \
#                                                            mapping_method = 'nearest_neighbor',
#                                                            radius_of_influence = 120000)               
#                        
                        
                        F_ll_masked = np.expand_dims(F_ll * land_mask_ll, 0)
                        
        
                        Z = ecco_grid.Z.values
                        
                        F_DA = xr.DataArray(F_ll_masked, \
                                            coords=[times, Z, \
                                                    target_grid_lats_1D,\
                                                    target_grid_lons_1D], \
                                            dims=["time", "Z", "latitude","longitude"]     )     
    
                                   
                    
                    # grouping dim
                    
                    F_DA=F_DA.assign_coords({"time_step": (("time"), [np.int32(cur_ts)])})
        
                    # assign name to data array
                    F_DA.name = var
                    
                    # cast to appropriate precision
                    F_DA = F_DA.astype(array_precision)
                    
                    # replace nan with fill value
                    F_DA.values = np.where(np.isnan(F_DA.values), netcdf_fill_value, F_DA.values)
                    
                    F_DS = F_DA.to_dataset()
                    
                    F_DS=F_DS.assign_coords({"latitude_bnds": (("latitude","nv"), lat_bounds)})
                    F_DS=F_DS.assign_coords({"longitude_bnds": (("longitude","nv"), lon_bounds)})
                        
                        
                    if grouping_dim == '3D':
                        F_DS=F_DS.assign_coords({"Z_bnds": (("Z","nv"), depth_bounds)})
                        
    
                    if 'AVG' in output_freq_code:
                        tb, ct = ecco.make_time_bounds_and_center_times_from_ecco_dataset(F_DS, output_freq_code)
                        
                        F_DS.time.values[0] = ct
                        F_DS = xr.merge((F_DS, tb))
                        F_DS = F_DS.set_coords('time_bnds')
                        
                        F_DS_vars.append(F_DS)
                    
                   
            # merge the data arrays to make one DATASET
            G = xr.merge((F_DS_vars))
            
            
            
            
            # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
            metadata_variable_latlon = metadata['ECCOv4r4_variables_on_latlon_grid_metadata']
            metadata_variable_native = metadata['ECCOv4r4_variables_on_native_grid_metadata']
            
            data_vars = G.data_vars
            keys_to_exclude = ['grid_dimension','name', 'gcmd_keywords','comments_1','comments_2']

            for var in data_vars:
            
                mv = find_metadata_in_json_dictionary(var, 'name', metadata_variable_latlon)
                
                if len(mv) == 0:
                    mv = find_metadata_in_json_dictionary(var, 'name', metadata_variable_native)

                if len(mv) == 0:
                    print('NO METADATA FOUND')
                    
                # loop through each key, add if not on exclude list
                for m_key in sorted(mv.keys()):
                    if m_key not in keys_to_exclude:
                        G[var].attrs[m_key] = mv[m_key]
                
                # merge the two comment fields (both *MUST* be present)
                G[var].attrs['comment'] =  mv['comments_1'] + '. ' + mv['comments_2']
                
          
            # ADD COORDINATE SPECIFIC METADATA TO COORDINATE ATTRIBUTES (DATA ARRAYS)
            metadata_coord_latlon = metadata['ECCOv4r4_coordinate_variable_metadata_for_latlon_datasets']
            
            # don't include these helper fields in the attributes
            keys_to_exclude = ['grid_dimension','name']
            
            for coord in G.coords:
                
                mv = find_metadata_in_json_dictionary(coord, 'name', metadata_coord_latlon)
                
                if len(mv) == 0:
                    print(coord, 'NO METADATA FOUND')
                else:
                    for m_key in sorted(mv.keys()):
                        if m_key not in keys_to_exclude:
                            G[coord].attrs[m_key] = mv[m_key]
                            
              
            # ADD ECCOV4 COMMON METADATA TO THE DATASET ATTRS
            for mc in metadata['ECCOv4r4_common_metadata']:
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
                    print('>>>>>>>> not adding ')
                    print(mc)
              
                    
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
                        add_field = False
                        
                if add_field == True:
                    if mtype == 's':
                        G.attrs[mname] = mc['value']
                    elif mtype == 'f':
                        G.attrs[mname] = float(mc['value'])
                    elif mtype == 'i':
                        G.attrs[mname] = np.int32(mc['value'])
                else:
                    print('>>>>>>>> not adding ')
                    print(mc)
                                 
            
            # Add common metadata keywords
            G.attrs['keywords'] = metadata['ECCOv4r4_common_GCMD_keywords'][0]['GCMDkeywords']
   
            # ADD METADATA ASSOCIATED WITH THE DATASET GROUPING TO THE DATASET ATTRS)
            G.attrs['title'] = grouping['name']
            G.attrs['summary'] = grouping['dataset_description']
         
            if 'AVG' in output_freq_code:
                G.attrs['time_coverage_start'] = str(G.time_bnds.values[0][0])
                G.attrs['time_coverage_end'] = str(G.time_bnds.values[0][1])
    
            else:
                G.attrs['time_coverage_start'] = str(G.time.values[0])
                G.attrs['time_coverage_end'] = str(G.time.values[0])
    
            G.attrs['date_modified'] = datetime.datetime.now().isoformat()
            G.attrs['date_metadata_modified'] = datetime.datetime.now().isoformat()
            G.attrs['date_issued'] = datetime.datetime.now().isoformat()
               
            # add coordinate attribute to the variables
            coord_attrs, coord_G= get_coordinate_attribute_to_data_vars(G)
    
            #G.attrs['coordinates'] = coord_G

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
                
                if 'time' in coord:
                    coord_encoding[coord] = {'_FillValue':None, 'dtype':'int32'}
                    
                if 'latitude' in coord:
                    coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
    
                if 'longitude' in coord:
                    coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                    
                if 'Z' in coord:
                    coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                    
            # MERGE ENCODINGS
            encoding = {**dv_encoding, **coord_encoding} 
         
            
            # ADD FINISHING TOUCHES AND MAKE FILENAME 
            # uuid
            
            G.attrs['uuid'] = str(uuid.uuid1())
            
            # --- AVG MON
            if output_freq_code == 'AVG_MON':
                
                G.attrs['time_coverage_duration'] = 'P1M'
                G.attrs['time_coverage_resolution'] = 'P1M'
        
                date_str = str(np.datetime64(G.time.values[0],'M'))
                ppp_tttt = 'mon_mean'
                filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + \
                    '_ECCO_V4r4_latlon_0p50deg.nc'
                    
                
            # --- AVG DAY
            elif output_freq_code == 'AVG_DAY':
                
                G.attrs['time_coverage_duration'] = 'P1D'
                G.attrs['time_coverage_resolution'] = 'P1D'
        
                date_str = str(np.datetime64(G.time.values[0],'D'))
                ppp_tttt = 'day_mean'
                filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + \
                    '_ECCO_V4r4_latlon_0p50deg.nc'
                    
       
            # --- SNAPSHOT
            elif output_freq_code == 'SNAPSHOT':
                
                G.attrs['time_coverage_duration'] = 'P0S'
                G.attrs['time_coverage_resolution'] = 'P0S'
        
                date_str = str(np.datetime64(G.time.values[0],'S'))
                ppp_tttt = 'snapshot'
                filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + \
                    '_ECCO_V4r4_latlon_0p50deg.nc'
       
            # SAVE 
            print(filename)
                    
            netcdf_output_filename = output_dir / filename
            
            sort_all_attrs(G)
            G.to_netcdf(netcdf_output_filename, encoding=encoding)
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
