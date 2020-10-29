"""
Created on Fri May 10 17:27:09 2019

@author: ifenty"""

import sys
import json
import numpy as np
from importlib import reload
sys.path.append('/home/ifenty/ECCOv4-py')
import ecco_v4_py as ecco
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
from pandas import read_csv
reload(ecco)                               

def find_all_time_steps(vars_to_load, var_directories):
    all_time_steps_var = dict()
    all_time_steps_all_vars = []
    
    for var in vars_to_load:
        all_time_steps_var[var] = []
        
        print(var, var_directories[var])
        all_files = np.sort(list(var_directories[var].glob('*data')))

        for file in all_files:
            ts = int(str(file).split('.')[-2])
            all_time_steps_all_vars.append(ts)
            all_time_steps_var[var].append(ts)
        
        
    unique_ts = np.unique(all_time_steps_all_vars)
    for var in vars_to_load:
        if np.array_equal(unique_ts, np.array(all_time_steps_var[var])):
            print('--- number of time steps equal ', var)
        else:
            print('--- number of time steps not equal ', var)
            sys.exit()
        
    # if we've made it this far, then we have a list of all
    # of the time steps to load in unique_ts
    print(unique_ts)
    all_time_steps = unique_ts
    return all_time_steps

#%%

def find_grouping_to_process_by_job_id(job_id, num_groupings:int):
    print('-- finding grouping to process')
    print('--- job id, num_groupings:', job_id, num_groupings)
    job_id_grouping = job_id % num_groupings
    return job_id_grouping
    
#%%
def find_time_steps_to_process_by_job_id(num_jobs:int, job_id:int, \
                                         num_groupings:int, all_time_steps):
# def find_time_steps_to_process_by_job_id(num_jobs, job_id, num_groupings, all_time_steps):

    print('\n-- find time steps to process by job id')
    print('--- this job ', job_id)
    print('--- num jobs ', num_jobs)
    print('--- num groupings ', num_groupings)
    print('--- num time steps ' , len(all_time_steps))

    print('--- job ids')
    job_ids = np.arange(num_jobs)
    pprint(job_ids)
    
    job_groupings = find_grouping_to_process_by_job_id(job_ids, num_groupings)
    print ('--- job groupings')
    print(job_groupings)
    
    num_jobs_per_grouping = dict()
    job_id_per_grouping = dict()
    for grouping in range(num_groupings):
        job_id_per_grouping[grouping] = np.where(job_groupings == grouping)[0]
        num_jobs_per_grouping[grouping] = len(job_id_per_grouping[grouping])
        
    print('--- num_jobs_per_grouping')
    pprint(num_jobs_per_grouping)

    print('--- job_id_per_gropuing')
    pprint(job_id_per_grouping)
    
    this_grouping = find_grouping_to_process_by_job_id(job_id, num_groupings)
    num_jobs_this_grouping = num_jobs_per_grouping[this_grouping]
    
    print('--- this grouping ', this_grouping)
    print('--- num jobs this grouping: ', num_jobs_this_grouping)
    
    cur_slot = np.where(job_id_per_grouping[this_grouping] == job_id)[0][0]
    print('--- which slot ', cur_slot)
    
    time_step_chunks = (np.linspace(0, len(all_time_steps), \
                                   num=num_jobs_this_grouping+1)).astype(int)
    
    print('--- time step chunks')
    pprint(time_step_chunks)
    time_steps_to_process = \
        all_time_steps[time_step_chunks[cur_slot]:time_step_chunks[cur_slot+1]]
    print('--- time steps to load')
    pprint(time_steps_to_process)
    
    return time_steps_to_process

#%%
def find_podaac_metadata(podaac_dataset_table, filename):
    """Return revised file metadata based on an input ECCO `filename`.
    
    This should consistently parse a filename that conforms to the 
    ECCO filename conventions and match it to a row in my metadata 
    table.
    
    """
    
    # Use filename components to find metadata row from podaac_dataset_table.
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
    
    print (head, tail)
    tail = tail.split("_ECCO_V4r4_")[1]
    
    # Get the filenames column from my table as a list of strings.
    names = podaac_dataset_table['DATASET.FILENAME']
    
    # Find the index of the row with a filename with matching head, tail chars.
    index = names.apply(lambda x: all([x.startswith(head), x.endswith(tail)]))
    
    # Select that row from podaac_dataset_table table and make a copy of it.
    metadata = podaac_dataset_table[index].iloc[0].to_dict()
    
#    gcmd_keywords = []
    # Select the gcmd keywords to match this dataset ShortName.
#    for prefix, keywords in __keywords__.items():
#        if metadata['DATASET.SHORT_NAME'].startswith(prefix):
#            gcmd_keywords = ", ".join([" > ".join(kw.values()) for kw in keywords])
#            break
    
#    print(gcmd_keywords)
    
    podaac_metadata = {
        'id': metadata['DATASET.PERSISTENT_ID'].replace("PODAAC-","10.5067/"),
        'metadata_link': f"https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName={metadata['DATASET.SHORT_NAME']}",
        'title': metadata['DATASET.LONG_NAME'],
    }
  
    print(metadata['DATASET.LONG_NAME'])
    return podaac_metadata

#%%
def apply_podaac_metadata(xrds, podaac_metadata):
    """Apply attributes podaac_metadata to ECCO dataset and its variables.
    
    Attributes that are commented `#` are retained with no modifications.
    Attributes that are assigned `None` are dropped.
    New attributes added to dataset.
    
    """
    # REPLACE GLOBAL ATTRIBUTES WITH NEW/UPDATED DICTIONARY.
    atts = xrds.attrs
    for name, modifier in podaac_metadata.items():
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

    return xrds  # Return the updated xarray Dataset.
    #return apply_podaac_metadata  # Return the function.



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
   
    
                        
def add_global_metadata(metadata, G, grouping_dim):
    
    # loop through pairs
    for mc in metadata:
        # get name and type
        mname = mc['name']
        mtype = mc['type']
        
        # by default add the key/pair
        add_field = True
        
        # unless it has as specific 'grid_dimension' associated 
        # with it. If so, then only add it if this dataset indicates
        # it is necessary.  for example, don't provide geospatial
        # depth information for 2D datasets
        if 'grid_dimension' in mc.keys():
            gd = mc['grid_dimension']
            
            if grouping_dim not in gd:
                add_field = False
        
        # if we do add the field, we have to convert to the 
        # appropriate data type
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
        
    return G

                    
def add_coordinate_metadata(metadata_dict, G):
    keys_to_exclude = ['grid_dimension','name']
    
    for coord in G.coords:
        
        print('\n### ', coord)
        # look for coordinate in metadat dictionary
        mv = find_metadata_in_json_dictionary(coord, 'name', metadata_dict)
        
        if len(mv) > 0:
            # if metadata for this coordinate is present
            # loop through all of the keys and if it is not
            # on the excluded list, add it
            for m_key in sorted(mv.keys()):
                if m_key not in keys_to_exclude:
                    G[coord].attrs[m_key] = mv[m_key]
                    print('\t',m_key, ':', mv[m_key])
        else:
            print('...... no metadata found in dictionary')
    
    return G
                
                
def add_variable_metadata(variable_metadata_dict, G, grouping_gcmd_keywords):

    # ADD VARIABLE METADATA  & SAVE GCMD KEYWORDS         
    keys_to_exclude = ['grid_dimension','name', 'GCMD_keywords', "variable_rename",\
                       'comments_1', 'comments_2', 'internal_note','internal note',\
                       'grid_location']

    for var in G.data_vars:
        print('\n### ', var)
        mv = find_metadata_in_json_dictionary(var, 'name', variable_metadata_dict)
    
        if len(mv) == 0:
            print('...... no metadata found in dictionary')
            
        else:
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
            
            # Get the GCMD keywords, these will be added into the global
            # attributes
            gcmd_keywords = mv['GCMD_keywords'].split(',')
            
            print('\t','GCMD keywords : ', gcmd_keywords)
            
            for gcmd_keyword in gcmd_keywords:
                grouping_gcmd_keywords.append(gcmd_keyword.strip())
    
    return G, grouping_gcmd_keywords    


def generate_netcdfs(output_freq_code, job_id:int, num_jobs:int, \
                     product_type,\
                     grouping_to_process='by_job',\
                     time_steps_to_process='by_job', \
                     debug_mode=False):

    #%%
    #
    #
    # output_freq_codes  one of either 'AVG_DAY' or 'AVG_MON'
    #
    # time_steps_to_process : one of 
    # 'all'
    # a list [1,3,4,...]
    # 'by_job'

    # grouping to process:
    #   can be either
    #   a single number
    #   'by_job' determine from job id and num_jobs


    print('\nBEGIN: generate_netcdfs')
    print('OFC',output_freq_code)
    print('JID',job_id)
    print('NJB',num_jobs)
    print('PDT', product_type)
    print('GTP',grouping_to_process)
    print('TSP',time_steps_to_process)
    print('DBG',debug_mode)
    print('\n')
    #%%
    # Define precision of output files, float32 is standard
    # ------------------------------------------------------
    array_precision = np.float32


    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999

    # num of depth levels
    nk=50

    ecco_start_time = np.datetime64('1992-01-01T12:00:00')

    # hack to ensure that time bounds and time use same 'encoding' in xarray
    #time_encoding_start = 'hours since 1992-01-01 12:00:00'
    # --- not necessary with "units" attribute defined in coordinate metadata.

    mapping_factors_dir = Path('/home/ifenty/tmp/ecco-v4-podaac-mapping-factors')
    #napping_factors_dir = Path('/nobackupp2/ifenty/podaac/lat-lon/mapping-factors')


    ## OUTPUT DIRECTORY
    output_dir_base = Path('/home/ifenty/tmp/v4r4_nc_output_20201028_native')
    #output_dir_base = Path('/nobackupp2/ifenty/podaac/')
    

    ## ECCO FIELD INPUT DIRECTORY 
    # model diagnostic output 
    # subdirectories must be 
    #  'diags_all/diag_mon'
    #  'diags_all/diag_mon'
    #  'diags_all/snap'
            
    diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/diags_all')
    #diags_root = Path('/nobackupp2/ifenty/ECCO_V4r4_Ou/V4r4')
    
    # Define tail for dataset description (summary)
    dataset_description_tail_native = ' on the native Lat-Lon-Cap 90 (LLC90) model grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'
    dataset_description_tail_latlon = ' interpolated to a regular 0.5-degree grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'

    filename_tail_latlon = '_ECCO_V4r4_latlon_0p50deg.nc'
    filename_tail_native = '_ECCO_V4r4_native_llc0090.nc'


    ## GRID DIR -- MAKE SURE USE GRID FIELDS WITHOUT BLANK TILES!!
    #mds_grid_dir = Path('/Users/ifenty/tmp/no_blank_all')

    ## METADATA
    metadata_json_dir = Path('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')

    metadata_fields = ['ECCOv4r4_global_metadata_for_all_datasets', 
                       'ECCOv4r4_global_metadata_for_latlon_datasets',
                       'ECCOv4r4_global_metadata_for_native_datasets',
                       'ECCOv4r4_coordinate_metadata_for_all_datasets',
                       'ECCOv4r4_coordinate_metadata_for_latlon_datasets',
                       'ECCOv4r4_coordinate_metadata_for_native_datasets',
                       'ECCOv4r4_groupings_for_1D_datasets',
                       'ECCOv4r4_groupings_for_latlon_datasets',
                       'ECCOv4r4_groupings_for_native_datasets',
                       'ECCOv4r4_variable_metadata',
                       'ECCOv4r4_variable_metadata_for_latlon_datasets']

                        ## PODAAC fields
    podaac_dir = Path('/home/ifenty/git_repos_others/ecco-data-pub/metadata')


    ## ECCO GRID 
    ecco_grid_dir = Path('/nobackupp2/ifenty/podaac/')
    ecco_grid_dir = Path('/home/ifenty/ian1/ifenty/ECCOv4/Version4/Release4/nctiles_grid')

    ecco_grid_dir_mds = Path('/home/ifenty/data/grids/grid_ECCOV4r4')
    max_k = 50


    #mon_ts_to_load = [732, 1428]#,2172]   
    #day_ts_to_load = [12, 36]#, 60]

    divide_OBP_by_g = True


    #%% -- -program start


    # Define fill values for binary and netcdf
    # ---------------------------------------------
    if array_precision == np.float32:
        binary_output_dtype = '>f4'
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif array_precision == np.float64:
        binary_output_dtype = '>f8'
        netcdf_fill_value = nc4.default_fillvals['f8']
        
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

    coordinate_metadata_for_all_datasets = metadata['ECCOv4r4_coordinate_metadata_for_all_datasets']
    coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
    coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']

    groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
    groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
    groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

    variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']
    variable_metadata = metadata['ECCOv4r4_variable_metadata']

    #%%
    # load PODAAC fields        
    podaac_dataset_table = read_csv(podaac_dir / 'datasets.csv')

    # load ECCO grid
    ecco_grid = ecco.load_ecco_grid_nc(ecco_grid_dir, 'ECCO-GRID.nc')
            
     
    # land masks
    ecco_land_mask_c_nan  = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask_c_nan.values = np.where(ecco_land_mask_c_nan==True,1,np.nan)
    ecco_land_mask_w_nan  = ecco_grid.maskW.copy(deep=True)
    ecco_land_mask_w_nan.values = np.where(ecco_land_mask_w_nan==True,1,np.nan)
    ecco_land_mask_s_nan  = ecco_grid.maskS.copy(deep=True)
    ecco_land_mask_s_nan.values = np.where(ecco_land_mask_s_nan==True,1,np.nan)

    #%%
    print('product type', product_type)

    if product_type == 'native':
        dataset_description_tail = dataset_description_tail_native
        filename_tail = filename_tail_native
        groupings = groupings_for_native_datasets
        output_dir_type = output_dir_base / 'native'

    
    elif product_type == 'latlon':
        dataset_description_tail = dataset_description_tail_latlon
        filename_tail = filename_tail_latlon
        groupings = groupings_for_latlon_datasets
        output_dir_type = output_dir_base / 'lat-lon'

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
        print('\nGrid Mappings')
        grid_mapping_fname = mapping_factors_dir / "ecco_latlon_grid_mappings.p"
        
        if debug_mode:
            print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
            grid_mappings_all = []
            grid_mappings_k = []
        else:

            if 'grid_mappings_k' not in globals():
                    
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
            else:
                print('... grid mappings k already in memory')
            
            
        # make a land mask in lat-lon using hfacC
        print('\nLand Mask')
        if debug_mode:
            print('...DEBUG MODE -- SKIPPING LAND MASK')
            land_mask_ll = []

        else:
            if 'land_mask_ll' not in globals():
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
    #%%   
    
       
    ## end product_type = 'latlon'    
        
    # Make depth bounds
    depth_bounds = np.zeros((nk,2))
    tmp = np.cumsum(ecco_grid.drF.values)
    
    for k in range(nk):
        if k == 0:
            depth_bounds[k,0] = 0.0
        else:
            depth_bounds[k,0] = -tmp[k-1]
        depth_bounds[k,1] = -tmp[k]
        
    #%%
    print('\nGetting directories for group variables')
    if output_freq_code == 'AVG_DAY':
        mds_diags_root_dir = diags_root / 'diags_daily'
        period_suffix = 'day_mean'
        dataset_description_head = 'This dataset contains daily-averaged '
        
    elif output_freq_code == 'AVG_MON':
        mds_diags_root_dir = diags_root / 'diags_monthly'
        period_suffix = 'mon_mean'
        dataset_description_head = 'This dataset contains monthly-averaged '
    
    elif output_freq_code == 'SNAPSHOT':
        mds_diags_root_dir = diags_root / 'diags_inst'
        period_suffix = 'inst'
        dataset_description_head = 'This dataset contains instantaneous '


    print('...output_freq_code ', output_freq_code)

    output_dir_freq = output_dir_type / period_suffix
    print('...making output_dir freq ', output_dir_freq)

    # make output directory
    if not output_dir_freq.exists():
        try:
            output_dir_freq.mkdir()
            print('... made %s ' % output_dir_freq)
        except:
            print('...cannot make %s ' % output_dir_freq)
      
    # load files    
    field_paths = np.sort(list(mds_diags_root_dir.glob('*' + period_suffix + '*')))
            
    ## load variable file and directory names
    print ('...number of subdirectories found ', len(field_paths)) 
    all_field_names = []
    
    # extract record name out of full directory
    for f in field_paths:
        all_field_names.append(f.name)
    
    print (all_field_names)
    

    # determine which grouping to process
    print('\nDetermining grouping to process')
    grouping = []
    if grouping_to_process == 'by_job':
        print('... using grouping provided by job_id')
        grouping_num = find_grouping_to_process_by_job_id(job_id, len(groupings))
        
    else:
        print('... using provided grouping ', grouping_to_process)
        grouping_num = grouping_to_process

    grouping = groupings[grouping_num]
    print('... grouping to use ', grouping['name'])
    print('... fields in grouping ', grouping['fields'])
    
    # dimension of dataset
    grouping_dim = grouping['dimension']
    print('... grouping dimension', grouping_dim)
    
    # find variables in dataset
    tmp = grouping['fields'].split(',')
    vars_to_load  = []
    for var in tmp:
        vars_to_load.append(var.strip())
    
        
    # find directories with variables
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
    
    print('\nDirectories with the variables in the grouping')
    for var in vars_to_load:
        print('... ', var, var_directories[var])
        
    # define empty list of gcmd keywords pertaining to this dataset
    grouping_gcmd_keywords = []

    print('\nDetermining time steps to load')
    # determine which time steps to process
    if time_steps_to_process == 'all':
        print('...using all time steps')
        time_steps_to_process = find_all_time_steps(vars_to_load, var_directories)

    elif time_steps_to_process == 'by_job':
        print('...finding time steps to process by job id', job_id)
        all_time_steps = find_all_time_steps(vars_to_load, var_directories)
        
        if grouping_to_process == 'by_job':
            print('... divinding up time by num_jobs and num_groupings')
            time_steps_to_process = \
                find_time_steps_to_process_by_job_id(num_jobs, job_id, len(groupings), all_time_steps)
        else:
            print('...dividing up time into num_jobs using 1 grouping', num_jobs)
            time_steps_to_process = \
                find_time_steps_to_process_by_job_id(num_jobs, job_id, 1, all_time_steps)
    else:
        print('...using provided time steps to process list ', time_steps_to_process)


    # create dataset description head
    dataset_description =\
        dataset_description_head +\
            grouping['name'] +\
            dataset_description_tail
            
    #%%
    # PROCESS EACH TIME LEVEL           
    #   loop through time
    print('\nLooping through time levels')
    for cur_ts_i, cur_ts in enumerate(time_steps_to_process):
        
        #%%
        time_delta = np.timedelta64(cur_ts, 'h')
        cur_ts = int(cur_ts)        
        cur_time = ecco_start_time + time_delta
        times = [pd.to_datetime(str(cur_time))]
        
        if 'AVG' in output_freq_code:
            tb, ct = ecco.make_time_bounds_from_ds64(np.datetime64(times[0]),output_freq_code)
            record_start_time = tb[0]
            record_end_time = tb[1]
        else:
            record_start_time = np.datetime64(times)
                  
        print('cur_ts, i, tb ', str(cur_ts).zfill(10), str(cur_ts_i).zfill(4), tb)
                    

        gd = Path('/home/ifenty/data/grids/grid_llc90/no_blank_all')
        # loop through variables to load
        F_DS_vars = []

        if not debug_mode:
            for var in vars_to_load:

                mds_var_dir = var_directories[var]
                print (var, mds_var_dir)
                        
                mds_file = list(mds_var_dir.glob(var + '*' + str(cur_ts).zfill(10) + '*.data'))
                
                if len(mds_file) != 1:
                    print('invalid # of mds files')
                    print(mds_file)
                    sys.exit()
                else:
                    mds_file = mds_file[0]          
                      
                print(mds_var_dir)
                print(mds_file.name)
                      
                if product_type == 'latlon':
                    if grouping_dim == '2D':
                        
                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file.name,\
                                               llc=90, skip=0,\
                                               nk=1, nl=1,
                                               filetype = '>f',
                                               less_output = True, 
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
                    
                    elif grouping_dim == '3D':
                        
                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file, \
                                                   llc=90, skip=0, nk=nk,\
                                                   nl=1, 
                          	      filetype = '>f', less_output = True, 
                                  use_xmitgcm=False)
                  
                        F_ll = np.zeros((nk,360,720))
                        
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
                            
                elif product_type == 'native':
                    print('native')
                    print('--- cur_ts', cur_ts)
                    print(type(cur_ts))
                    short_mds_name = mds_file.name.split('.')[0]
                    F_DS = \
                        ecco.load_ecco_vars_from_mds(mds_var_dir,\
                                                     gd, \
                                                     mds_files=short_mds_name,\
                                                     output_freq_code=output_freq_code,\
                                                     model_time_steps_to_load=cur_ts)     
                    #print(F_DS)
                    var_name_from_F_DS = list(F_DS.keys())[0]
                                                    
                    F_DA = F_DS[var_name_from_F_DS]
                    
                    if var_name_from_F_DS != var:
                        print(var,var_name_from_F_DS )
                        sys.exit()
                        
                    #F_DA.name = var
                    # apply land mask 
                    if grouping_dim == '2D':
                        F_DA = F_DA * ecco_land_mask_c_nan[0,:]
                    elif grouping_dim == '3D':
                        F_DA = F_DA * ecco_land_mask_c_nan
                            
                    F_DA = F_DA.compute()
                    
                    WRONG MASK -- THIS IS ONLY FOR C-GRID POINTS
                    
                            


                # ADD TIME STEP COORDINATE
                print('\n... assigning time step', np.int32(cur_ts))
                F_DA=F_DA.assign_coords({"time_step": (("time"), [np.int32(cur_ts)])})
    
                print('... assigning name', var)
                # assign name to data array
                F_DA.name = var
    
                # Possibly rename to something else if indicated
                if 'variable_rename' in grouping.keys():
                    rename_pairs = grouping['variable_rename'].split(',')
    
                    for rename_pair in rename_pairs:
                        orig_var_name, new_var_name = rename_pair.split(':')
                        
                        if var == orig_var_name:
                            F_DA.name = new_var_name
    
                # cast to appropriate precision
                F_DA = F_DA.astype(array_precision)
                
                
                if divide_OBP_by_g:
                    if F_DA.name == 'OBP' or F_DA.name == 'OBPGMAP':
                        print('DIVIDING BY g! ', F_DA.name)
                        F_DA.values = F_DA.values / 9.81000
                
    
                F_DA_min = np.nanmin(F_DA.values)
                F_DA_max = np.nanmax(F_DA.values)
                
                # replace nan with fill value
                F_DA.values = np.where(np.isnan(F_DA.values), \
                                       netcdf_fill_value, F_DA.values)
            
    
                # valid min max
                F_DA.attrs['valid_min'] = F_DA_min
                F_DA.attrs['valid_max'] = F_DA_max
                
                # convert to dataset
                F_DS = F_DA.to_dataset()
                
                if product_type == 'latlon':
                    ## ADD BOUNDS TO COORDINATES
                    #   assign lat and lon bounds
                    F_DS=F_DS.assign_coords({"latitude_bnds": (("latitude","nv"), lat_bounds)})
                    F_DS=F_DS.assign_coords({"longitude_bnds": (("longitude","nv"), lon_bounds)})
                               
                #   if 3D assign depth bounds
                if grouping_dim == '3D':
                    F_DS = F_DS.assign_coords({"Z_bnds": (("Z","nv"), depth_bounds)})
                
                #   add appropriate time bounds.
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
                        
            
            #####-- > END LOOPING THROUGH VARIABLES
            
            
            ## merge the data arrays to make one DATASET
            G = xr.merge((F_DS_vars))
            
            # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
            print('\n... adding metadata specific to the variable')
            G, grouping_gcmd_keywords = \
                add_variable_metadata(variable_metadata, G, grouping_gcmd_keywords)
            
            if product_type == 'latlon':
                print('\n... using latlon dataseta metadata specific to the variable')
                G, grouping_gcmd_keywords = \
                    add_variable_metadata(variable_metadata_latlon, G, grouping_gcmd_keywords)

                
            # ADD COORDINATE METADATA 
            print('\n... adding coordinate metadata common to all datasets')
            G = add_coordinate_metadata(coordinate_metadata_for_all_datasets,G)
            
            if product_type == 'latlon':
                print('... adding latlon dataset specific coordinate metadata')
                G = add_coordinate_metadata(coordinate_metadata_for_latlon_datasets,G)
            elif product_type == 'native':
                print('... adding native dataset specific coordinate metadata')                        
                G = add_coordinate_metadata(coordinate_metadata_for_native_datasets,G)
                
            # ADD GLOBAL METADATA
            print("\n... adding global metadata for all datasets")
            G = add_global_metadata(global_metadata_for_all_datasets, G,\
                                    grouping_dim)
            
            if product_type == 'latlon':
                print('... adding global meta for latlon products')
                G = add_global_metadata(global_metadata_for_latlon_datasets, G,\
                                        grouping_dim)
            elif product_type == 'native':
                print('... adding global metadata for native grid products')
                G = add_global_metadata(global_metadata_for_native_datasets, G,\
                                        grouping_dim)

            # ADD GLOBAL METADATA ASSOCIATED WITH TIME AND DATE
            print('\n... adding time / data global attrs')
            if 'AVG' in output_freq_code:
                G.attrs['time_coverage_start'] = str(G.time_bnds.values[0][0])[0:19]
                G.attrs['time_coverage_end'] = str(G.time_bnds.values[0][1])[0:19]

            else:
                G.attrs['time_coverage_start'] = str(G.time.values[0])[0:19]
                G.attrs['time_coverage_end'] = str(G.time.values[0])[0:19]

            # current time and date 
            current_time = datetime.datetime.now().isoformat()[0:19]
            G.attrs['date_created'] = current_time
            G.attrs['date_modified'] = current_time
            G.attrs['date_metadata_modified'] = current_time
            G.attrs['date_issued'] = current_time
               
            # add coordinate attributes to the variables
            coord_attrs, coord_G= get_coordinate_attribute_to_data_vars(G)
            for coord in coord_attrs.keys():
                # REMOVE TIME STEP FROM LIST OF COORDINATES (PODAAC REQUEST)
                coord_attrs[coord] = coord_attrs[coord].split('time_step')[0].strip()
                
            print('\n... creating variable encodings')
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
            dv_encoding = dict()
            for dv in G.data_vars:
                dv_encoding[dv] =  {'zlib':True, \
                                    'complevel':5,\
                                    'shuffle':True,\
                                    '_FillValue':netcdf_fill_value}
                
                # overwrite default coordinats attribute (PODAAC REQUEST)
                G[dv].encoding['coordinates'] = coord_attrs[dv]
                 
                
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
            print('\n... creating coordinate encodings')
            coord_encoding = dict()
            
            for coord in G.coords:
                # default encoding: no fill value, float32
                coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                
                if coord == 'time' or coord == 'time_bnds':
                    coord_encoding[coord]['dtype'] ='int32'
                    if 'units' in G[coord].attrs:
                        # apply units as encoding for time
                        coord_encoding[coord]['units'] = G[coord].attrs['units']
                        # delete from the attributes list
                        del G[coord].attrs['units']
                        
                elif coord == 'time_step':
                    coord_encoding[coord]['dtype'] ='int32'
               
                    
            # MERGE ENCODINGS for coordinates and variables
            encoding = {**dv_encoding, **coord_encoding} 
         
            
            # MERGE GCMD KEYWORDS
            print('\n... merging GCMD keywords')
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
            
            
            ## ADD FINISHING TOUCHES 

            # uuic
            print('\n... adding uuid')
            G.attrs['uuid'] = str(uuid.uuid1())
            
            # add any dataset grouping specific comments.
            if 'comment' in grouping:
                G.attrs['comment'] = G.attrs['comment'] + ' ' \
                + grouping['comment']

            # set the long name of the time attribute
            if 'AVG' in output_freq_code:
                G.time.attrs['long_name'] = 'center time of averaging period'
            else:
                G.time.attrs['long_name'] = 'snapshot time'
                
            # set averaging period duration and resolution
            print('\n... setting time coverage resolution')
            # --- AVG DAY
            if output_freq_code == 'AVG_MON':
                G.attrs['time_coverage_duration'] = 'P1M'
                G.attrs['time_coverage_resolution'] = 'P1M'
        
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

            ## construct filename
            print('\n... creating filename')

            filename = grouping['filename'] + '_' + ppp_tttt + '_' + \
                date_str + filename_tail
            
            # make subdirectory for the grouping
            output_dir = output_dir_freq / grouping['filename'] 
            print('\n... creating output_dir', output_dir)

            if not output_dir.exists():
                try:
                    output_dir.mkdir()
                except:
                    print ('cannot make %s ' % output_dir)
                    
                
            netcdf_output_filename = output_dir / filename
       
            # add product name attribute = filename                     
            G.attrs['product_name'] = filename
            
            # add summary attribute = description of dataset 
            G.attrs['summary'] = dataset_description + ' ' + G.attrs['summary']

            # get podaac metadata based on filename       
            print('\n... getting PODAAC metadata')
            podaac_metadata = \
                find_podaac_metadata(podaac_dataset_table, filename)

            # apply podaac metadata based on filename
            print('\n... applying PODAAC metadata')
            pprint(podaac_metadata)
            G = apply_podaac_metadata(G, podaac_metadata)

            # sort comments alphabetically
            print('\n... sorting attributes')
            sort_all_attrs(G)
            
            # add one final comment (PODAAC request)
            G.attrs["coordinates_comment"] = \
                "Note: the global 'coordinates' attribute descibes auxillary coordinates."

            ## SAVE 

            print('\n... saving to netcdf ', netcdf_output_filename)
            G.to_netcdf(netcdf_output_filename, encoding=encoding)
            G.close()


#%%
if __name__ == "__main__":

    
    #%%
    num_jobs = 3
    job_id = 0
    grouping_to_process = 0
    #grouping_to_process='by_job'
    time_steps_to_process = 'by_job'


    print (sys.argv)
    if len(sys.argv) > 1:
        num_jobs = int(sys.argv[1])
    if len(sys.argv) > 2:
        job_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        grouping_to_process = int(sys.argv[3])

    product_type = 'native'
    output_freq_code = 'AVG_MON'
   
    debug_mode=False
    
    print('\n\n===================================')
    print('starting python: num jobs, job_id', num_jobs, job_id)
    print('grouping to process', grouping_to_process)
    print('time_steps_to_process', time_steps_to_process)
    
    
    # def generate_netcdfs(output_freq_code, job_id:int, num_jobs:int, \
    #                  product_type,\
    #                  grouping_to_process='by_job',\
    #                  time_steps_to_process='by_job', \
    #                  debug_mode=False):
       
    #%%
    for grouping_to_process in range(3,13):
        generate_netcdfs(output_freq_code, job_id, num_jobs,\
                         product_type, \
                         grouping_to_process,\
                         time_steps_to_process, \
                         debug_mode)
            
    import time
    #time.sleep(10)
