"""
Created on Thur May 12 15:41:00 2022

Author: Duncan Bark
Adapted from ifenty's "eccov4r4_gen_for_podaac.py"

"""

import sys
import json
import uuid
import yaml
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import pyresample as pr
from pathlib import Path
from pprint import pprint
from pandas import read_csv
from importlib import reload
from collections import OrderedDict

path_to_ecco_group = Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(f'{path_to_ecco_group}/ECCO-ACCESS/ecco-cloud-utils')
sys.path.append(f'{path_to_ecco_group}/ECCOv4-py')
import ecco_v4_py as ecco
import ecco_cloud_utils as ea

# -------------------------------------------------------------------------------------------------


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


def find_podaac_metadata(podaac_dataset_table, filename, debug=False):
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

    if debug:
        print('split filename into ', head, tail)

    tail = tail.split("_ECCO_V4r4_")[1]

    if debug:
        print('further split tail into ',  tail)

    # Get the filenames column from my table as a list of strings.
    names = podaac_dataset_table['DATASET.FILENAME']

    # Find the index of the row with a filename with matching head, tail chars.
    index = names.apply(lambda x: all([x.startswith(head), x.endswith(tail)]))

    # Select that row from podaac_dataset_table table and make a copy of it.
    metadata = podaac_dataset_table[index].iloc[0].to_dict()

    podaac_metadata = {
        'id': metadata['DATASET.PERSISTENT_ID'].replace("PODAAC-","10.5067/"),
        'metadata_link': f"https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName={metadata['DATASET.SHORT_NAME']}",
        'title': metadata['DATASET.LONG_NAME'],
    }
    if debug:
        print('\n... podaac metadata:')
        pprint(podaac_metadata)

    return podaac_metadata


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


def sort_attrs(attrs):
    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    for k in keys:
        od[k] = attrs[k]

    return od


def generate_netcdfs(output_freq_code,
                     product_type,
                     mapping_factors_dir,
                     output_dir_base,
                     diags_root,
                     metadata_json_dir,
                     podaac_dir,
                     ecco_grid_dir,
                     ecco_grid_dir_mds,
                     ecco_grid_filename,
                     grouping_to_process,
                     time_steps_to_process,
                     array_precision = np.float32,
                     debug_mode=False):


    
    # output_freq_codes  one of either 'AVG_DAY' or 'AVG_MON'
    #
    # time_steps_to_process : one of
    # 'all'
    # a list [1,3,4,...]

    # grouping to process:
    #   a single number


    # diags_root

    # ECCO FIELD INPUT DIRECTORY
    # model diagnostic output
    # subdirectories must be
    #  'diags_all/diag_mon'
    #  'diags_all/diag_mon'
    #  'diags_all/snap'

    print('\nBEGIN: generate_netcdfs')
    print('OFC', output_freq_code)
    print('PDT', product_type)
    print('GTP', grouping_to_process)
    print('TSP', time_steps_to_process)
    print('DBG', debug_mode)
    print('\n')



    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999

    # num of depth levels
    nk = 50

    # levels to process (for testing purposes make less than nk)
    max_k = 50

    ecco_start_time = np.datetime64('1992-01-01T12:00:00')
    ecco_end_time   = np.datetime64('2017-12-31T12:00:00')

    # Define tail for dataset description (summary)
    dataset_description_tail_native = ' on the native Lat-Lon-Cap 90 (LLC90) model grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'
    dataset_description_tail_latlon = ' interpolated to a regular 0.5-degree grid from the ECCO Version 4 revision 4 (V4r4) ocean and sea-ice state estimate.'

    filename_tail_latlon = '_ECCO_V4r4_latlon_0p50deg.nc'
    filename_tail_native = '_ECCO_V4r4_native_llc0090.nc'

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

    coordinate_metadata_for_1D_datasets = metadata['ECCOv4r4_coordinate_metadata_for_1D_datasets']
    coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
    coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']

    geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
    geometry_metadata_for_native_datasets = metadata['ECCOv4r4_geometry_metadata_for_native_datasets']

    groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
    groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
    groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

    variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']
    variable_metadata_default = metadata['ECCOv4r4_variable_metadata']

    variable_metadata_native = variable_metadata_default + geometry_metadata_for_native_datasets

    
    # load PODAAC fields
    podaac_dataset_table = read_csv(podaac_dir / 'datasets.csv')

    # load ECCO grid
    #ecco_grid = ecco.load_ecco_grid_nc(ecco_grid_dir, ecco_grid_filename)
    ecco_grid = xr.open_dataset(ecco_grid_dir / ecco_grid_filename)

    print(ecco_grid)

    # land masks
    ecco_land_mask_c_nan  = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask_c_nan.values = np.where(ecco_land_mask_c_nan==True, 1, np.nan)
    ecco_land_mask_w_nan  = ecco_grid.maskW.copy(deep=True)
    ecco_land_mask_w_nan.values = np.where(ecco_land_mask_w_nan==True, 1, np.nan)
    ecco_land_mask_s_nan  = ecco_grid.maskS.copy(deep=True)
    ecco_land_mask_s_nan.values = np.where(ecco_land_mask_s_nan==True, 1, np.nan)

    
    print('\nproduct type', product_type)

    if product_type == 'native':
        dataset_description_tail = dataset_description_tail_native
        filename_tail = filename_tail_native
        groupings = groupings_for_native_datasets
        output_dir_type = output_dir_base / 'native'

        global_metadata = global_metadata_for_all_datasets + global_metadata_for_native_datasets

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
            wet_pts_k[k] = np.where(ecco_grid.hFacC[k,:] > 0)
            xc_wet_k[k] = ecco_grid.XC.values[wet_pts_k[k]]
            yc_wet_k[k] = ecco_grid.YC.values[wet_pts_k[k]]

            source_grid_k[k] = pr.geometry.SwathDefinition(lons=xc_wet_k[k], lats=yc_wet_k[k])


        # The pyresample 'grid' information for the 'source' (ECCO grid) defined using
        # all XC and YC points, even land.  Used to create the land mask
        source_grid_all =  pr.geometry.SwathDefinition(lons=ecco_grid.XC.values.ravel(),
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
        target_grid_lons, target_grid_lats = ea.generalized_grid_product(product_name,
                                                                         data_res,
                                                                         data_max_lat,
                                                                         area_extent,
                                                                         dims,
                                                                         proj_info)


        # pull out just the lats and lons (1D arrays)
        target_grid_lons_1D = target_grid_lons[0,:]
        target_grid_lats_1D = target_grid_lats[:,0]

        # calculate the areas of the lat-lon grid
        ea_area = ea.area_of_latlon_grid(-180, 180, -90, 90, data_res, data_res, less_output=True)
        lat_lon_grid_area=ea_area['area']
        target_grid_shape = lat_lon_grid_area.shape


        # calculate effective radius of each target grid cell.  required for the bin
        # averaging
        target_grid_radius = np.sqrt(lat_lon_grid_area / np.pi).ravel()


        # CALCULATE GRID-TO-GRID MAPPING FACTORS
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

                    [grid_mappings_all, grid_mappings_k] = pickle.load(open(grid_mapping_fname, 'rb'))

                else:
                    # if not, make new grid mapping factors
                    print('... no mapping factors found, recalculating')

    			    # find the mapping between all points of the ECCO grid and the target grid.
                    grid_mappings_all = \
                        ea.find_mappings_from_source_to_target(source_grid_all,
                                                               target_grid,
                                                               target_grid_radius,
                                                               source_grid_min_L,
                                                               source_grid_max_L)

                    # then find the mapping factors between all wet points of the ECCO grid
                    # at each vertical level and the target grid
                    grid_mappings_k = dict()

                    for k in range(nk):
                        print(k)
                        grid_mappings_k[k] = \
                            ea.find_mappings_from_source_to_target(source_grid_k[k],
                                                                   target_grid,
                                                                   target_grid_radius,
                                                                   source_grid_min_L,
                                                                   source_grid_max_L)
                    if not mapping_factors_dir.exists():
                        try:
                            mapping_factors_dir.mkdir()
                        except:
                            print ('cannot make %s ' % mapping_factors_dir)

                    try:
                        pickle.dump([grid_mappings_all, grid_mappings_k], open(grid_mapping_fname, 'wb'))
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
                    num_source_indices_within_target_radius_i, \
                    nearest_source_index_to_target_index_i = grid_mappings_all

                    for k in range(nk):
                        print(k)

                        source_field = ecco_land_mask_c_nan.values[k,:].ravel()

                        land_mask_ll[k,:] =  \
                            ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                        num_source_indices_within_target_radius_i,
                                                        nearest_source_index_to_target_index_i,
                                                        source_field, target_grid_shape,
                                                        operation='nearest', 
                                                        allow_nearest_neighbor=True)

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

        lon_bounds = np.zeros((dims[0],2))
        for i in range(dims[0]):
            lon_bounds[i,0] = target_grid_lons[0,i] - data_res/2
            lon_bounds[i,1] = target_grid_lons[0,i] + data_res/2

    # END IF NATIVE VS. LATLON

    # show groupings
    print('\nAll groupings')
    for gi, gg in enumerate(groupings_for_native_datasets):
        print('\t', gi, gg['name'])

    # Make depth bounds
    depth_bounds = np.zeros((nk,2))
    tmp = np.cumsum(ecco_grid.drF.values)

    for k in range(nk):
        if k == 0:
            depth_bounds[k,0] = 0.0
        else:
            depth_bounds[k,0] = -tmp[k-1]
        depth_bounds[k,1] = -tmp[k]

    
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
        period_suffix = 'day_inst'
        dataset_description_head = 'This dataset contains instantaneous '
    else:
        print('valid options are AVG_DAY, AVG_MON, SNAPSHOT')
        print('you provided ', output_freq_code)
        sys.exit()

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
    print('... using provided grouping ', grouping_to_process)
    grouping_num = grouping_to_process

    grouping = groupings[grouping_num]
    print('... grouping to use ', grouping['name'])
    print('... fields in grouping ', grouping['fields'])

    # dimension of dataset
    dataset_dim = grouping['dimension']
    print('... grouping dimension', dataset_dim)

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
    else:
        print('...using provided time steps to process list ', time_steps_to_process)


    # create dataset description head
    dataset_description = dataset_description_head + grouping['name'] + dataset_description_tail

    
    # PROCESS EACH TIME LEVEL
    #   loop through time
    print('\nLooping through time levels')
    for cur_ts_i, cur_ts in enumerate(time_steps_to_process):

        print('\n\n=== TIME LEVEL ===', str(cur_ts_i).zfill(5), str(cur_ts).zfill(10))
        print('\n')
        time_delta = np.timedelta64(cur_ts, 'h')
        cur_ts = int(cur_ts)
        cur_time = ecco_start_time + time_delta
        times = [pd.to_datetime(str(cur_time))]

        if 'AVG' in output_freq_code:
            tb, record_center_time = ecco.make_time_bounds_from_ds64(np.datetime64(times[0]), output_freq_code)
            #print('tb', type(tb))
            #print(tb)
            print('ORIG  tb, ct ', tb, record_center_time)

            # fix beginning of last record
            if tb[1].astype('datetime64[D]') == ecco_end_time.astype('datetime64[D]'):
                print('end time match ')
                time_delta = np.timedelta64(12,'h')
                rec_avg_start = tb[0] + time_delta
                rec_avg_end   = tb[1]
                rec_avg_delta = rec_avg_end - rec_avg_start
                rec_avg_middle = rec_avg_start + rec_avg_delta/2
                #print(rec_avg_start, rec_avg_middle, rec_avg_end)

                tb[0] = rec_avg_start
                record_center_time = rec_avg_middle
                #print('NEW  cur_ts, i, tb, ct ', str(cur_ts).zfill(10), str(cur_ts_i).zfill(4), tb, record_center_time)

            # truncate to ecco_start_time
            if tb[0].astype('datetime64[D]') == ecco_start_time.astype('datetime64[D]'):
                print('start time match ')
                rec_avg_start = ecco_start_time
                rec_avg_end   = tb[1]
                rec_avg_delta = tb[1] - ecco_start_time
                rec_avg_middle = rec_avg_start + rec_avg_delta/2
                #print(rec_avg_start, rec_avg_middle, rec_avg_end)

                tb[0] = ecco_start_time
                record_center_time = rec_avg_middle
                #print('NEW  cur_ts, i, tb, ct ', str(cur_ts).zfill(10), str(cur_ts_i).zfill(4), tb, record_center_time)

            record_start_time = tb[0]
            record_end_time = tb[1]
            print('FINAL tb, ct ', tb, record_center_time)

        else:
            #snapshot, all times are the same
            print(times)
            print(type(times[0]))

            record_start_time = np.datetime64(times[0])
            record_end_time = np.datetime64(times[0])
            record_center_time = np.datetime64(times[0])


        # loop through variables to load
        F_DS_vars = []

        if not debug_mode:
            for var in vars_to_load:

                mds_var_dir = var_directories[var]

                print ('\nProcessing ', var, mds_var_dir)

                mds_file = list(mds_var_dir.glob(var + '*' + str(cur_ts).zfill(10) + '*.data'))

                if len(mds_file) != 1:
                    print('invalid # of mds files')
                    print(mds_file)
                    sys.exit()
                else:
                    mds_file = mds_file[0]

                if product_type == 'latlon':
                    if dataset_dim == '2D':

                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file.name,
                                                    llc=90, skip=0,
                                                    nk=1, nl=1,
                                                    filetype='>f',
                                                    less_output=True,
                                                    use_xmitgcm=False)

                        F_wet_native = F[wet_pts_k[0]]

                        # get mapping factors for the the surface level
                        source_indices_within_target_radius_i, \
                        num_source_indices_within_target_radius_i, \
                        nearest_source_index_to_target_index_i = grid_mappings_k[0]

                        # transform to new grid
                        F_ll =  \
                            ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                        num_source_indices_within_target_radius_i,
                                                        nearest_source_index_to_target_index_i,
                                                        F_wet_native,
                                                        target_grid_shape,
                                                        operation='mean',
                                                        allow_nearest_neighbor=True)

                        F_ll_masked = np.expand_dims(F_ll * land_mask_ll[0,:],0)

                        F_DA = xr.DataArray(F_ll_masked,
                                            coords=[[record_end_time],
                                                    target_grid_lats_1D,
                                                    target_grid_lons_1D],
                                            dims=["time", "latitude","longitude"])

                    elif dataset_dim == '3D':

                        F = ecco.read_llc_to_tiles(mds_var_dir, mds_file,
                                                   llc=90, skip=0, nk=nk,
                                                   nl=1,
                                                   filetype='>f',
                                                   less_output=True,
                                                   use_xmitgcm=False)

                        F_ll = np.zeros((nk,360,720))

                        for k in range(max_k):
                            #print(var,k)
                            F_k = F[k]
                            F_wet_native = F_k[wet_pts_k[k]]

                            source_indices_within_target_radius_i, \
                            num_source_indices_within_target_radius_i,\
                            nearest_source_index_to_target_index_i = grid_mappings_k[k]

                            F_ll[k,:] =  \
                                ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                            num_source_indices_within_target_radius_i,
                                                            nearest_source_index_to_target_index_i,
                                                            F_wet_native, target_grid_shape,
                                                            operation='mean', allow_nearest_neighbor=True)


                        # multiple by land mask
                        F_ll_masked = np.expand_dims(F_ll * land_mask_ll, 0)

                        Z = ecco_grid.Z.values

                        F_DA = xr.DataArray(F_ll_masked,
                                            coords=[[record_end_time],
                                                    Z,
                                                    target_grid_lats_1D,
                                                    target_grid_lons_1D],
                                            dims=["time", "Z", "latitude","longitude"])

                    # --- end 2D or 3D

                    # assign name to data array
                    print('... assigning name', var)
                    F_DA.name = var

                    F_DS = F_DA.to_dataset()

                    #   add time bounds object
                    if 'AVG' in output_freq_code:
                        tb_ds, ct_ds = \
                            ecco.make_time_bounds_and_center_times_from_ecco_dataset(F_DS,
                                                                                     output_freq_code)
                        F_DS = xr.merge((F_DS, tb_ds))
                        F_DS = F_DS.set_coords('time_bnds')


                elif product_type == 'native':
                    short_mds_name = mds_file.name.split('.')[0]

                    F_DS = ecco.load_ecco_vars_from_mds(mds_var_dir,
                                                        mds_grid_dir = ecco_grid_dir_mds,
                                                        mds_files = short_mds_name,
                                                        vars_to_load = var,
                                                        drop_unused_coords = True,
                                                        grid_vars_to_coords = False,
                                                        output_freq_code=output_freq_code,
                                                        model_time_steps_to_load=cur_ts,
                                                        less_output = True)

                    vars_to_drop = set(F_DS.data_vars).difference(set([var]))
                    F_DS.drop_vars(vars_to_drop)

                    # determine all of the dimensions used by data variables
                    all_var_dims = set([])
                    for ecco_var in F_DS.data_vars:
                        all_var_dims = set.union(all_var_dims, set(F_DS[ecco_var].dims))

                    # mask the data variables
                    for data_var in F_DS.data_vars:
                        data_var_dims = set(F_DS[data_var].dims)
                        if len(set.intersection(data_var_dims, set(['k','k_l','k_u','k_p1']))) > 0:
                            data_var_3D = True
                        else:
                            data_var_3D = False

                        # 'i, j = 'c' point
                        if len(set.intersection(data_var_dims, set(['i','j']))) == 2 :
                            if data_var_3D:
                                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_c_nan.values
                                print('... masking with 3D maskC ', data_var)
                            else:
                                print('... masking with 2D maskC ', data_var)
                                F_DS[data_var].values= F_DS[data_var].values * ecco_land_mask_c_nan[0,:].values

                        # i_g, j = 'u' point
                        elif len(set.intersection(data_var_dims, set(['i_g','j']))) == 2 :
                            if data_var_3D:
                                print('... masking with 3D maskW ', data_var)
                                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_w_nan.values
                            else:
                                print('... masking with 2D maskW ', data_var)
                                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_w_nan[0,:].values

                        # i, j_g = 's' point
                        elif len(set.intersection(data_var_dims, set(['i','j_g']))) == 2 :
                            if data_var_3D:
                                print('... masking with 3D maskS ', data_var)
                                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_s_nan.values
                            else:
                                print('... masking with 2D maskS ', data_var)
                                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_s_nan[0,:].values

                        else:
                            print('I cannot determine dimension of data variable ', data_var)
                            sys.exit()


                ############## end latlon vs. native load

                #   fix time bounds.
                if 'AVG' in output_freq_code:
                    #print('\nfixing time and  time bounds')
                    #print('----before')
                    #print(F_DS.time)
                    #print(F_DS.time_bnds)

                    F_DS.time_bnds.values[0][0] = record_start_time
                    F_DS.time_bnds.values[0][1] = record_end_time
                    F_DS.time.values[0] = record_center_time

                    #print('----after')
                    #print(F_DS.time)
                    #print(F_DS.time_bnds)

                # ADD TIME STEP COORDINATE
            #    print('\n... assigning time step', np.int32(cur_ts))
            #    for data_var in F_DS.data_vars:
                #    F_DS[data_var] = F_DS[data_var].assign_coords({"time_step": (("time"), [np.int32(cur_ts)])})


                # Possibly rename variable if indicated
                if 'variable_rename' in grouping.keys():
                    rename_pairs = grouping['variable_rename'].split(',')

                    for rename_pair in rename_pairs:
                        orig_var_name, new_var_name = rename_pair.split(':')

                        if var == orig_var_name:
                            F_DS = F_DS.rename({orig_var_name:new_var_name})
                            print('renaming from ', orig_var_name, new_var_name)
                            print(F_DS.data_vars)

                # cast data variable to desired precision
                for data_var in F_DS.data_vars:
                    if F_DS[data_var].values.dtype != array_precision:
                        F_DS[data_var].values = F_DS[data_var].astype(array_precision)


                # set valid min and max, and replace nan with fill values
                for data_var in F_DS.data_vars:
                    F_DS[data_var].attrs['valid_min'] = np.nanmin(F_DS[data_var].values)
                    F_DS[data_var].attrs['valid_max'] = np.nanmax(F_DS[data_var].values)

                    # replace nan with fill value
                    F_DS[data_var].values = np.where(np.isnan(F_DS[data_var].values),
                                                     netcdf_fill_value, F_DS[data_var].values)

                # add bounds to spatial coordinates
                if product_type == 'latlon':
                    #   assign lat and lon bounds
                    F_DS=F_DS.assign_coords({"latitude_bnds": (("latitude","nv"), lat_bounds)})
                    F_DS=F_DS.assign_coords({"longitude_bnds": (("longitude","nv"), lon_bounds)})

                    #   if 3D assign depth bounds, use Z as index
                    if dataset_dim == '3D' and 'Z' in list(F_DS.coords):
                        F_DS = F_DS.assign_coords({"Z_bnds": (("Z","nv"), depth_bounds)})

                elif product_type == 'native':
                    if 'XC_bnds' in ecco_grid.coords:
                        F_DS = F_DS.assign_coords({"XC_bnds": (("tile","j","i","nb"), ecco_grid['XC_bnds'])})
                    if 'YC_bnds' in ecco_grid.coords:
                        F_DS = F_DS.assign_coords({"YC_bnds": (("tile","j","i","nb"), ecco_grid['YC_bnds'])})

                    #   if 3D assign depth bounds, use k as index
                    if dataset_dim == '3D' and 'Z' in list(F_DS.coords):
                        F_DS = F_DS.assign_coords({"Z_bnds": (("k","nv"), depth_bounds)})


                # add this dataset to F_DS_vars and repeat for next variable
                F_DS_vars.append(F_DS)

            #####-- > END LOOPING THROUGH VARIABLES


            ## merge the data arrays to make one DATASET
            print('\n... merging F_DS_vars')
            G = xr.merge((F_DS_vars))

            # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
            print('\n... adding metadata specific to the variable')
            G, grouping_gcmd_keywords = \
                ecco.add_variable_metadata(variable_metadata_native, G, grouping_gcmd_keywords, less_output=False)

            if product_type == 'latlon':
                print('\n... using latlon dataseta metadata specific to the variable')
                G, grouping_gcmd_keywords = \
                    ecco.add_variable_metadata(variable_metadata_latlon, G, grouping_gcmd_keywords, less_output=False)


            # ADD COORDINATE METADATA
            if product_type == 'latlon':
                print('\n... adding coordinate metadata for latlon dataset')
                G = ecco.add_coordinate_metadata(coordinate_metadata_for_latlon_datasets,G)

            elif product_type == 'native':
                print('\n... adding coordinate metadata for native dataset')
                G = ecco.add_coordinate_metadata(coordinate_metadata_for_native_datasets,G)

            # ADD GLOBAL METADATA
            print("\n... adding global metadata for all datasets")
            G = ecco.add_global_metadata(global_metadata_for_all_datasets, G, dataset_dim)

            if product_type == 'latlon':
                print('\n... adding global meta for latlon dataset')
                G = ecco.add_global_metadata(global_metadata_for_latlon_datasets, G, dataset_dim)
            elif product_type == 'native':
                print('\n... adding global metadata for native dataset')
                G = ecco.add_global_metadata(global_metadata_for_native_datasets, G, dataset_dim)

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
            #coord_attrs, coord_G=  get_coordinate_attribute_to_data_vars(G)
            #print(coord_G)
            dv_coordinate_attrs = dict()

            for dv in list(G.data_vars):
                dv_coords_orig = set(list(G[dv].coords))

                #print(dv, dv_coords_orig)

                # REMOVE TIME STEP FROM LIST OF COORDINATES (PODAAC REQUEST)
                #coord_attrs[coord] = coord_attrs[coord].split('time_step')[0].strip()
                #data_var_coorcoord_attrs[coord].split()
                set_intersect = dv_coords_orig.intersection(set(['XC','YC','XG','YG','Z','Zp1','Zu','Zl','time']))

                dv_coordinate_attrs[dv] = " ".join(set_intersect)

                #print(dv, dv_coordinate_attrs[dv])

            print('\n... creating variable encodings')
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
            dv_encoding = dict()
            for dv in G.data_vars:
                dv_encoding[dv] = {'zlib':True,
                                    'complevel':5,
                                    'shuffle':True,
                                    '_FillValue':netcdf_fill_value}

                # overwrite default coordinats attribute (PODAAC REQUEST)
                G[dv].encoding['coordinates'] = dv_coordinate_attrs[dv]
                #print(G[dv].encoding)
                #dv_encoding[dv]['coordinates'] = dv_coordinate_attrs[dv]

            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
            print('\n... creating coordinate encodings')
            coord_encoding = dict()

            for coord in G.coords:
                # default encoding: no fill value, float32
                coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}

                if (G[coord].values.dtype == np.int32) or (G[coord].values.dtype == np.int64):
                    coord_encoding[coord]['dtype'] ='int32'

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

            #print(gcmd_keyword_str)
            G.attrs['keywords'] = gcmd_keyword_str

            ## ADD FINISHING TOUCHES

            # uuic
            print('\n... adding uuid')
            G.attrs['uuid'] = str(uuid.uuid1())

            # add any dataset grouping specific comments.
            if 'comment' in grouping:
                G.attrs['comment'] = G.attrs['comment'] + ' ' + grouping['comment']

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

                G.time.attrs.pop('bounds')

                # convert from oroginal
                #   '1992-01-16T12:00:00.000000000'
                # to new format
                # '1992-01-16T120000'
                date_str = str(G.time.values[0])[0:19].replace(':','')
                ppp_tttt = 'snap'

            ## construct filename
            print('\n... creating filename')

            filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + filename_tail

            # make subdirectory for the grouping
            output_dir = output_dir_freq / grouping['filename']
            print('\n... creating output_dir', output_dir)

            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except:
                    print ('cannot make %s ' % output_dir)


            # create full pathname for netcdf file
            netcdf_output_filename = output_dir / filename

            # add product name attribute = filename
            G.attrs['product_name'] = filename

            # add summary attribute = description of dataset
            G.attrs['summary'] = dataset_description + ' ' + G.attrs['summary']

            # get podaac metadata based on filename
            print('\n... getting PODAAC metadata')
            podaac_metadata = find_podaac_metadata(podaac_dataset_table, filename)

            # apply podaac metadata based on filename
            print('\n... applying PODAAC metadata')
            #pprint(podaac_metadata)
            G = apply_podaac_metadata(G, podaac_metadata)

            # sort comments alphabetically
            print('\n... sorting global attributes')
            G.attrs = sort_attrs(G.attrs)

            # add one final comment (PODAAC request)
            G.attrs["coordinates_comment"] = \
                "Note: the global 'coordinates' attribute describes auxillary coordinates."

            # SAVE
            print('\n... saving to netcdf ', netcdf_output_filename)
            G.load()

            G.to_netcdf(netcdf_output_filename, encoding=encoding)
            G.close()

            print('\n... checking existence of new file: ', netcdf_output_filename.exists())
            print('\n')

    return G, ecco_grid


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--time_steps_to_process', nargs="+",\
                        help='which time steps to process')

    parser.add_argument('--grouping_to_process', type=int,\
                        help='which dataset grouping to process, there are 20 in v4r4')

    parser.add_argument('--product_type', type=str, choices=['latlon', 'native'], \
                        help='one of either "latlon" or "native" ')

    parser.add_argument('--output_freq_code', type=str, choices=['AVG_MON','AVG_DAY','SNAPSHOT'],\
                        help='one of AVG_MON, AVG_DAY, or SNAPSHOT')

    parser.add_argument('--output_dir', type=str,\
                        help='output directory')
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    dict_key_args = {key: value for key, value in args._get_kwargs()} 

    # Testing/setup paths and config -------------------------------------
    path_to_yaml = Path(__file__).parent.resolve() / 'gen_netcdf_config.yaml'
    with open(path_to_yaml, "r") as f:
        config = yaml.load(f, yaml.Loader)

    # Collect arguments.
    # Check for a value from the command line and from the config file.
    # If the command line value exists then use that value, otherwise
    # take the config value. If neither exist, exit. 
    print('\nCollecting arguments (command line/config file)')
    final_args = {}
    missing_args = False
    for (name, value) in dict_key_args.items():
        if value == None:
            if config[name] == '':
                print(f'\t"{name}" required')
                missing_args = True
                final_args[name] = None
            else:
                final_args[name] = config[name]
        else:
            final_args[name] = value

    if missing_args:
        print(f'One or more arugments not supplied, exiting. Please check and re-run.\n')
        sys.exit()

    time_steps_to_process = final_args['time_steps_to_process']
    grouping_to_process = final_args['grouping_to_process']
    product_type = final_args['product_type']
    output_freq_code = final_args['output_freq_code']
    output_dir_base = Path(final_args['output_dir'])

    print(f'time_steps_to_process: {time_steps_to_process} ({type(time_steps_to_process)})')
    print(f'grouping_to_process: {grouping_to_process} ({type(grouping_to_process)})')
    print(f'product_type: {product_type} ({type(product_type)})')
    print(f'output_freq_code: {output_freq_code} ({type(output_freq_code)})')
    print(f'output_dir: {output_dir_base} ({type(output_dir_base)})')

    local = True
    if local:
        print('\nGetting local directories from config file')
        mapping_factors_dir = config['mapping_factors_dir']

        diags_root = config['diags_root']

        #  METADATA
        metadata_json_dir = config['metadata_json_dir']
        podaac_dir = config['podaac_dir']

        ecco_grid_dir = config['ecco_grid_dir']
        ecco_grid_dir_mds = config['ecco_grid_dir_mds']
    else:
        print('\nGetting AWS Cloud directories from config file')
        # mapping_factors_dir = config['mapping_factors_dir_cloud']

    reload(ecco)

    # Ian's paths --------------------------------------------------------
    # mapping_factors_dir = Path('/home/ifenty/tmp/ecco-v4-podaac-mapping-factors')

    # diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/diags_all')

    # # METADATA
    # metadata_json_dir = Path('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')
    # podaac_dir = Path('/home/ifenty/git_repos_others/ecco-data-pub/metadata')

    # ecco_grid_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4')
    # ecco_grid_dir_mds = Path('/home/ifenty/data/grids/grid_ECCOV4r4')

    # # PODAAC fields
    # ecco_grid_filename = 'ECCO_V4r4_llc90_grid_geometry.nc'
    # --------------------------------------------------------------------
    
    # PODAAC fields
    ecco_grid_filename = config['ecco_grid_filename']

    # Define precision of output files, float32 is standard
    array_precision = np.float32

    # 20 NATIVE GRID GROUPINGS
    #        0 dynamic sea surface height and model sea level anomaly
    # 	 1 ocean bottom pressure and model ocean bottom pressure anomaly
    # 	 2 ocean and sea-ice surface freshwater fluxes
    # 	 3 ocean and sea-ice surface heat fluxes
    # 	 4 atmosphere surface temperature, humidity, wind, and pressure
    # 	 5 ocean mixed layer depth
    # 	 6 ocean and sea-ice surface stress
    # 	 7 sea-ice and snow concentration and thickness
    # 	 8 sea-ice velocity
    # 	 9 sea-ice and snow horizontal volume fluxes
    # 	 10 Gent-McWilliams ocean bolus transport streamfunction
    # 	 11 ocean three-dimensional volume fluxes
    # 	 12 ocean three-dimensional potential temperature fluxes
    # 	 13 ocean three-dimensional salinity fluxes
    # 	 14 sea-ice salt plume fluxes
    # 	 15 ocean potential temperature and salinity
    # 	 16 ocean density, stratification, and hydrostatic pressure
    # 	 17 ocean velocity
    # 	 18 Gent-McWilliams ocean bolus velocity
    # 	 19 ocean three-dimensional momentum tendency

    # ----- > groupings_native_snap (5) = [0, 1, 7, 8, 15]
    # SSH, obp, sea ice and snow, sea ice velocity, TS


    # 13 LATLON GRID GROUPINGS
    #         0 "dynamic sea surface height",
    #         1 "ocean bottom pressure",
    #         2 "ocean and sea-ice surface freshwater fluxes",
    #         3 "ocean and sea-ice surface heat fluxes",
    #         4 "atmosphere surface temperature, humidity, wind, and pressure",
    #         5 "ocean mixed layer depth",
    #         6 "ocean and sea-ice surface stress",
    #         7 "sea-ice and snow concentration and thickness",
    #         8 "sea-ice velocity",
    #         9 "ocean potential temperature and salinity",
    #        10 "ocean density, stratification, and hydrostatic pressure",
    #        11 "ocean velocity",
    #        12 "Gent-McWilliams ocean bolus velocity",

    debug_mode = False

    G = []
    # G, ecco_grid =  generate_netcdfs(output_freq_code,
    #                                 product_type,
    #                                 mapping_factors_dir,
    #                                 output_dir_base,
    #                                 diags_root,
    #                                 metadata_json_dir,
    #                                 podaac_dir,
    #                                 ecco_grid_dir,
    #                                 ecco_grid_dir_mds,
    #                                 ecco_grid_filename,
    #                                 grouping_to_process,
    #                                 time_steps_to_process,
    #                                 array_precision,
    #                                 debug_mode)
