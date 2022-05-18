"""
Created on Thur May 12 15:41:00 2022

Author: Duncan Bark
Adapted from ifenty's "eccov4r4_gen_for_podaac.py"

"""

import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from pathlib import Path
from importlib import reload

path_to_ecco_group = Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(f'{path_to_ecco_group}/ECCO-ACCESS/ecco-cloud-utils')
sys.path.append(f'{path_to_ecco_group}/ECCOv4-py')
import ecco_v4_py as ecco
import ecco_cloud_utils as ea

from . import gen_netcdf_utils as ut

# -------------------------------------------------------------------------------------------------


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

    print('\nBEGIN: generate_netcdfs')
    print('OFC', output_freq_code)
    print('PDT', product_type)
    print('GTP', grouping_to_process)
    print('TSP', time_steps_to_process)
    print('DBG', debug_mode)
    print('')

    # Define fill values for binary and netcdf
    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999
    if array_precision == np.float32:
        # binary_output_dtype = '>f4'
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif array_precision == np.float64:
        # binary_output_dtype = '>f8'
        netcdf_fill_value = nc4.default_fillvals['f8']

    # num of depth levels
    nk = 50

    # levels to process (for testing purposes make less than nk)
    max_k = 50

    ecco_start_time = np.datetime64('1992-01-01T12:00:00')
    ecco_end_time   = np.datetime64('2017-12-31T12:00:00')


    # ======================================================================================================================
    # METADATA SETUP
    # ======================================================================================================================
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

    # load METADATA
    print('\nLOADING METADATA')
    metadata = {}

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

    all_metadata = {'var_native':variable_metadata_native, 
                    'var_latlon':variable_metadata_latlon, 
                    'coord_native':coordinate_metadata_for_native_datasets, 
                    'coord_latlon':coordinate_metadata_for_latlon_datasets, 
                    'global_all':global_metadata_for_all_datasets, 
                    'global_native':global_metadata_for_native_datasets, 
                    'global_latlon':global_metadata_for_latlon_datasets}
    
    # create dataset description head
    dataset_description = dataset_description_head + grouping['name'] + dataset_description_tail
    # ======================================================================================================================


    # ======================================================================================================================
    # NATIVE vs LATLON SETUP (ECCO GRID & MASK SETUP)
    # ======================================================================================================================
    # load ECCO grid
    ecco_grid = xr.open_dataset(ecco_grid_dir / ecco_grid_filename)
    print(ecco_grid)

    # land masks
    ecco_land_mask = {}
    ecco_land_mask['c_nan']  = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask['c_nan'].values = np.where(ecco_land_mask['c_nan']==True, 1, np.nan)
    ecco_land_mask['w_nan']  = ecco_grid.maskW.copy(deep=True)
    ecco_land_mask['w_nan'].values = np.where(ecco_land_mask['w_nan']==True, 1, np.nan)
    ecco_land_mask['s_nan']  = ecco_grid.maskS.copy(deep=True)
    ecco_land_mask['s_nan'].values = np.where(ecco_land_mask['s_nan']==True, 1, np.nan)

    print('\nproduct type', product_type)
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

        ll_grid, target_grid, bounds = ut.latlon_setup(ea, ecco_grid, 
                                                        mapping_factors_dir, nk, 
                                                        ecco_land_mask['c_nan'], debug_mode)
    
    # Make depth bounds
    depth_bounds = np.zeros((nk,2))
    tmp = np.cumsum(ecco_grid.drF.values)

    for k in range(nk):
        if k == 0:
            depth_bounds[k,0] = 0.0
        else:
            depth_bounds[k,0] = -tmp[k-1]
        depth_bounds[k,1] = -tmp[k]
    # ======================================================================================================================


    # ======================================================================================================================
    # DIRECTORIES & FILE PATHS
    # ======================================================================================================================
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
    # ======================================================================================================================


    # ======================================================================================================================
    # GROUPINGS
    # ======================================================================================================================
    # show groupings
    print('\nAll groupings')
    for gi, gg in enumerate(groupings_for_native_datasets):
        print('\t', gi, gg['name'])

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

    # define empty list of gcmd keywords pertaining to this dataset
    grouping_gcmd_keywords = []
    # ======================================================================================================================


    # ======================================================================================================================
    # CREATE VARIABLES & TIME STEPS
    # ======================================================================================================================
    # find variables in dataset
    tmp = grouping['fields'].split(',')
    vars_to_load  = []
    for var in tmp:
        vars_to_load.append(var.strip())

    # find directories with variables
    var_directories = {}

    for var in vars_to_load:
        var_match =  "".join([var, "_", period_suffix])
        num_matching_dirs = 0
        for fp in field_paths:
            if var_match in str(fp):
                var_directories[var] = fp
                num_matching_dirs += 1
        if num_matching_dirs == 0:
            print('>>>>>> no match found for ', var)
        elif num_matching_dirs > 1 :
            print('>>>>>> more than one matching dir for ', var)

    print('\nDirectories with the variables in the grouping')
    for var in vars_to_load:
        print('... ', var, var_directories[var])

    print('\nDetermining time steps to load')
    # determine which time steps to process
    if time_steps_to_process == 'all':
        print('...using all time steps')
        time_steps_to_process = ut.find_all_time_steps(vars_to_load, var_directories)
    else:
        print('...using provided time steps to process list ', time_steps_to_process)
    # ======================================================================================================================


    # ======================================================================================================================
    # PROCESS EACH TIME LEVEL
    # ======================================================================================================================
    print('\nLooping through time levels')
    for cur_ts_i, cur_ts in enumerate(time_steps_to_process):

        # ==================================================================================================================
        # CALCULATE TIMES
        # ==================================================================================================================
        print('\n\n=== TIME LEVEL ===', str(cur_ts_i).zfill(5), str(cur_ts).zfill(10))
        print('\n')
        time_delta = np.timedelta64(cur_ts, 'h')
        cur_ts = int(cur_ts)
        cur_time = ecco_start_time + time_delta
        times = [pd.to_datetime(str(cur_time))]

        if 'AVG' in output_freq_code:
            tb, record_center_time = ecco.make_time_bounds_from_ds64(np.datetime64(times[0]), output_freq_code)
            print('ORIG  tb, ct ', tb, record_center_time)

            # fix beginning of last record
            if tb[1].astype('datetime64[D]') == ecco_end_time.astype('datetime64[D]'):
                print('end time match ')
                time_delta = np.timedelta64(12,'h')
                rec_avg_start = tb[0] + time_delta
                rec_avg_end   = tb[1]
                rec_avg_delta = rec_avg_end - rec_avg_start
                rec_avg_middle = rec_avg_start + rec_avg_delta/2

                tb[0] = rec_avg_start
                record_center_time = rec_avg_middle

            # truncate to ecco_start_time
            if tb[0].astype('datetime64[D]') == ecco_start_time.astype('datetime64[D]'):
                print('start time match ')
                rec_avg_start = ecco_start_time
                rec_avg_end   = tb[1]
                rec_avg_delta = tb[1] - ecco_start_time
                rec_avg_middle = rec_avg_start + rec_avg_delta/2

                tb[0] = ecco_start_time
                record_center_time = rec_avg_middle

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
        # ==================================================================================================================


        # ==================================================================================================================
        # LOOP THROUGH VARIABLES & CREATE DATASET
        # ==================================================================================================================
        F_DS_vars = []

        if not debug_mode:
            # Load variables and place them in the dataset
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

                # Load latlon vs native variable
                if product_type == 'latlon':
                    F_DS = ut.latlon_load(ea, ecco, ecco_grid, ll_grid, target_grid, 
                                            mds_var_dir, mds_file, record_end_time, nk, 
                                            max_k, dataset_dim, var, output_freq_code)
                    
                elif product_type == 'native':
                    F_DS = ut.native_load(ecco, var, ecco_land_mask, ecco_grid_dir_mds, 
                                            mds_var_dir, mds_file, output_freq_code, cur_ts)
                
                record_times = {'start':record_start_time, 'center':record_center_time, 'end':record_end_time}
                F_DS = ut.global_DS_changes(output_freq_code, grouping, var, array_precision, ecco_grid, depth_bounds, product_type, bounds, netcdf_fill_value, dataset_dim, record_times)

                # add this dataset to F_DS_vars and repeat for next variable
                F_DS_vars.append(F_DS)


            # merge the data arrays to make one DATASET
            print('\n... merging F_DS_vars')
            G = xr.merge((F_DS_vars))

            netcdf_output_filename, encoding = ut.set_metadata(ecco, G, product_type, all_metadata, dataset_dim, 
                                                                output_freq_code, netcdf_fill_value, 
                                                                grouping, filename_tail, output_dir_freq, 
                                                                dataset_description, podaac_dir)

            # SAVE DATASET
            print('\n... saving to netcdf ', netcdf_output_filename)
            G.load()

            G.to_netcdf(netcdf_output_filename, encoding=encoding)
            G.close()

            print('\n... checking existence of new file: ', netcdf_output_filename.exists())
            print('\n')
        # ==================================================================================================================
    # =============================================================================================

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
    # output_freq_codes  one of either 'AVG_DAY' or 'AVG_MON'
    
    # time_steps_to_process : one of
    # 'all'
    # a list [1,3,4,...]

    # grouping to process:
    #   a single number

    # diags_root:
    #   ECCO FIELD INPUT DIRECTORY
    #   model diagnostic output
    #   subdirectories must be:
    #       'diags_all/diags_mon'
    #       'diags_all/diags_mon'
    #       'diags_all/diags_inst'

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
        mapping_factors_dir = Path(config['mapping_factors_dir'])

        diags_root = Path(config['diags_root'])

        # METADATA
        metadata_json_dir = Path(config['metadata_json_dir'])
        podaac_dir = Path(config['podaac_dir'])

        ecco_grid_dir = Path(config['ecco_grid_dir'])
        ecco_grid_dir_mds = Path(config['ecco_grid_dir_mds'])
    else:
        print('\nGetting AWS Cloud directories from config file')
        # mapping_factors_dir = config['mapping_factors_dir_cloud']
    
    # PODAAC fields
    ecco_grid_filename = config['ecco_grid_filename']

    # Define precision of output files, float32 is standard
    array_precision = np.float32

    debug_mode = False

    reload(ecco)

    G = []
    G, ecco_grid =  generate_netcdfs(output_freq_code,
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
                                    array_precision,
                                    debug_mode)

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
