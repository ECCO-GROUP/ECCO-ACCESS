"""
Created May 18, 2022

Author: Duncan Bark
Adapted from ifenty's "eccov4r4_gen_for_podaac.py"

"""

import os
import sys
import json
import boto3
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from pathlib import Path
from collections import defaultdict

sys.path.append(f'{Path(__file__).parent.resolve()}')
import ecco_v4_py as ecco
import ecco_cloud_utils as ea
import gen_netcdf_utils as ut

# ==========================================================================================================================


def generate_netcdfs(output_freq_code,
                     product_type,
                     grouping_to_process,
                     time_steps_to_process,
                     config_metadata,
                     debug_mode,
                     local,
                     credentials):

    print('\nBEGIN: generate_netcdfs')
    print('OFC', output_freq_code)
    print('PDT', product_type)
    print('GTP', grouping_to_process)
    print('TSP', time_steps_to_process)
    print('DBG', debug_mode)
    print('')

    # Setup S3
    buckets = (config_metadata['source_bucket'], config_metadata['output_bucket'])
    if not local and buckets != None and credentials != None:
        boto3.setup_default_session(profile_name=credentials['profile_name'])
        s3 = boto3.client('s3')
        model_granule_bucket, processed_data_bucket = buckets

    # Define fill values for binary and netcdf
    # ECCO always uses -9999 for missing data.
    binary_fill_value = config_metadata['binary_fill_value']
    if config_metadata['array_precision'] == 'float32':
        # binary_output_dtype = '>f4'
        array_precision = np.float32
        netcdf_fill_value = nc4.default_fillvals['f4']
    else:
        # binary_output_dtype = '>f8'
        array_precision = np.float64
        netcdf_fill_value = nc4.default_fillvals['f8']

    # num of depth levels
    nk = config_metadata['num_vertical_levels']

    ecco_start_time = np.datetime64(config_metadata['model_start_time'])
    ecco_end_time   = np.datetime64(config_metadata['model_end_time'])


    # ======================================================================================================================
    # METADATA SETUP
    # ======================================================================================================================
    # Define tail for dataset description (summary)
    dataset_description_tail_native = config_metadata['dataset_description_tail_native']
    dataset_description_tail_latlon = config_metadata['dataset_description_tail_latlon']

    filename_tail_native = config_metadata['filename_tail_native']
    filename_tail_latlon = config_metadata['filename_tail_latlon']

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
        with open(str(config_metadata['metadata_dir'] / mf_e), 'r') as fp:
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
    # ======================================================================================================================


    # ======================================================================================================================
    # NATIVE vs LATLON SETUP
    # ======================================================================================================================
    print('\nproduct type', product_type)
    if product_type == 'native':
        dataset_description_tail = dataset_description_tail_native
        filename_tail = filename_tail_native
        groupings = groupings_for_native_datasets
        output_dir_type = config_metadata['output_dir_base'] / 'native'

    elif product_type == 'latlon':
        dataset_description_tail = dataset_description_tail_latlon
        filename_tail = filename_tail_latlon
        groupings = groupings_for_latlon_datasets
        output_dir_type = config_metadata['output_dir_base'] / 'lat-lon'

        latlon_bounds, depth_bounds, target_grid, wet_pts_k = ut.get_latlon_grid(config_metadata['mapping_factors_dir'], debug_mode)
        # CHANGE TO TWO DICTS (BOUNDS, and GRID)
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
    # ======================================================================================================================


    # ======================================================================================================================
    # DIRECTORIES & FILE PATHS
    # ======================================================================================================================
    diags_root = config_metadata['model_data_dir']

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
    if local:
        if not output_dir_freq.exists():
            try:
                output_dir_freq.mkdir()
                print('... made %s ' % output_dir_freq)
            except:
                print('...cannot make %s ' % output_dir_freq)
    else:
        print(f'Making output bucket {processed_data_bucket}')
        # *****
        # TODO: Make bucket if not present (maybe)
        # *****


    # get files
    s3_files = defaultdict(list)
    if local:
        # get list of files from local directory
        model_granule_roots = np.sort(list(mds_diags_root_dir.glob('*' + period_suffix + '*')))
    else:
        # get list of files from S3 bucket
        response = s3.list_objects(Bucket=model_granule_bucket)
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            print(f'Unable to collect objects in bucket {model_granule_bucket}')
            return -1
        else:
            model_granule_roots = []
            for k in response['Contents']:
                k = k['Key']
                if '.data' in k:
                    var = k.split('/')[-2].replace(f'_{period_suffix}', '')
                    s3_files[var].append(k)
                    k_root = k.split(f'{period_suffix}/')[0] + period_suffix
                    if k_root not in model_granule_roots:
                        model_granule_roots.append(k_root)


    ## load variable file and directory names
    print ('...number of subdirectories found ', len(model_granule_roots))
    all_field_names = []

    # extract record name out of full directory
    if local:
        for f in model_granule_roots:
            all_field_names.append(f.name)
    else:
        for f in model_granule_roots:
            all_field_names.append(f.split('/')[-2])

    print(all_field_names)

    # create dataset description head
    dataset_description = dataset_description_head + grouping['name'] + dataset_description_tail
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
        for fp in model_granule_roots:
            if var_match == str(fp).split('/')[-1]:
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
        time_steps_to_process = ut.find_all_time_steps(vars_to_load, var_directories, local, s3_files)
    else:
        print('...using provided time steps to process list ', time_steps_to_process)
    # ======================================================================================================================


    # ======================================================================================================================
    # PROCESS EACH TIME LEVEL
    # ======================================================================================================================
    # load ECCO grid
    ecco_grid = xr.open_dataset(config_metadata['ecco_grid_dir'] / config_metadata['ecco_grid_filename'])
    print(ecco_grid)

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

        record_times = {'start':record_start_time, 'center':record_center_time, 'end':record_end_time}
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

                if local:
                    mds_file = list(mds_var_dir.glob(var + '*' + str(cur_ts).zfill(10) + '*.data'))
                else:
                    mds_file = []
                    for f in s3_files[var]:
                        if str(cur_ts).zfill(10) in f:
                            mds_file.append(f)

                if len(mds_file) != 1:
                    print('invalid # of mds files')
                    print(mds_file)
                    sys.exit()
                else:
                    mds_file = mds_file[0]

                # Download mds_file from S3
                if not local:
                    # temp_mds_var_dir = f''
                    if not Path(mds_var_dir).exists():
                        try:
                            Path(mds_var_dir).mkdir(parents=True, exist_ok=True)
                        except:
                            print (f'cannot make {mds_var_dir}')
                            sys.exit()
                    s3.download_file(model_granule_bucket, mds_file, mds_file)
                    

                # Load latlon vs native variable
                if product_type == 'latlon':
                    F_DS = ut.latlon_load(ea, ecco, ecco_grid.Z.values, wet_pts_k, target_grid, 
                                            mds_var_dir, mds_file, record_end_time, nk, 
                                            dataset_dim, var, output_freq_code, config_metadata['mapping_factors_dir'])
                    
                elif product_type == 'native':
                    F_DS = ut.native_load(ecco, var, ecco_grid, config_metadata['ecco_grid_dir_mds'], 
                                            mds_var_dir, mds_file, output_freq_code, cur_ts)

                # delete downloaded file from cloud disks
                if not local:
                    os.remove(mds_file)
                
                F_DS = ut.global_DS_changes(F_DS, output_freq_code, grouping, var,
                                            array_precision, ecco_grid, depth_bounds, product_type, 
                                            latlon_bounds, netcdf_fill_value, dataset_dim, record_times)

                # TODO: Figure out way to not need to append each var DS to a list and merge via xarray. New way should
                # do it in less memory

                # add this dataset to F_DS_vars and repeat for next variable
                F_DS_vars.append(F_DS)

            # merge the data arrays to make one DATASET
            print('\n... merging F_DS_vars')
            G = xr.merge((F_DS_vars))

            # delete F_DS_vars from memory
            del(F_DS_vars)

            podaac_dir = config_metadata['metadata_dir'] / config_metadata['podaac_metadata_filename']
            G, netcdf_output_filename, encoding = ut.set_metadata(ecco, G, product_type, all_metadata, dataset_dim, 
                                                                output_freq_code, netcdf_fill_value, 
                                                                grouping, filename_tail, output_dir_freq, 
                                                                dataset_description, podaac_dir)

            # SAVE DATASET
            print('\n... saving to netcdf ', netcdf_output_filename)
            G.load()

            G.to_netcdf(netcdf_output_filename, encoding=encoding)
            G.close()

            print('\n... checking existence of new file: ', netcdf_output_filename.exists())


            # Upload output netcdf to s3
            if not local:
                print('\n... uploading new file to S3 bucket')
                name = str(netcdf_output_filename).replace(f'{str(config_metadata["output_dir_base"])}/', '')
                try:
                    response = s3.upload_file(str(netcdf_output_filename), processed_data_bucket, name)
                    print(f'\n... Uploaded {netcdf_output_filename} to bucket {processed_data_bucket}')
                except:
                    print(f'Unable to upload file {netcdf_output_filename} to bucket {processed_data_bucket}')
                    sys.exit()
                os.remove(netcdf_output_filename)

            print('\n')
        # ==================================================================================================================
    # =============================================================================================

    return
