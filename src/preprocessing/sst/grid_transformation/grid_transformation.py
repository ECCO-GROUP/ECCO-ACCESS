import numpy as np
import xarray as xr
import pyresample as pr
import yaml
import requests
import json
import os
import hashlib
import pickle
from datetime import datetime
from netCDF4 import default_fillvals  # pylint: disable=import-error


np.warnings.filterwarnings('ignore')


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq):
    solr_host = config['solr_host']
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, update_body):
    solr_host = config['solr_host']
    solr_collection_name = config['solr_collection_name']

    url = solr_host + solr_collection_name + '/update?commit=true'

    requests.post(url, json=update_body)


def run_locally_wrapper(system_path, source_file_path, remaining_transformations, output_dir):
    try:
        run_locally(system_path, source_file_path,
                    remaining_transformations, output_dir)
    except:
        run_locally(system_path, source_file_path,
                    remaining_transformations, output_dir)
        print('Unable to run local transformation')


def run_locally(system_path, source_file_path, remaining_transformations, output_dir):
    #
    # Code to import ecco utils locally... #
    # NOTE: assumes /src/preprocessing/ECCO-ACCESS
    from pathlib import Path
    import sys

    p = Path(__file__).parents[2]
    generalized_functions_path = Path(f'{p}/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea
    # NOTE: generalized functions added to ecco_cloud_utils __init__.py
    # import generalized_functions as gf
    # END Code to import ecco utils locally... #
    #

    # =====================================================
    # Set configuration options
    # =====================================================
    path_to_yaml = system_path + "/grid_transformation_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    file_name = source_file_path.split('/')[-1]
    dataset = config['dataset_name']

    # =====================================================
    # Load file to transform
    # =====================================================
    print("=====loading " + file_name + " data======")
    ds = xr.open_dataset(source_file_path, decode_times=True)
    ds.attrs['original_file_name'] = file_name

    for grid_name in remaining_transformations.keys():
        fields = remaining_transformations[grid_name]

        # Query Solr for grid path
        fq = ['type_s:grid', f'grid_name_s:{grid_name}']
        grid_metadata = solr_query(config, fq)[0]
        grid_path = grid_metadata['grid_path_s']
        grid_type = grid_metadata['grid_type_s']
        grid_dir = grid_path.rsplit('/', 1)[0] + '/'

        # =====================================================
        # Load grid
        # =====================================================
        print("======loading "+grid_name+" model grid=======")
        # read grid file
        model_grid = xr.open_dataset(grid_path).reset_coords()

        # =====================================================
        # Check for model grid factors
        # =====================================================
        grid_factors = grid_name + '_factors_path_s'
        fq = [f'dataset_s:{dataset}', 'type_s:dataset']

        dataset_metadata = solr_query(config, fq)[0]

        if grid_factors in dataset_metadata.keys():
            factors_path = dataset_metadata[grid_factors]

            print("===loading grid factors===")
            # Load factors
            with open(factors_path, "rb") as f:
                factors = pickle.load(f)

        else:
            print("===creating grid factors===")

            # %%
            #######################################################
            ## BEGIN GRID PRODUCT                                ##

            fq = [f'dataset_s:{dataset}', 'type_s:dataset']
            short_name = dataset_metadata['short_name_s']

            source_grid_min_L, source_grid_max_L, source_grid, \
                data_grid_lons, data_grid_lats = ea.generalized_grid_product(short_name,
                                                                             config['data_res'],
                                                                             config['data_max_lat'],
                                                                             config['area_extent'],
                                                                             config['dims'],
                                                                             config['proj_info'])

            ## END GRID PRODUCT                                  ##
            #######################################################
            # %%

            # %%
            #######################################################
            ## BEGIN MAPPING                                     ##

            # Define the 'swath' (in the terminology of the pyresample module)
            # as the lats/lon pairs of the model grid
            # The routine needs the lats and lons to be one-dimensional vectors.
            target_grid = \
                pr.geometry.SwathDefinition(lons=model_grid.XC.values.ravel(),
                                            lats=model_grid.YC.values.ravel())

            # TODO look if target_grid_radius exists in grid model, otherwise use the line below
            # TODO ask Ian!
            target_grid_radius = 0.5*np.sqrt(model_grid.rA.values.ravel())

            # Compute the mapping between the data and model grid
            source_indices_within_target_radius_i,\
                num_source_indices_within_target_radius_i,\
                nearest_source_index_to_target_index_i = \
                ea.find_mappings_from_source_to_target(source_grid,
                                                       target_grid,
                                                       target_grid_radius,
                                                       source_grid_min_L,
                                                       source_grid_max_L)

            factors = (source_indices_within_target_radius_i,
                       num_source_indices_within_target_radius_i,
                       nearest_source_index_to_target_index_i)

            print('===Saving grid factors===')
            factors_path = f'{grid_dir}{grid_name}_factors_{dataset}'
            with open(factors_path, 'wb') as f:
                pickle.dump(factors, f)

            print('===Updating Solr with factors===')
            query_fq = [f'dataset_s:{config["dataset_name"]}',
                        'type_s:dataset']
            update_body = [
                {
                    "id": solr_query(config, query_fq)[0]['id'],
                    f'{grid_name}_factors_path_s': {"set": factors_path},
                    f'{grid_name}_factors_stored_dt': {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
                }
            ]
            solr_update(config, update_body)

        # =====================================================
        # Run transformation
        # =====================================================
        # Creates or updates Solr entry for this grid/field/granule combination
        # Must query for harvested entry to get origin_checksum and date
        query_fq = [f'dataset_s:{dataset}', 'type_s:harvested',
                    f'pre_transformation_file_path_s:"{source_file_path}"']
        harvested_doc = solr_query(config, query_fq)[0]
        origin_checksum = harvested_doc['checksum_s']
        date = harvested_doc['date_s']

        update_body = []

        for field in fields:
            # Query if entry exists
            query_fq = [f'dataset_s:{dataset}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field["name_s"]}', f'pre_transformation_file_path_s:"{source_file_path}"']

            docs = solr_query(config, query_fq)
            updating = len(docs) > 0

            update_body = []
            transform = {}
            if updating:
                transform['id'] = docs[0]['id']
                transform['transformation_in_progress_b']: {"set": True}
                transform['success_b']: {"set": False}
            else:
                transform['type_s'] = 'transformation'
                transform['date_s'] = date
                transform['dataset_s'] = dataset
                transform['pre_transformation_file_path_s'] = source_file_path
                transform['origin_checksum_s'] = origin_checksum
                transform['grid_name_s'] = grid_name
                transform['field_s'] = field["name_s"]
                transform['transformation_in_progress_b'] = True
                transform['success_b'] = False
            update_body.append(transform)
            solr_update(config, update_body)

        # Returns list of DAs, one for each field in fields
        print("===Running transformations for " + file_name + "===")
        field_DAs = run_in_any_env(
            model_grid, grid_name, grid_type, fields, factors, ds, date, config)

        # =====================================================
        # Save the output in netCDF format
        # =====================================================
        print("=========saving output=========")
        # field is list of dictionaries
        for field, field_DA in zip(fields, field_DAs):

            output_filename = f'{grid_name}_{field["name_s"]}_{file_name}'

            # Define precision of output files from config
            array_precision = getattr(np, config['array_precision'])

            # Define fill values for binary and netcdf
            # ---------------------------------------------
            if array_precision == np.float32:
                netcdf_fill_value = default_fillvals['f4']

            elif array_precision == np.float64:
                netcdf_fill_value = default_fillvals['f8']

            field_DA.values = np.where(np.isnan(field_DA.values),
                                       netcdf_fill_value,
                                       field_DA.values)

            output_path = f'{output_dir}{config["dataset_name"]}/{grid_name}/transformed/{field["name_s"]}/'

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            encoding_each = {'zlib': True,
                             'complevel': 5,
                             'fletcher32': True,
                             '_FillValue': netcdf_fill_value}

            field_DA.to_netcdf(output_path + output_filename)

            # TODO: ask ian about encoding - messes up aggregation (saving as dataset/opening as dataarray)

            # field_DS = field_DA.to_dataset()

            # encoding = {var: encoding_each for var in field_DS.data_vars}

            # coord_encoding_each = {'zlib': True,
            #                        'complevel': 5,
            #                        'fletcher32': True,
            #                        '_FillValue': False}

            # encoding_coords = {
            #     var: coord_encoding_each for var in field_DS.dims}

            # encoding_2 = {**encoding_coords, **encoding}

            # field_DS.to_netcdf(output_path + output_filename,
            #                    encoding=encoding_2)
            # field_DS.close()

            # update with new info in solr
            transformed_location = output_path + output_filename

            # First query for the id
            query_fq = [f'dataset_s:{dataset}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field["name_s"]}', f'pre_transformation_file_path_s:"{source_file_path}"']

            doc_id = solr_query(config, query_fq)[0]['id']

            # Then update the new metadata
            success = True if output_filename != '' else False

            update_body = [
                {
                    "id": doc_id,
                    "filename_s": {"set": output_filename},
                    "transformation_file_path_s": {"set": transformed_location},
                    "transformation_completed_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "transformation_in_progress_b": {"set": False},
                    "success_b": {"set": success},
                    "transformation_checksum_s": {"set": md5(transformed_location)},
                    "transformation_version_s": {"set": config['version']}
                }
            ]

            solr_update(config, update_body)

        print("======saving output DONE=======")


def run_in_any_env(model_grid, model_grid_name, model_grid_type, fields, factors, ds, record_date, config):
    #
    # Code to import ecco utils locally... #
    from pathlib import Path
    import sys

    p = Path(__file__).parents[2]
    generalized_functions_path = Path(f'{p}/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea
    # END Code to import ecco utils locally... #
    #

    source_indices_within_target_radius_i, \
        num_source_indices_within_target_radius_i, \
        nearest_source_index_to_target_index_i = factors

    # Check if ends in z and Drop if it does
    if record_date[-1] == 'Z':
        record_date = record_date[:-1]

    array_precision = getattr(np, config['array_precision'])
    record_file_name = ds.attrs['original_file_name']
    extra_information = config['extra_information']
    time_zone_included_with_time = config['time_zone_included_with_time']

    # Get dataset metadata. Used for dataarray attributes
    fq = ['type_s:dataset', f'dataset_s:{config["dataset_name"]}']
    dataset_metadata = solr_query(config, fq)[0]

    field_DAs = []

    original_dataset_metadata = {
        key: dataset_metadata[key] for key in dataset_metadata.keys() if 'original' in key}

    # fields is a list of dictionaries
    for data_field_info in fields:

        # field_DA = ea.generalized_transform_to_model_grid_solr(data_field_info, record_date, model_grid, model_grid_type,
        #                                                        array_precision, record_file_name, original_dataset_metadata,
        #                                                        extra_information, ds, factors, time_zone_included_with_time,
        #                                                        model_grid_name)

        # initialize notes for this record
        record_notes = ''

        # set data info values
        data_field = data_field_info['name_s']
        standard_name = data_field_info['standard_name_s']
        long_name = data_field_info['long_name_s']
        units = data_field_info['units_s']

        # create empty data array
        data_DA = ea.make_empty_record(standard_name, long_name, units,
                                       record_date,
                                       model_grid, model_grid_type,
                                       array_precision, variable_name=data_field)

        print(data_DA)

        # add some metadata to the newly formed data array object
        data_DA.attrs['original_filename'] = record_file_name
        data_DA.attrs['original_field_name'] = data_field
        data_DA.attrs['interplation_parameters'] = 'bin averaging'
        data_DA.attrs['interplation_code'] = 'pyresample'
        data_DA.attrs['interpolation_date'] = \
            str(np.datetime64(datetime.now(), 'D'))

        data_DA.time.attrs['long_name'] = 'center time of averaging period'

        data_DA.attrs['original_dataset_title'] = dataset_metadata['original_dataset_title_s']
        data_DA.attrs['original_dataset_short_name'] = dataset_metadata['original_dataset_short_name_s']
        data_DA.attrs['original_dataset_url'] = dataset_metadata['original_dataset_url_s']
        data_DA.attrs['original_dataset_reference'] = dataset_metadata['original_dataset_reference_s']
        data_DA.attrs['original_dataset_doi'] = dataset_metadata['original_dataset_doi_s']
        data_DA.attrs['interpolated_grid_id'] = model_grid_name
        data_DA.name = f'{data_field}_interpolated_to_{model_grid_name}'

        # load the file, do the mapping and update the record times
        print('reading ', record_file_name)

        if 'transpose' in extra_information:
            orig_data = ds[data_field].values[0, :].T
        else:
            orig_data = ds[data_field].values

        # see if we have any valid data
        if np.sum(~np.isnan(orig_data)) > 0:

            data_model_projection = ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                                num_source_indices_within_target_radius_i,
                                                                nearest_source_index_to_target_index_i,
                                                                orig_data, model_grid.XC.shape)

            # put the new data values into the data_DA array.
            # --where the mapped data are not nan, replace the original values
            # --where they are nan, just leave the original values alone
            data_DA.values = np.where(~np.isnan(data_model_projection),
                                      data_model_projection, data_DA.values)

        else:
            print('file loaded but empty')
            record_notes = record_notes + ' -- empty record -- '

        # update time values
        if 'time_bounds_var' in extra_information:
            data_DA.time_start.values[0] = ds.Time_bounds[0][0].values
            data_DA.time_end.values[0] = ds.Time_bounds[0][1].values
        elif 'no_time' in extra_information:
            data_DA.time_start.values[0] = record_date
            data_DA.time_end.values[0] = record_date
        elif 'no_time_dashes' in extra_information:
            new_start_time = f'{ds.time_coverage_start[0:4]}-{ds.time_coverage_start[4:6]}-{ds.time_coverage_start[6:8]}'
            new_end_time = f'{ds.time_coverage_end[0:4]}-{ds.time_coverage_end[4:6]}-{ds.time_coverage_end[6:8]}'
            data_DA.time_start.values[0] = new_start_time
            data_DA.time_end.values[0] = new_end_time
        elif config['time_zone_included_with_time']:
            data_DA.time_start.values[0] = ds.time_coverage_start[:-1]
            data_DA.time_end.values[0] = ds.time_coverage_end[:-1]
        else:
            data_DA.time_start.values[0] = ds.time_coverage_start
            data_DA.time_end.values[0] = ds.time_coverage_end

        if 'time_var' in extra_information:
            data_DA.time.values[0] = ds.Time[0].values
        else:
            data_DA.time.values[0] = record_date

        data_DA.attrs['notes'] = record_notes

        field_DAs.append(data_DA)

    return field_DAs


# Placeholder AWS code


# def run_using_aws_wrapper(s3, filename):
#     try:
#         run_using_aws(s3, filename)
#     finally:
#         with open("grid_transformation_config.yaml", "r") as stream:
#             config = yaml.load(stream)

#         decrement_solr_count(config)


# def run_using_aws(s3, filename):
#     # =====================================================
#     # Set configuration options
#     # =====================================================

#     with open("grid_transformation_config.yaml", "r") as stream:
#         config = yaml.load(stream)

#     source_bucket_name = config['source_bucket']
#     target_bucket_name = config['target_bucket']
#     output_suffix = config['output_suffix']
#     output_dir = config['aws_output_dir']
#     version = config['version']

#     ###########################################

#     source_bucket = s3.Bucket(source_bucket_name)
#     target_bucket = s3.Bucket(target_bucket_name)

#     # =====================================================
#     # Load model grid coordinates (longitude, latitude)
#     # =====================================================
#     print("======loading model grid=======")
#     dt = np.dtype('>f')

#     # read longitude file
#     datafile = config['aws_lon_grid_path']
#     xgrid_object = s3.Object(source_bucket_name, datafile)
#     f = xgrid_object.get()['Body'].read()
#     model_lons_1d = np.frombuffer(f, dtype=dt, count=-1)

#     # read latitude file
#     datafile = config['aws_lat_grid_path']
#     ygrid_object = s3.Object(source_bucket_name, datafile)
#     f = ygrid_object.get()['Body'].read()
#     model_lats_1d = np.frombuffer(f, dtype=dt, count=-1)

#     print("====loading model grid DONE====")

#     # =====================================================
#     # Load the sea ice data
#     # =====================================================
#     print("=====loading sea ice data======")
#     print("file being transformed is: ", filename)

#     source_bucket.download_file(filename, "/tmp/icedata.nc")
#     # print(file_stream)
#     # ds = xr.open_dataset(obj.get()['Body'].read(), decode_times=True)
#     ds = xr.open_dataset("/tmp/icedata.nc", decode_times=True)

#     print("===loading sea ice data DONE===")

#     # =====================================================
#     # Transform/remap data to grid
#     # =====================================================
#     data_model_projection = run_in_any_env(model_lons_1d, model_lats_1d, ds)

#     # =====================================================
#     # ### Save the output in the model grid format
#     # =====================================================
#     print("=========saving output=========")
#     basefilename = filename.split('/')[-1][:-3]  # to remove ".nc"
#     output_filename = output_dir + basefilename + output_suffix
#     local_fp = '/tmp/' + basefilename + output_suffix

#     dt_out = np.dtype('>f4')

#     with open(local_fp, 'wb') as f:
#         data_model_projection.astype(dt_out).tofile(f)

#     # target_object = s3.Object('ecco-ames',output_filename)

#     # #error
#     # target_object.put(Body=data_model_projection.astype(dt_out))

#     target_bucket.upload_file(local_fp, output_filename)
#     print(output_filename)
#     # update with new info in solr
#     path = "s3://" + target_bucket_name + '/' + output_filename
#     update_solr(config, filename, path)

#     print("======saving output DONE=======")
