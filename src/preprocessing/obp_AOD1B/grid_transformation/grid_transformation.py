import os
import sys
import json
import yaml
import pickle
import hashlib
import requests
import numpy as np
import xarray as xr
import pyresample as pr
from pathlib import Path
from datetime import datetime

np.warnings.filterwarnings('ignore')


# Creates checksum from filename
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Queries Solr based on config information and filter query
# Returns list of Solr entries (docs)
def solr_query(config, solr_host, fq):
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


# Posts update to Solr with provided update body
# Optional return of posting status code
def solr_update(config, solr_host, update_body, r=False):
    solr_collection_name = config['solr_collection_name']

    url = solr_host + solr_collection_name + '/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


# Calls run_locally and catches any errors
def run_locally_wrapper(source_file_path, remaining_transformations, output_dir, path=''):
    # try:
    run_locally(source_file_path,
                remaining_transformations, output_dir, path=path)
    # except Exception as e:
    #     print(e)
    #     print('Unable to run local transformation')


# Performs and saves locally all remaining transformations for a given source granule
# Updates Solr with transformation entries and updates descendants, and dataset entries
def run_locally(source_file_path, remaining_transformations, output_dir, path=''):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    if path:
        path_to_yaml = f'{path}/grid_transformation_config.yaml'
    else:
        path_to_yaml = f'{os.path.dirname(sys.argv[0])}/grid_transformation_config.yaml'
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    # =====================================================
    # Code to import ecco utils locally...
    # =====================================================
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[5]}/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    # =====================================================
    # Set configuration options
    # =====================================================
    file_name = source_file_path.split('/')[-1]
    dataset_name = config['ds_name']
    transformation_version = config['version']

    solr_host = config['solr_host_local']

    # Query Solr for dataset entry
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    dataset_metadata = solr_query(config, solr_host, fq)[0]

    # Query Solr for harvested entry to get origin_checksum and date
    print(source_file_path)
    query_fq = [f'dataset_s:{dataset_name}', 'type_s:harvested',
                f'pre_transformation_file_path_s:"{source_file_path}"']
    harvested_metadata = solr_query(config, solr_host, query_fq)[0]
    origin_checksum = harvested_metadata['checksum_s']
    date = harvested_metadata['date_s']

    # If data is stored in hemispheres, use that hemisphere when naming files and updating Solr
    # Otherwise, leave it blank
    # While using hemi in naming files, care has been taken to ensure proper filenames are created when working with both types
    if 'hemisphere_s' in harvested_metadata.keys():
        hemi = f'_{harvested_metadata["hemisphere_s"]}'
    else:
        hemi = ''

    transformation_successes = True
    transformation_file_paths = {}

    # =====================================================
    # Load file to transform
    # =====================================================
    print(f'=====loading {file_name} data======')
    ds = xr.open_dataset(source_file_path, decode_times=True)
    ds.attrs['original_file_name'] = file_name

    # Iterate through grids in remaining_transformations
    for grid_name in remaining_transformations.keys():
        fields = remaining_transformations[grid_name]

        # Query Solr for grid metadata
        fq = ['type_s:grid', f'grid_name_s:{grid_name}']
        grid_metadata = solr_query(config, solr_host, fq)[0]

        grid_path = grid_metadata['grid_path_s']
        grid_type = grid_metadata['grid_type_s']
        grid_dir = grid_path.rsplit('/', 2)[0] + '/'

        # =====================================================
        # Load grid
        # =====================================================
        print(f'======Loading {grid_name} model grid=======')
        model_grid = xr.open_dataset(grid_path).reset_coords()

        # =====================================================
        # Make model grid factors if not present locally
        # =====================================================
        grid_factors = f'{grid_name}{hemi}_factors_path_s'
        grid_factors_version = f'{grid_name}{hemi}_factors_version_f'

        if grid_factors_version in dataset_metadata.keys() and transformation_version == dataset_metadata[grid_factors_version]:
            factors_path = dataset_metadata[grid_factors]

            print(f'===Loading {grid_name} factors===')
            with open(factors_path, "rb") as f:
                factors = pickle.load(f)

        else:
            print(f'===Creating {grid_name} factors===')

            fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
            short_name = dataset_metadata['short_name_s']

            data_res = config['data_res']

            # If data_res is fractional, convert from string to float
            if type(data_res) is str and '/' in data_res:
                num, den = data_res.replace(' ', '').split('/')
                data_res = float(num) / float(den)

            # Use hemisphere specific variables if data is hemisphere specific
            if hemi:
                hemi_dim = config[f'dims{hemi}']
                hemi_area_extent = config[f'area_extent{hemi}']
                hemi_proj_info = config[f'proj_info{hemi}']
                hemi_data_max_lat = config[f'data_max_lat{hemi}']

                source_grid_min_L, source_grid_max_L, source_grid, \
                    data_grid_lons, data_grid_lats = ea.generalized_grid_product(short_name,
                                                                                 data_res,
                                                                                 hemi_data_max_lat,
                                                                                 hemi_area_extent,
                                                                                 hemi_dim,
                                                                                 hemi_proj_info)
            else:
                source_grid_min_L, source_grid_max_L, source_grid, \
                    data_grid_lons, data_grid_lats = ea.generalized_grid_product(short_name,
                                                                                 data_res,
                                                                                 config['data_max_lat'],
                                                                                 config['area_extent'],
                                                                                 config['dims'],
                                                                                 config['proj_info'])

            # Define the 'swath' as the lats/lon pairs of the model grid
            target_grid = pr.geometry.SwathDefinition(lons=model_grid.XC.values.ravel(),
                                                      lats=model_grid.YC.values.ravel())

            # Retrieve target_grid_radius from model_grid file
            if 'effective_grid_radius' in model_grid:
                target_grid_radius = model_grid.effective_grid_radius.values.ravel()
            elif 'effective_radius' in model_grid:
                target_grid_radius = model_grid.effective_radius.values.ravel()
            elif 'rA' in model_grid:
                target_grid_radius = 0.5*np.sqrt(model_grid.rA.values.ravel())
            else:
                print(f'{grid_name} grid not supported')

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

            print(f'===Saving {grid_name} factors===')
            factors_path = f'{grid_dir}grid_factors/{dataset_name}/'

            # Create directory if needed and save factors
            if not os.path.exists(factors_path):
                os.makedirs(factors_path)

            factors_path += f'{grid_name}_factors'

            if '\\' in factors_path:
                factors_path = factors_path.replace('\\', '/')

            with open(factors_path, 'wb') as f:
                pickle.dump(factors, f)

            print('===Updating Solr with factors===')
            # Query Solr for dataset entry
            query_fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
            doc_id = solr_query(config, solr_host, query_fq)[0]['id']

            # Update Solr dataset entry with factors metadata
            update_body = [
                {
                    "id": doc_id,
                    f'{grid_name}{hemi}_factors_path_s': {"set": factors_path},
                    f'{grid_name}{hemi}_factors_stored_dt': {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    f'{grid_name}{hemi}_factors_version_f': {"set": transformation_version}
                }
            ]

            r = solr_update(config, solr_host, update_body, r=True)

            if r.status_code == 200:
                print('Successfully updated Solr with factors information')
            else:
                print('Failed to update Solr with factors information')

        update_body = []

        # Iterate through remaining transformation fields
        for field in fields:
            field_name = field["name_s"]

            # Query if grid/field combination transformation entry exists
            query_fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']
            docs = solr_query(config, solr_host, query_fq)

            update_body = []
            transform = {}

            # If grid/field combination transformation exists, update transformation status
            # Otherwise initialize new transformation entry
            if len(docs) > 0:
                # Reset status fields
                transform['id'] = docs[0]['id']
                transform['transformation_in_progress_b']: {"set": True}
                transform['success_b']: {"set": False}
                r = solr_update(config, solr_host, transform, r=True)
            else:
                # Initialize new transformation entry
                transform['type_s'] = 'transformation'
                transform['date_s'] = date
                transform['dataset_s'] = dataset_name
                transform['pre_transformation_file_path_s'] = source_file_path
                if hemi:
                    transform['hemisphere_s'] = hemi
                transform['origin_checksum_s'] = origin_checksum
                transform['grid_name_s'] = grid_name
                transform['field_s'] = field_name
                transform['transformation_in_progress_b'] = True
                transform['success_b'] = False
                update_body.append(transform)
                r = solr_update(config, solr_host, update_body, r=True)

            if r.status_code != 200:
                print(
                    f'Failed to update Solr transformation status for {dataset_name} on {date}')

        # =====================================================
        # Run transformation
        # =====================================================

        print(f'===Running transformations for {file_name}===')

        # Returns list of transformed DAs, one for each field in fields

        field_DAs = run_in_any_env(
            model_grid, grid_name, grid_type, fields, factors, ds, date, dataset_metadata, config)

        # =====================================================
        # Save the output in netCDF format
        # =====================================================
        print(f'======saving {file_name} output=======')

        # Save each transformed granule for the current field
        for field, (field_DA, success) in zip(fields, field_DAs):
            field_name = field["name_s"]

            # Change .bz2 file extension to .nc
            if 'bz2' in file_name:
                file_name = file_name[:-3] + 'nc'

            output_filename = f'{grid_name}_{field_name}_{file_name}'
            output_path = f'{output_dir}{dataset_name}/{grid_name}/transformed/{field_name}/'
            transformed_location = f'{output_path}{output_filename}'

            if '\\' in transformed_location:
                transformed_location = transformed_location.replace('\\', '/')

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            field_DS = field_DA.to_dataset()
            field_DS.to_netcdf(output_path + output_filename)
            field_DS.close()

            # Query Solr for transformation entry
            query_fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                        f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']

            docs = solr_query(config, solr_host, query_fq)
            doc_id = solr_query(config, solr_host, query_fq)[0]['id']

            transformation_successes = transformation_successes and success
            transformation_file_paths[f'{grid_name}_{field_name}_transformation_file_path_s'] = transformed_location

            # Update Solr transformation entry with file paths and status
            update_body = [
                {
                    "id": doc_id,
                    "filename_s": {"set": output_filename},
                    "transformation_file_path_s": {"set": transformed_location},
                    "transformation_completed_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "transformation_in_progress_b": {"set": False},
                    "success_b": {"set": success},
                    "transformation_checksum_s": {"set": md5(transformed_location)},
                    "transformation_version_f": {"set": transformation_version}
                }
            ]

            r = solr_update(config, solr_host, update_body, r=True)

            if r.status_code != 200:
                print(
                    f'Failed to update Solr transformation entry for {field["name_s"]} in {dataset_name} on {date}')

        print(f'======saving {file_name} output DONE=======')

    # Query Solr for descendants entry by date
    query_fq = [f'dataset_s:{dataset_name}',
                'type_s:descendants', f'date_s:{date[:10]}*']
    if hemi:
        query_fq.append(f'hemisphere_s:{hemi[1:]}')

    docs = solr_query(config, solr_host, query_fq)
    doc_id = solr_query(config, solr_host, query_fq)[0]['id']

    # Update descendants entry in Solr
    update_body = [
        {
            "id": doc_id,
            "all_transformations_success_b": {"set": transformation_successes}
        }
    ]

    # Add transformaiton file path fields to descendants entry
    for key, path in transformation_file_paths.items():
        update_body[0][key] = {"set": path}

    r = solr_update(config, solr_host, update_body, r=True)

    if r.status_code != 200:
        print(
            f'Failed to update Solr with descendants information for {dataset_name} on {date}')


def run_using_aws_wrapper(s3, filename):
    try:
        run_using_aws(s3, filename)
    except Exception as e:
        print(e)
        print('Unable to run AWS transformation')


def run_using_aws(s3, filename):
    # =====================================================
    # Set configuration options
    # =====================================================

    with open("grid_transformation_config.yaml", "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    # Must import ecco_cloud_utils as ea, but can't until it is released
    # TODO ask Ian about releasing ecco_cloud_utils
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    dataset_name = config['ds_name']
    source_bucket_name = config['source_bucket']
    target_bucket_name = config['target_bucket']
    output_suffix = config['aws_output_suffix']
    output_dir = f'{dataset_name}_transformed/'
    transformation_version = config['version']

    solr_host = config['solr_host_aws']

    # Query Solr for dataset entry
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    dataset_metadata = solr_query(config, solr_host, fq)[0]

    # Query Solr for harvested entry to get origin_checksum and date
    query_fq = [f'dataset_s:{dataset_name}', 'type_s:harvested',
                f'filename_s:"{filename}"']
    harvested_metadata = solr_query(config, solr_host, query_fq)[0]
    source_file_path = harvested_metadata['pre_transformation_file_path_s']
    origin_checksum = harvested_metadata['checksum_s']
    date = harvested_metadata['date_s']

    if 'hemisphere_s' in harvested_metadata.keys():
        hemi = f'_{harvested_metadata["hemisphere_s"]}'
    else:
        hemi = ''

    transformation_successes = True
    transformation_file_paths = {}

    ###########################################

    source_bucket = s3.Bucket(source_bucket_name)
    target_bucket = s3.Bucket(target_bucket_name)

    # =====================================================
    # Load model grid coordinates (longitude, latitude)
    # =====================================================

    # TODO: grid name business
    grid_name = 'ECCO_llc90_demo'

    # Query Solr for grid metadata
    fq = ['type_s:grid', f'grid_name_s:{grid_name}']
    grid_metadata = solr_query(config, solr_host, fq)[0]

    grid_path = grid_metadata['grid_path_s']
    grid_type = grid_metadata['grid_type_s']
    grid_dir = grid_path.rsplit('/', 2)[0] + '/'

    print(f'======Loading {grid_name} model grid=======')
    dt = np.dtype('>f')

    model_grid = xr.open_dataset(grid_path).reset_coords()

    print("====loading model grid DONE====")

    # =====================================================
    # Make model grid factors if not present locally
    # =====================================================
    grid_factors = f'{grid_name}{hemi}_factors_path_s'
    grid_factors_version = f'{grid_name}{hemi}_factors_version_f'

    if grid_factors_version in dataset_metadata.keys() and transformation_version == dataset_metadata[grid_factors_version]:
        factors_path = dataset_metadata[grid_factors]

        print(f'===Loading {grid_name} factors===')
        xobject = s3.Object(source_bucket_name, factors_path)
        f = xobject.get()['Body'].read()
        factors = np.frombuffer(f, count=-1)

    else:
        print(f'===Creating {grid_name} factors===')

        fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
        short_name = dataset_metadata['short_name_s']

        data_res = config['data_res']

        # If data_res is fractional, convert from string to float
        if type(data_res) is str and '/' in data_res:
            num, den = data_res.replace(' ', '').split('/')
            data_res = float(num) / float(den)

        # Use hemisphere specific variables if data is hemisphere specific
        if hemi:
            hemi_dim = config[f'dims{hemi}']
            hemi_area_extent = config[f'area_extent{hemi}']
            hemi_proj_info = config[f'proj_info{hemi}']
            hemi_data_max_lat = config[f'data_max_lat{hemi}']

            source_grid_min_L, source_grid_max_L, source_grid, \
                data_grid_lons, data_grid_lats = ea.generalized_grid_product(short_name,
                                                                             data_res,
                                                                             hemi_data_max_lat,
                                                                             hemi_area_extent,
                                                                             hemi_dim,
                                                                             hemi_proj_info)
        else:
            source_grid_min_L, source_grid_max_L, source_grid, \
                data_grid_lons, data_grid_lats = ea.generalized_grid_product(short_name,
                                                                             data_res,
                                                                             config['data_max_lat'],
                                                                             config['area_extent'],
                                                                             config['dims'],
                                                                             config['proj_info'])

        # Define the 'swath' as the lats/lon pairs of the model grid
        target_grid = pr.geometry.SwathDefinition(lons=model_grid.XC.values.ravel(),
                                                  lats=model_grid.YC.values.ravel())

        # Retrieve target_grid_radius from model_grid file
        if 'effective_grid_radius' in model_grid:
            target_grid_radius = model_grid.effective_grid_radius.values.ravel()
        elif 'effective_radius' in model_grid:
            target_grid_radius = model_grid.effective_radius.values.ravel()
        elif 'rA' in model_grid:
            target_grid_radius = 0.5*np.sqrt(model_grid.rA.values.ravel())
        else:
            print(f'{grid_name} grid not supported')

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

        print(f'===Saving {grid_name} factors===')
        factors_path_dir = f'{grid_dir}grid_factors/{dataset_name}/'

        # Create directory if needed and save factors
        if not os.path.exists(factors_path_dir):
            os.makedirs(factors_path_dir)

        factors_path = f'{factors_path_dir}{dataset_name}_{grid_name}_factors'

        with open(factors_path, 'wb') as f:
            pickle.dump(factors, f)

        output_filename = f'{dataset_name}_factors/{grid_name}_factors'

        target_bucket.upload_file(factors_path, output_filename)

        print('===Updating Solr with factors===')
        # Query Solr for dataset entry
        query_fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
        doc_id = solr_query(config, solr_host, query_fq)[0]['id']

        aws_factors_path = "s3://" + target_bucket_name + '/' + output_filename

        # Update Solr dataset entry with factors metadata
        update_body = [
            {
                "id": doc_id,
                f'{grid_name}{hemi}_factors_path_s': {"set": f'"{aws_factors_path}"'},
                f'{grid_name}{hemi}_factors_stored_dt': {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                f'{grid_name}{hemi}_factors_version_f': {"set": transformation_version}
            }
        ]

        r = solr_update(config, solr_host, update_body, r=True)

        if r.status_code == 200:
            print('Successfully updated Solr with factors information')
        else:
            print('Failed to update Solr with factors information')

    # =====================================================
    # Load the data
    # =====================================================
    print("=====loading data======")
    print("file being transformed is: ", filename)

    source_bucket.download_file(filename, "/tmp/data.nc")
    ds = xr.open_dataset("/tmp/data.nc", decode_times=True)
    ds.attrs['original_file_name'] = filename

    print("===loading data DONE===")

    query_fq = [f'dataset_s:{dataset_name}', f'type_s:field']
    fields = solr_query(config, solr_host, query_fq)

    for field in fields:
        field_name = field["name_s"]

        # Query if grid/field combination transformation entry exists
        query_fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                    f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']
        docs = solr_query(config, solr_host, query_fq)

        update_body = []
        transform = {}

        # If grid/field combination transformation exists, update transformation status
        # Otherwise initialize new transformation entry
        if len(docs) > 0:
            # Reset status fields
            transform['id'] = docs[0]['id']
            transform['transformation_in_progress_b']: {"set": True}
            transform['success_b']: {"set": False}
            r = solr_update(config, solr_host, transform, r=True)
        else:
            # Initialize new transformation entry
            transform['type_s'] = 'transformation'
            transform['date_s'] = date
            transform['dataset_s'] = dataset_name
            transform['pre_transformation_file_path_s'] = source_file_path
            if hemi:
                transform['hemisphere_s'] = hemi
            transform['origin_checksum_s'] = origin_checksum
            transform['grid_name_s'] = grid_name
            transform['field_s'] = field_name
            transform['transformation_in_progress_b'] = True
            transform['success_b'] = False
            update_body.append(transform)
            r = solr_update(config, solr_host, update_body, r=True)

        if r.status_code != 200:
            print(
                f'Failed to update Solr transformation status for {dataset_name} on {date}')

    # =====================================================
    # Transform/remap data to grid
    # =====================================================
    field_DAs = run_in_any_env(
        model_grid, grid_name, grid_type, fields, factors, ds, date, dataset_metadata, config)

    # =====================================================
    # ### Save the output in the model grid format
    # =====================================================
    print("=========saving output=========")

    # Save each transformed granule for the current field
    for field, (field_DA, success) in zip(fields, field_DAs):
        field_name = field["name_s"]

        basefilename = filename.split('/')[-1]
        if basefilename[:-3] == '.':
            basefilename = basefilename[:-3]
        elif basefilename[:-2] == '.':
            basefilename = basefilename[:-2]

        # Local saving
        output_filename = f'{grid_name}_{field_name}_{basefilename}'
        output_path = f'{output_dir}/{grid_name}/transformed/{field_name}/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_suffix == '.nc':
            field_DS = field_DA.to_dataset()
            field_DS.to_netcdf(f'{output_path}{output_filename}.nc')
            field_DS.close()
        else:
            with open(output_path + output_filename, 'wb') as f:
                pickle.dump(field_DA, f)

        # AWS saving
        # output_filename = output_dir + basefilename + output_suffix
        aws_output_filename = f'{output_dir}{basefilename}{output_suffix}'
        path = f's3://{target_bucket_name}/{aws_output_filename}'

        target_bucket.upload_file(
            output_path + output_filename, aws_output_filename)

        # Query Solr for transformation entry
        query_fq = [f'dataset_s:{dataset_name}', 'type_s:transformation', f'grid_name_s:{grid_name}',
                    f'field_s:{field_name}', f'pre_transformation_file_path_s:"{source_file_path}"']

        docs = solr_query(config, solr_host, query_fq)
        doc_id = solr_query(config, solr_host, query_fq)[0]['id']

        transformation_successes = transformation_successes and success
        transformation_file_paths[f'{grid_name}_{field_name}_transformation_file_path_s'] = path

        # Update Solr transformation entry with file paths and status
        update_body = [
            {
                "id": doc_id,
                "filename_s": {"set": basefilename + output_suffix},
                "transformation_file_path_s": {"set": path},
                "transformation_completed_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                "transformation_in_progress_b": {"set": False},
                "success_b": {"set": success},
                "transformation_checksum_s": {"set": md5(path)},
                "transformation_version_f": {"set": transformation_version}
            }
        ]

        r = solr_update(config, solr_host, update_body, r=True)

        if r.status_code != 200:
            print(
                f'Failed to update Solr transformation entry for {field["name_s"]} in {dataset_name} on {date}')

    print("======saving output DONE=======")

    # Query Solr for descendants entry by date
    query_fq = [f'dataset_s:{dataset_name}',
                'type_s:descendants', f'date_s:{date[:10]}*']
    if hemi:
        query_fq.append(f'hemisphere_s:{hemi[1:]}')

    docs = solr_query(config, solr_host, query_fq)
    doc_id = solr_query(config, solr_host, query_fq)[0]['id']

    # Update descendants entry in Solr
    update_body = [
        {
            "id": doc_id,
            "all_transformations_success_b": {"set": transformation_successes}
        }
    ]

    # Add transformaiton file path fields to descendants entry
    for key, path in transformation_file_paths.items():
        update_body[0][key] = {"set": path}

    r = solr_update(config, solr_host, update_body, r=True)

    if r.status_code != 200:
        print(
            f'Failed to update Solr with descendants information for {dataset_name} on {date}')


def run_in_any_env(model_grid, model_grid_name, model_grid_type, fields, factors, ds, record_date, dataset_metadata, config):
    # =====================================================
    # Code to import ecco utils locally...
    # =====================================================
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[5]}/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    # Check if ends in z and drop it if it does
    if record_date[-1] == 'Z':
        record_date = record_date[:-1]

    array_precision = getattr(np, config['array_precision'])
    record_file_name = ds.attrs['original_file_name']
    extra_information = config['extra_information']
    time_zone_included_with_time = config['time_zone_included_with_time']

    field_DAs = []

    original_dataset_metadata = {
        key: dataset_metadata[key] for key in dataset_metadata.keys() if 'original' in key}

    pre_transformations = config['pre_transformation_steps']
    post_transformations = config['post_transformation_steps']

    if pre_transformations:
        for func_to_run in pre_transformations:
            callable_func = getattr(ea, func_to_run)
            try:
                ds = callable_func(ds)
            except Exception as e:
                print(e)
                print(f'Pre-transformation {func_to_run} failed.')
                return []

    # fields is a list of dictionaries
    for data_field_info in fields:
        try:
            field_DA = ea.generalized_transform_to_model_grid_solr(data_field_info, record_date, model_grid, model_grid_type,
                                                                   array_precision, record_file_name, original_dataset_metadata,
                                                                   extra_information, ds, factors, time_zone_included_with_time,
                                                                   model_grid_name)
            success = True

            if post_transformations:
                for func_to_run in post_transformations:
                    callable_func = getattr(ea, func_to_run)
                    try:
                        field_DA = callable_func(field_DA)
                    except:
                        field_DA = ea.make_empty_record(data_field_info['standard_name_s'], data_field_info['long_name_s'], data_field_info['units_s'],
                                                        record_date, model_grid, model_grid_type, array_precision)
                        success = False
                        break
        except:
            field_DA = ea.make_empty_record(data_field_info['standard_name_s'], data_field_info['long_name_s'], data_field_info['units_s'],
                                            record_date, model_grid, model_grid_type, array_precision)
            success = False

        field_DAs.append((field_DA, success))

    return field_DAs
