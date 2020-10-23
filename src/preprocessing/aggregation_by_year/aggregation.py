import os
import sys
import json
import yaml
import hashlib
import requests
import numpy as np
import xarray as xr
from pathlib import Path
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module
from datetime import datetime, timedelta


np.warnings.filterwarnings('ignore')


def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)


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


# Aggregates data into annual files, saves them, and updates Solr
def run_aggregation(output_dir, s3=None, config_path=''):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    if not config_path:
        print('No path for configuration file. Can not run aggregation.')
        return

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    # =====================================================
    # Code to import ecco utils locally...
    # =====================================================
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[4]}/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    # =====================================================
    # Set configuration options and Solr metadata
    # =====================================================
    dataset_name = config['ds_name']

    if s3:
        solr_host = config['solr_host_aws']
    else:
        solr_host = config['solr_host_local']

    fq = ['type_s:grid']
    grids = [grid for grid in solr_query(config, solr_host, fq)]

    fq = ['type_s:field', f'dataset_s:{dataset_name}']
    fields = solr_query(config, solr_host, fq)

    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_metadata = solr_query(config, solr_host, fq)[0]

    aggregate_all_years = False
    aggregation_version = str(config['version'])
    if 'aggregation_version_s' in dataset_metadata.keys():
        existing_aggregation_version = dataset_metadata['aggregation_version_s']
        if existing_aggregation_version != aggregation_version:
            aggregate_all_years = True

    data_time_scale = dataset_metadata['data_time_scale_s']

    # Define precision of output files, float32 is standard
    array_precision = getattr(np, config['array_precision'])

    # Define fill values for binary and netcdf
    if array_precision == np.float32:
        binary_dtype = '>f4'
        netcdf_fill_value = default_fillvals['f4']

    elif array_precision == np.float64:
        binary_dtype = '>f8'
        netcdf_fill_value = default_fillvals['f8']

    fill_values = {'binary': -9999, 'netcdf': netcdf_fill_value}

    update_body = []

    aggregation_successes = True

    # Iterate through grids
    for grid in grids:

        grid_path = grid['grid_path_s']
        grid_name = grid['grid_name_s']
        grid_type = grid['grid_type_s']

        # Only aggregate years with updated transformations
        # Based on years_updated_ss field in dataset Solr entry
        solr_years_updated = f'{grid_name}_years_updated_ss'
        if not aggregate_all_years and solr_years_updated in dataset_metadata.keys():
            years = dataset_metadata[solr_years_updated]
        elif aggregate_all_years:
            start_year = int(dataset_metadata['start_date_dt'][:4])
            end_year = int(dataset_metadata['end_date_dt'][:4])
            years = [str(year) for year in range(start_year, end_year + 1)]
        else:
            # If no years to aggregate for this grid, continue to next grid
            print(f'No updated years to aggregate for {grid_name}')
            continue

        if grid_path[:5] == 's3://':
            source_bucket_name, key_name = split_s3_bucket_key(grid_path)
            obj = s3.Object(source_bucket_name, key_name)
            # TODO: not sure if xarray can open s3 obj
            model_grid = xr.open_dataset(obj, decode_times=True)
        else:
            model_grid = xr.open_dataset(grid_path, decode_times=True)

        # Iterate through years
        for year in years:

            # Construct list of dates corresponding to data time scale
            if data_time_scale == 'daily':
                dates_in_year = np.arange(
                    f'{year}-01-01', f'{int(year)+1}-01-01', dtype='datetime64[D]')

            elif data_time_scale == 'monthly':
                dates_in_year = np.arange(
                    f'{year}-01', f'{int(year)+1}-01', dtype='datetime64[M]')
                dates_in_year = [f'{date}-01' for date in dates_in_year]

            # Iterate through dataset fields
            for field in fields:
                json_output = {}
                transformations = []
                json_output['dataset'] = dataset_metadata

                field_name = field['name_s']

                print(
                    f'===initializing {str(year)}_{grid_name}_{field_name}===')

                print("===looping through all files===")
                daily_DA_year = []

                for date in dates_in_year:
                    # Query for date
                    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
                          f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{date}*']

                    docs = solr_query(config, solr_host, fq)

                    # If first of month is not found, query with 7 day tolerance only for monthly data
                    if not docs and data_time_scale == 'monthly':
                        if config['monthly_tolerance']:
                            tolerance = int(config['monthly_tolerance'])
                        else:
                            tolerance = 8
                        start_month_date = datetime.strptime(date, '%Y-%m-%d')
                        tolerance_days = []

                        for i in range(1, tolerance):
                            plus_date = start_month_date + timedelta(days=i)
                            neg_date = start_month_date - timedelta(days=i)

                            tolerance_days.append(
                                datetime.strftime(plus_date, '%Y-%m-%d'))
                            tolerance_days.append(
                                datetime.strftime(neg_date, '%Y-%m-%d'))

                        for tol_date in tolerance_days:
                            fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
                                  f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{tol_date}*']
                            docs = solr_query(config, solr_host, fq)

                            if docs:
                                break

                    # If transformed file is present for date, grid, and field combination
                    # open the file, otherwise make empty record
                    opened_datasets = []
                    for doc in docs:
                        # if running on AWS, get file from s3
                        if doc['transformation_file_path_s'][:5] == 's3://':
                            source_bucket_name, key_name = split_s3_bucket_key(
                                doc['transformation_file_path_s'])
                            obj = s3.Object(source_bucket_name, key_name)
                            data_DA = xr.open_dataarray(obj, decode_times=True)
                            # f = obj.get()['Body'].read()
                            # data = np.frombuffer(f, dtype=dt, count=-1)
                        else:
                            data_DA = xr.open_dataarray(
                                doc['transformation_file_path_s'], decode_times=True)

                        opened_datasets.append(data_DA)

                        # Update JSON transformations list
                        fq = [f'dataset_s:{dataset_name}', 'type_s:harvested',
                              f'pre_transformation_file_path_s:"{doc["pre_transformation_file_path_s"]}"']
                        harvested_metadata = solr_query(config, solr_host, fq)

                        transformation_metadata = doc
                        transformation_metadata['harvested'] = harvested_metadata
                        transformations.append(transformation_metadata)

                    # If there are more than one files for this grid/field/date combination (implies hemisphered data),
                    # combine hemispheres on nonempty datafile, if present.
                    if len(opened_datasets) == 2:
                        if ~np.isnan(opened_datasets[0].values).all():
                            data_DA = opened_datasets[0].copy()
                            data_DA.values = np.where(
                                np.isnan(data_DA.values), opened_datasets[1].values, data_DA.values)
                        else:
                            data_DA = opened_datasets[1].copy()
                            data_DA.values = np.where(
                                np.isnan(data_DA.values), opened_datasets[0].values, data_DA.values)
                    elif len(opened_datasets) == 1:
                        data_DA = opened_datasets[0]
                    else:
                        data_DA = ea.make_empty_record(
                            field['standard_name_s'], field['long_name_s'], field['units_s'], date, model_grid, grid_type, array_precision)
                    # Append each day's data to annual list
                    daily_DA_year.append(data_DA)

                # Concatenate all data files within annual list
                daily_DA_year_merged = xr.concat((daily_DA_year), dim='time')

                # Create metadata fields for aggregated data file
                new_data_attr = {}
                new_data_attr['original_dataset_title'] = dataset_metadata['original_dataset_title_s']
                new_data_attr['original_dataset_short_name'] = dataset_metadata['original_dataset_short_name_s']
                new_data_attr['original_dataset_url'] = dataset_metadata['original_dataset_url_s']
                new_data_attr['original_dataset_reference'] = dataset_metadata['original_dataset_reference_s']
                new_data_attr['original_dataset_doi'] = dataset_metadata['original_dataset_doi_s']
                new_data_attr['new_name'] = f'{field_name}_interpolated_to_{grid_name}'
                new_data_attr['interpolated_grid_id'] = grid_name

                # Create filenames based on date time scale
                # If data time scale is monthly, shortest_filename is monthly
                shortest_filename = f'{dataset_name}_{grid_name}_{data_time_scale.upper()}_{field_name}_{year}'
                monthly_filename = f'{dataset_name}_{grid_name}_MONTHLY_{field_name}_{year}'

                output_filenames = {'shortest': shortest_filename,
                                    'monthly': monthly_filename}

                output_path = f'{output_dir}{dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/'

                bin_output_dir = output_path + 'bin/'

                if not os.path.exists(bin_output_dir):
                    os.makedirs(bin_output_dir)

                netCDF_output_dir = output_path + 'netCDF/'

                if not os.path.exists(netCDF_output_dir):
                    os.makedirs(netCDF_output_dir)

                # generalized_aggregate_and_save expects Paths
                output_dirs = {'binary': Path(bin_output_dir),
                               'netcdf': Path(netCDF_output_dir)}

                output_filepaths = {'daily_bin': f'{output_path}bin/{shortest_filename}',
                                    'daily_netCDF': f'{output_path}netCDF/{shortest_filename}.nc',
                                    'monthly_bin': f'{output_path}bin/{monthly_filename}',
                                    'monthly_netCDF': f'{output_path}netCDF/{monthly_filename}.nc'}

                # print(daily_DA_year_merged)

                try:
                    # Performs the aggreagtion of the yearly data, and saves it
                    empty_year = ea.generalized_aggregate_and_save(daily_DA_year_merged,
                                                                   new_data_attr,
                                                                   config['do_monthly_aggregation'],
                                                                   int(year),
                                                                   config['skipna_in_mean'],
                                                                   output_filenames,
                                                                   fill_values,
                                                                   output_dirs,
                                                                   binary_dtype,
                                                                   grid_type,
                                                                   on_aws=s3,
                                                                   save_binary=config['save_binary'],
                                                                   save_netcdf=config['save_netcdf'],
                                                                   remove_nan_days_from_data=config['remove_nan_days_from_data'])

                    # Upload files to s3
                    if s3:
                        target_bucket_name = config['target_bucket_name']
                        s3_aggregated_path = config['s3_aggregated_path']
                        s3_output_dir = f'{s3_aggregated_path}{dataset_name}_transformed_by_year'
                        target_bucket = s3.Bucket(target_bucket_name)

                        if config['do_monthly_aggregation']:
                            target_bucket.upload_file(
                                output_filepaths['daily_bin'], output_filenames['shortest'])
                            if data_time_scale.upper() != 'MONTHLY':
                                target_bucket.upload_file(
                                    output_filepaths['monthly_bin'], output_filenames['monthly'])
                        else:
                            target_bucket.upload_file(
                                output_filepaths['daily_bin'], output_filenames['shortest'])

                        s3_path = f's3://{target_bucket_name}/{s3_output_dir}'

                    success = True

                except Exception as e:
                    print(e)
                    empty_year = True
                    success = False
                    output_filepaths = {'daily_bin': '',
                                        'daily_netCDF': '',
                                        'monthly_bin': '',
                                        'monthly_netCDF': ''}

                aggregation_successes = aggregation_successes and success
                empty_year = empty_year and success

                if empty_year:
                    output_filepaths = {'daily_bin': '',
                                        'daily_netCDF': '',
                                        'monthly_bin': '',
                                        'monthly_netCDF': ''}

                # Query Solr for existing aggregation
                fq = [f'dataset_s:{dataset_name}', 'type_s:aggregation',
                      f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
                docs = solr_query(config, solr_host, fq)

                # If aggregation exists, update using Solr entry id
                if len(docs) > 0:
                    doc_id = docs[0]['id']
                    update_body = [
                        {
                            "id": doc_id,
                            "aggregation_time_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                            "aggregation_version_s": {"set": aggregation_version}
                        }
                    ]

                    # Update file paths according to the data time scale and do monthly aggregation config field
                    if (data_time_scale == 'daily') and (config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = {
                            "set": output_filepaths['daily_bin']}
                        update_body[0]["aggregated_daily_netCDF_path_s"] = {
                            "set": output_filepaths['daily_netCDF']}
                        update_body[0]["aggregated_monthly_bin_path_s"] = {
                            "set": output_filepaths['monthly_bin']}
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = {
                            "set": output_filepaths['monthly_netCDF']}
                    elif (data_time_scale == 'daily') and not (config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = {
                            "set": output_filepaths['daily_bin']}
                        update_body[0]["aggregated_daily_netCDF_path_s"] = {
                            "set": output_filepaths['daily_netCDF']}
                    elif data_time_scale == 'monthly':
                        update_body[0]["aggregated_monthly_bin_path_s"] = {
                            "set": output_filepaths['monthly_bin']}
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = {
                            "set": output_filepaths['monthly_netCDF']}

                    if s3:
                        update_body[0]['s3_path_s'] = {"set": s3_path}

                    if empty_year:
                        update_body[0]["notes_s"] = {
                            "set": 'Empty year (no data present in grid), not saving to disk.'}
                    else:
                        update_body[0]["notes_s"] = {"set": ''}

                else:
                    # Create new aggregation entry if it doesn't exist
                    update_body = [
                        {
                            "type_s": 'aggregation',
                            "dataset_s": dataset_name,
                            "year_s": year,
                            "grid_name_s": grid_name,
                            "field_s": field_name,
                            "aggregation_time_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "aggregation_success_b": success,
                            "aggregation_version_s": aggregation_version
                        }
                    ]

                    # Update file paths according to the data time scale and do monthly aggregation config field
                    if (data_time_scale == 'daily') and (config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = {
                            "set": output_filepaths['daily_bin']}
                        update_body[0]["aggregated_daily_netCDF_path_s"] = {
                            "set": output_filepaths['daily_netCDF']}
                        update_body[0]["aggregated_monthly_bin_path_s"] = {
                            "set": output_filepaths['monthly_bin']}
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = {
                            "set": output_filepaths['monthly_netCDF']}
                    elif (data_time_scale == 'daily') and (not config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = {
                            "set": output_filepaths['daily_bin']}
                        update_body[0]["aggregated_daily_netCDF_path_s"] = {
                            "set": output_filepaths['daily_netCDF']}
                    elif data_time_scale == 'monthly':
                        update_body[0]["aggregated_monthly_bin_path_s"] = {
                            "set": output_filepaths['monthly_bin']}
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = {
                            "set": output_filepaths['monthly_netCDF']}

                    if s3:
                        update_body[0]['s3_path'] = s3_path

                    if empty_year:
                        update_body[0]["notes_s"] = {
                            "set": 'Empty year (no data present in grid), not saving to disk.'}
                    else:
                        update_body[0]["notes_s"] = {"set": ''}

                r = solr_update(config, solr_host, update_body, r=True)

                if r.status_code != 200:
                    print(
                        f'Failed to update Solr aggregation entry for {field_name} in {dataset_name} for {year} and grid {grid_name}')

                # Query for descendants entries from this year
                fq = ['type_s:descendants',
                      f'dataset_s:{dataset_name}', f'date_s:{year}*']
                existing_descendants_docs = solr_query(config, solr_host, fq)

                # if descendants entries already exist, update them
                if len(existing_descendants_docs) > 0:
                    for doc in existing_descendants_docs:
                        doc_id = doc['id']

                        update_body = [
                            {
                                "id": doc_id,
                                "all_aggregation_success_b": {"set": aggregation_successes}
                            }
                        ]

                        # Add aggregation file path fields to descendants entry
                        for key, value in output_filepaths.items():
                            update_body[0][f'{grid_name}_{field_name}_aggregated_{key}_path_s'] = {
                                "set": value}

                        r = solr_update(config, solr_host, update_body, r=True)

                        if r.status_code != 200:
                            print(
                                f'Failed to update Solr aggregation entry for {field_name} in {dataset_name} for {year} and grid {grid_name}')

                fq = [f'dataset_s:{dataset_name}', 'type_s:aggregation',
                      f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
                docs = solr_query(config, solr_host, fq)

                # Export annual descendants JSON file for each aggregation created
                print("=========exporting data descendants=========")
                json_output['aggregation'] = docs
                json_output['transformations'] = transformations
                json_output_path = f'{output_dir}{dataset_name}/transformed_products/{grid_name}/aggregated/{field_name}/{dataset_name}_{field_name}_{year}_descendants'
                with open(json_output_path, 'w') as f:
                    resp_out = json.dumps(json_output, indent=4)
                    f.write(resp_out)
                print("=========exporting data descendants DONE=========")

    # Update Solr dataset entry status and years_updated to empty
    update_body = [
        {
            "id": dataset_metadata['id'],
            "aggregation_version_s": {"set": aggregation_version},
            "status_s": {"set": 'aggregated'}
        }
    ]

    for grid in grids:
        solr_years_updated = f'{grid_name}_years_updated_ss'
        update_body[0][solr_years_updated] = {"set": []}

    r = solr_update(config, solr_host, update_body, r=True)

    if r.status_code != 200:
        print(
            f'Failed to update Solr dataset entry with aggregation information for {dataset_name}')
