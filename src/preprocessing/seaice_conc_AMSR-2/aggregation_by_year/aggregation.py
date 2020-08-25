import os
import json
import yaml
import hashlib
import requests
import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import default_fillvals  # pylint: disable=import-error


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


# Separately saves to disk all lineage entries for each year
def export_lineage(output_dir, years, solr_host, config):
    dataset_name = config['ds_name']

    lineages_dir = f'{output_dir}/{dataset_name}/annual_lineages/'

    if not os.path.exists(lineages_dir):
        os.makedirs(lineages_dir)

    for year in years:
        outfile = f'{lineages_dir}/{dataset_name}_{year}_lineage'

        fq = ['type_s:lineage', f'dataset_s:{dataset_name}', f'date_s:{year}*']
        lineage_docs = solr_query(config, solr_host, fq)

        with open(outfile, 'w') as f:
            resp_out = json.dumps(lineage_docs)
            f.write(resp_out)


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
def run_aggregation(system_path, output_dir, s3=None):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    path_to_yaml = system_path + "/aggregation_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    # =====================================================
    # Code to import ecco utils locally...
    # =====================================================
    from pathlib import Path
    import sys
    generalized_functions_path = Path(config['ecco_utils'])
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea

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

    short_name = dataset_metadata['short_name_s']
    data_time_scale = dataset_metadata['data_time_scale_s']

    # Only aggregate years with updated transformations
    # Based on years_updated_ss field in dataset Solr entry
    if 'years_updated_ss' in dataset_metadata.keys():
        years = dataset_metadata['years_updated_ss']
    else:
        # If no years to aggregate, update dataset Solr entry status
        print('No updated years to aggregate')
        update_body = [
            {
                "id": dataset_metadata['id'],
                "status_s": {"set": 'aggregated'},
            }
        ]

        r = solr_update(config, solr_host, update_body, r=True)

        if r.status_code != 200:
            print(
                f'Failed to update Solr dataset status entry for {dataset_name}')
        return

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

            # Iterate through dataset fields
            for field in fields:
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

                    # If transformed file is present for date, grid, and field combination
                    # open the file, otherwise make empty record

                    # Combines hemisphere data into one file, if data is in hemisphere format
                    if docs:
                        opened_datasets = []

                        for doc in docs:
                            # if running on AWS, get file from s3
                            if doc['transformation_file_path_s'][:5] == 's3://':
                                source_bucket_name, key_name = split_s3_bucket_key(
                                    doc['transformation_file_path_s'])
                                obj = s3.Object(source_bucket_name, key_name)
                                data_DA = xr.open_dataarray(
                                    obj, decode_times=True)
                                # f = obj.get()['Body'].read()
                                # data = np.frombuffer(f, dtype=dt, count=-1)
                            else:
                                data_DA = xr.open_dataarray(
                                    doc['transformation_file_path_s'], decode_times=True)

                            opened_datasets.append(data_DA)

                        # If there are more than one files for this grid/field/date combination (implies hemisphered data),
                        # combine hemispheres on nonempty datafile, if present.
                        if len(opened_datasets) == 2:
                            if ~np.isnan(opened_datasets[0].values).all():
                                data_DA = opened_datasets[0].copy()
                                data_DA.values = np.where(np.isnan(data_DA.values), opened_datasets[1].values, data_DA.values)
                            else:
                                data_DA = opened_datasets[1].copy()
                                data_DA.values = np.where(np.isnan(data_DA.values), opened_datasets[0].values, data_DA.values)
                        else:
                            data_DA = opened_datasets[0]
                    else:
                        data_DA = ea.make_empty_record(field['standard_name_s'], field['long_name_s'], field['units_s'],
                                                       date, model_grid, grid_type, array_precision)

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
                shortest_filename = f'{short_name}_{grid_name}_{data_time_scale.upper()}_{year}_{field_name}'
                monthly_filename = f'{short_name}_{grid_name}_MONTHLY_{year}_{field_name}'

                output_filenames = {'shortest': shortest_filename,
                                    'monthly': monthly_filename}

                output_path = f'{output_dir}{dataset_name}/{grid_name}/aggregated/{field_name}/'

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

                print(daily_DA_year_merged)

                try:
                    # Performs the aggreagtion of the yearly data, and saves it
                    ea.generalized_aggregate_and_save(daily_DA_year_merged,
                                                      new_data_attr,
                                                      config['do_monthly_aggregation'],
                                                      int(year),
                                                      config['skipna_in_mean'],
                                                      output_filenames,
                                                      fill_values,
                                                      output_dirs,
                                                      binary_dtype,
                                                      grid_type,
                                                      save_binary=config['save_binary'],
                                                      save_netcdf=config['save_netcdf'],
                                                      remove_nan_days_from_data=config['remove_nan_days_from_data'])

                    success = True
                except Exception as e:
                    print(e)
                    success = False
                    output_filepaths = {'daily_bin': '',
                                        'daily_netCDF': '',
                                        'monthly_bin': '',
                                        'monthly_netCDF': ''}

                aggregation_successes = aggregation_successes and success

                # Upload files to s3
                if s3:
                    target_bucket_name = config['target_bucket_name']
                    s3_aggregated_path = config['s3_aggregated_path']
                    s3_output_dir = f'{s3_aggregated_path}{dataset_name}_transformed_by_year'
                    target_bucket = s3.Bucket(target_bucket_name)

                    # Upload shortest and monthly aggregated files to target bucket
                    for filename in output_filenames:
                        target_bucket.upload_file(bin_output_dir, filename)

                    s3_path = f's3://{target_bucket_name}/{s3_output_dir}'

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
                            "aggregation_version_s": {"set": config['version']}
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
                        update_body[0]['s3_path'] = s3_path

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
                            "aggregation_version_s": config['version']
                        }
                    ]

                    # Update file paths according to the data time scale and do monthly aggregation config field
                    if (data_time_scale == 'daily') and (config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = output_filepaths['daily_bin']
                        update_body[0]["aggregated_daily_netCDF_path_s"] = output_filepaths['daily_netCDF']
                        update_body[0]["aggregated_monthly_bin_path_s"] = output_filepaths['monthly_bin']
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = output_filepaths['monthly_netCDF']
                    elif (data_time_scale == 'daily') and (not config['do_monthly_aggregation']):
                        update_body[0]["aggregated_daily_bin_path_s"] = output_filepaths['daily_bin']
                        update_body[0]["aggregated_daily_netCDF_path_s"] = output_filepaths['daily_netCDF']
                    elif data_time_scale == 'monthly':
                        update_body[0]["aggregated_monthly_bin_path_s"] = output_filepaths['monthly_bin']
                        update_body[0]["aggregated_monthly_netCDF_path_s"] = output_filepaths['monthly_netCDF']

                    if s3:
                        update_body[0]['s3_path'] = s3_path

                r = solr_update(config, solr_host, update_body, r=True)

                if r.status_code != 200:
                    print(
                        f'Failed to update Solr aggregation entry for {field_name} in {dataset_name} for {year} and grid {grid_name}')

                # Query for lineage entries from this year
                fq = ['type_s:lineage',
                      f'dataset_s:{dataset_name}', f'date_s:{year}*']
                existing_lineage_docs = solr_query(config, solr_host, fq)

                # if lineage entries already exist, update them
                if len(existing_lineage_docs) > 0:
                    for doc in existing_lineage_docs:
                        doc_id = doc['id']

                        update_body = [
                            {
                                "id": doc_id,
                                "all_aggregation_success_b": {"set": aggregation_successes}
                            }
                        ]

                        # Add aggregation file path fields to lineage entry
                        for key, value in output_filepaths.items():
                            update_body[0][f'{grid_name}_{field_name}_aggregated_{key}_path_s'] = {
                                "set": value}

                        r = solr_update(config, solr_host, update_body, r=True)

                        if r.status_code != 200:
                            print(
                                f'Failed to update Solr aggregation entry for {field_name} in {dataset_name} for {year} and grid {grid_name}')

    # Update Solr dataset entry status and years_updated to empty
    update_body = [
        {
            "id": dataset_metadata['id'],
            "status_s": {"set": 'aggregated'},
            "years_updated_ss": {"set": []}}
    ]

    r = solr_update(config, solr_host, update_body, r=True)

    if r.status_code != 200:
        print(
            f'Failed to update Solr dataset entry with aggregation information for {dataset_name}')

    # Export annual lineage JSON file for each year updated
    print("=========exporting data lineage=========")
    export_lineage(output_dir, years, solr_host, config)
    print("=========exporting data lineage DONE=========")
