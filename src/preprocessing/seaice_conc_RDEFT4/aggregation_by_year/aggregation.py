import numpy as np
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=import-error
from datetime import datetime
import json
import yaml
import requests
import os
import hashlib
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


def export_lineage(outfile, config):
    host = config['solr_host']
    collection_name = config['solr_collection_name']
    dataset = config['ds_name']

    url = host + collection_name + \
        '/select?q=dataset_name_s%3A{dataset}%20OR%20dataset_s%3A{dataset}&rows=30000'
    url = url.format(dataset=dataset)

    r = requests.get(url=url)
    resp = json.loads(r.content)

    with open(outfile, 'w') as f:
        resp_out = json.dumps(resp)
        f.write(resp_out)


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


def run_aggregation(system_path, output_dir, s3=None):
    #
    # Code to import ecco utils locally... #
    # NOTE: assumes /src/preprocessing/ECCO-ACCESS
    from pathlib import Path
    import sys

    p = Path(__file__).parents[2]
    generalized_functions_path = Path(
        f'{p}/ecco-access/ECCO-ACCESS/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea
    # NOTE: generalized functions added to ecco_cloud_utils __init__.py
    # import generalized_functions as gf
    # END Code to import ecco utils locally... #
    #

    path_to_yaml = system_path + "/aggregation_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    dataset = config['ds_name']

    fq = ['type_s:grid']
    grids = [grid for grid in solr_query(config, fq)]

    fq = ['type_s:field', f'dataset_s:{dataset}']
    fields = solr_query(config, fq)

    fq = ['type_s:dataset', f'dataset_s:{dataset}']
    dataset_metadata = solr_query(config, fq)[0]

    short_name = dataset_metadata['short_name_s']

    years = ['2010', '2011']

    if 'years_updated_ss' in dataset_metadata.keys():
        years = dataset_metadata['years_updated_ss']
    else:
        print('No updated years to aggregate')
        update_body = [
            {
                "id": dataset_metadata['id'],
                "status_s": {"set": 'aggregated'},
            }
        ]

        solr_update(config, update_body)
        return

    data_time_scale = dataset_metadata['data_time_scale_s']

    # Define precision of output files, float32 is standard
    # ------------------------------------------------------
    array_precision = getattr(np, config['array_precision'])

    # Define fill values for binary and netcdf
    # ---------------------------------------------
    if array_precision == np.float32:
        binary_dtype = '>f4'
        netcdf_fill_value = default_fillvals['f4']

    elif array_precision == np.float64:
        binary_dtype = '>f8'
        netcdf_fill_value = default_fillvals['f8']

    fill_values = {'binary': -9999, 'netcdf': netcdf_fill_value}

    update_body = []

    if years[0] != '':

        for grid in grids:
            grid_path = grid['grid_path_s']
            grid_name = grid['grid_name_s']
            grid_type = grid['grid_type_s']

            model_grid = xr.open_dataset(grid_path, decode_times=True)

            for year in years:

                if data_time_scale == 'daily':
                    dates_in_year = np.arange(
                        f'{year}-01-01', f'{int(year)+1}-01-01', dtype='datetime64[D]')
                elif data_time_scale == 'monthly':
                    months_in_year = np.arange(
                        f'{year}-01', f'{int(year)+1}-01', dtype='datetime64[M]')
                    dates_in_year = []
                    for month in months_in_year:
                        dates_in_year.append(f'{month}')

                for field in fields:
                    field_name = field['name_s']
                    print("======initalizing yr " + str(year) +
                          "_" + grid_name + "_"+field_name+"======")

                    print("===looping through all files===")
                    daily_DA_year = []

                    print("===creating empty records for missing days===")
                    for date in dates_in_year:
                        print('dates_in_year: ', date)
                        # Query for date
                        fq = [f'dataset_s:{dataset}', 'type_s:transformation',
                              f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'date_s:{date}*']

                        docs = solr_query(config, fq)

                        if docs:
                            if docs[0]['transformation_file_path_s'][:5] == 's3://':
                                source_bucket_name, key_name = split_s3_bucket_key(
                                    docs[0]['transformation_file_path_s'])
                                obj = s3.Object(source_bucket_name, key_name)
                                data_DA = xr.open_dataarray(
                                    obj, decode_times=True)
                                # f = obj.get()['Body'].read()
                                # data = np.frombuffer(f, dtype=dt, count=-1)
                            else:
                                data_DA = xr.open_dataarray(
                                    docs[0]['transformation_file_path_s'], decode_times=True)

                            print('data DA: ', data_DA)

                        else:
                            data_DA = ea.make_empty_record(field['standard_name_s'], field['long_name_s'], field['units_s'],
                                                           date, model_grid, grid_type, array_precision)
                            print('empty DA: ', data_DA)

                        daily_DA_year.append(data_DA)

                    daily_DA_year_merged = xr.concat(
                        (daily_DA_year), dim='time')

                    new_data_attr = {}
                    new_data_attr['original_dataset_title'] = dataset_metadata['original_dataset_title_s']
                    new_data_attr['original_dataset_short_name'] = dataset_metadata['original_dataset_short_name_s']
                    new_data_attr['original_dataset_url'] = dataset_metadata['original_dataset_url_s']
                    new_data_attr['original_dataset_reference'] = dataset_metadata['original_dataset_reference_s']
                    new_data_attr['original_dataset_doi'] = dataset_metadata['original_dataset_doi_s']
                    new_data_attr['new_name'] = f'{field_name}_interpolated_to_{grid_name}'
                    new_data_attr['interpolated_grid_id'] = grid_name

                    shortest_filename = f'{short_name}_{grid_name}_DAILY_{year}_{field_name}'
                    monthly_filename = f'{short_name}_{grid_name}_MONTHLY_{year}_{field_name}'

                    output_filenames = {'shortest': monthly_filename,
                                        'monthly': monthly_filename}

                    output_path = output_dir + dataset + '/' + \
                        grid_name + '/aggregated/' + field_name + '/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    bin_output_dir = output_path + 'bin/'
                    if not os.path.exists(bin_output_dir):
                        os.mkdir(bin_output_dir)

                    netCDF_output_dir = output_path + 'netCDF/'
                    if not os.path.exists(netCDF_output_dir):
                        os.mkdir(netCDF_output_dir)

                    # generalized_aggregate_and_save expects Paths
                    output_dirs = {'binary': Path(bin_output_dir),
                                   'netcdf': Path(netCDF_output_dir)}

                    print(daily_DA_year_merged)

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

                    # Upload files to s3
                    if s3:
                        target_bucket_name = config['target_bucket_name']
                        s3_aggregated_path = config['s3_aggregated_path']
                        s3_output_dir = s3_aggregated_path + \
                            config['ds_name'] + '_transformed_by_year'
                        target_bucket = s3.Bucket(target_bucket_name)

                        # Upload shortest and monthly aggregated files to target bucket
                        for filename in output_filenames:
                            target_bucket.upload_file(
                                bin_output_dir, filename)

                        s3_path = "s3://" + target_bucket_name + '/' + s3_output_dir

                    # Check if aggregation already exists
                    fq = [f'dataset_s:{dataset}', 'type_s:aggregation',
                          f'grid_name_s:{grid_name}', f'field_s:{field_name}', f'year_s:{year}']
                    docs = solr_query(config, fq)

                    if len(docs) > 0:
                        doc_id = docs[0]['id']
                        update_body = [
                            {
                                "id": doc_id,
                                "aggregation_time_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                                "aggregated_daily_bin_path_s": {"set": output_path + 'bin/' + shortest_filename},
                                "aggregated_monthly_bin_path_s": {"set": output_path + 'bin/' + monthly_filename},
                                "aggregated_daily_netCDF_path_s": {"set": output_path + 'netCDF/' + shortest_filename},
                                "aggregated_monthly_netCDF_path_s": {"set": output_path + 'netCDF/' + monthly_filename},
                                "aggregation_version_s": {"set": config['version']}
                            }
                        ]

                        if s3:
                            update_body[0]['s3_path'] = s3_path

                        solr_update(config, update_body)

                    else:
                        update_body = [
                            {
                                "type_s": 'aggregation',
                                "dataset_s": dataset,
                                "year_s": year,
                                "grid_name_s": grid_name,
                                "field_s": field_name,
                                "aggregation_time_dt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "aggregation_success_b": True,
                                "aggregated_daily_bin_path_s": output_path + 'bin/' + shortest_filename,
                                "aggregated_monthly_bin_path_s": output_path + 'bin/' + monthly_filename,
                                "aggregated_daily_netCDF_path_s": output_path + 'netCDF/' + shortest_filename,
                                "aggregated_monthly_netCDF_path_s": output_path + 'netCDF/' + monthly_filename,
                                "aggregation_version_s": config['version'],
                            }
                        ]

                        if s3:
                            update_body[0]['s3_path'] = s3_path

                        solr_update(config, update_body)

        # Clear out years updated in dataset level Solr object
        update_body = [
            {
                "id": dataset_metadata['id'],
                "status_s": {"set": 'aggregated'},
                "years_updated_ss": {"set": []}}
        ]

        solr_update(config, update_body)

        print("=========exporting data lineage=========")
        export_lineage(f'{output_dir}/{dataset}/{dataset}_lineage', config)

        print("=========exporting data lineage DONE=========")
    else:
        print("=========No new files to aggregate=========")
