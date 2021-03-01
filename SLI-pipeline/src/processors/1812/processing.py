import os
import sys
import yaml
import requests
import hashlib
import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, date
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module


def md5(fname):
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, solr_host, fq, solr_collection_name):
    """
    Queries Solr database using the filter query passed in.
    Returns list of Solr entries that satisfies the query.
    """

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000,
               'sort': 'date_s asc'}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, solr_host, update_body, solr_collection_name, r=False):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def processing(config_path='', output_path='', solr_info=''):

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    version = config['version']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    date_regex = '%Y-%m-%dT%H:%M:%S'

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, solr_host, fq, solr_collection_name)

    cycles = {}

    if solr_cycles:
        for cycle in solr_cycles:
            cycles[cycle['start_date_s']] = cycle

    # Generate list of cycle date tuples (start, end)
    # Dataset ends at roughly 2019-02-01
    cycle_dates = []
    start_date = datetime.strptime('1992-01-01T00:00:00', date_regex)
    end_date = datetime.strptime('2020-01-01T00:00:00', date_regex)
    delta = timedelta(days=10)
    curr = start_date
    while curr < end_date:
        cycle_dates.append((curr, curr + delta))
        curr += delta

    var = 'SLA'

    for (start_date, end_date) in cycle_dates:

        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)

        query_start = datetime.strftime(start_date, '%Y-%m-%dT%H:%M:%SZ')
        query_end = datetime.strftime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
              f'date_s:{{{query_start} TO {query_end}]']

        cycle_granules = solr_query(
            config, solr_host, fq, solr_collection_name)

        if not cycle_granules:
            print(f'No granules for cycle {start_date_str} to {end_date_str}')
            continue

        updating = False

        # If any single granule in a cycle satisfies any of the following conditions:
        # - has been updated,
        # - previously failed,
        # - has a different version than what is in the config
        # reaggregate the entire cycle
        if cycles:
            if start_date_str in cycles.keys():
                existing_cycle = cycles[start_date_str]
                prior_time = existing_cycle['aggregation_time_s']
                prior_success = existing_cycle['aggregation_success_b']
                prior_version = existing_cycle['aggregation_version_f']

                if not prior_success or prior_version != version:
                    updating = True

                for granule in cycle_granules:
                    if prior_time < granule['modified_time_dt']:
                        updating = True
                        continue
            else:
                updating = True
        else:
            updating = True

        if updating:
            aggregation_success = False
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            granules = []

            overall_center_time = start_date + ((end_date - start_date)/2)
            overall_center_time_str = datetime.strftime(
                overall_center_time, '%Y-%m-%dT%H:%M:%S')
            units_time = datetime.strftime(
                overall_center_time, "%Y-%m-%d %H:%M:%S")

            for granule in cycle_granules:
                ds = xr.open_dataset(granule['granule_file_path_s'])

                # Rename var to 'SSHA'
                ds = ds.rename({var: 'SSHA'})

                granules.append(ds)

            try:
                # Merge opened granules
                if len(granules) > 1:

                    merged_cycle_ds = xr.concat((granules), dim='Time')

                else:
                    merged_cycle_ds = granules[0]

                merged_cycle_ds.attrs = {}
                merged_cycle_ds.attrs['title'] = 'Sea Level Anormaly Estimate based on Altimeter Data'

                merged_cycle_ds.attrs['cycle_start'] = start_date_str
                merged_cycle_ds.attrs['cycle_center'] = overall_center_time_str
                merged_cycle_ds.attrs['cycle_end'] = end_date_str

                data_time_start = merged_cycle_ds.Time_bounds.values[0][0]
                data_time_end = merged_cycle_ds.Time_bounds.values[-1][1]
                data_time_center = data_time_start + \
                    ((data_time_end - data_time_start)/2)

                merged_cycle_ds.attrs['data_time_start'] = np.datetime_as_string(
                    data_time_start, unit='s')
                merged_cycle_ds.attrs['data_time_center'] = np.datetime_as_string(
                    data_time_center, unit='s')
                merged_cycle_ds.attrs['data_time_end'] = np.datetime_as_string(
                    data_time_end, unit='s')

                # Center time
                filename_time = datetime.strftime(
                    overall_center_time, '%Y%m%dT%H%M%S')
                filename = f'sla_{filename_time}.nc'

                # Var Attributes
                merged_cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(
                    merged_cycle_ds['SSHA'].values)
                merged_cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(
                    merged_cycle_ds['SSHA'].values)

                encoding_each = {'zlib': True,
                                 'complevel': 5,
                                 'dtype': 'float32',
                                 'shuffle': True,
                                 '_FillValue': default_fillvals['f8']}

                coord_encoding = {}
                for coord in merged_cycle_ds.coords:
                    coord_encoding[coord] = {'_FillValue': None,
                                             'dtype': 'float32',
                                             'complevel': 6}

                    if 'SSHA' in coord:
                        coord_encoding[coord] = {
                            '_FillValue': default_fillvals['f8']}

                    if 'Time' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'zlib': True,
                                                 'contiguous': False,
                                                 'calendar': 'gregorian',
                                                 'units': f'days since {units_time}',
                                                 'shuffle': False}

                    if 'lat' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}
                    if 'lon' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}

                var_encoding = {
                    var: encoding_each for var in merged_cycle_ds.data_vars}

                encoding = {**coord_encoding, **var_encoding}

                save_dir = f'{output_path}{dataset_name}/aggregated_products/'
                save_path = f'{save_dir}{filename}'

                # If paths don't exist, make them
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save to netcdf
                merged_cycle_ds.to_netcdf(save_path, encoding=encoding)
                checksum = md5(save_path)
                file_size = os.path.getsize(save_path)
                aggregation_success = True

            except Exception as e:
                print(e)
                filename = ''
                save_path = ''
                checksum = ''
                file_size = 0

            # Add cycle to Solr
            item = {}
            item['type_s'] = 'cycle'
            item['dataset_s'] = dataset_name
            item['start_date_s'] = start_date_str
            item['center_date_s'] = overall_center_time_str
            item['end_date_s'] = end_date_str
            item['filename_s'] = filename
            item['filepath_s'] = save_path
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['aggregation_success_b'] = aggregation_success
            item['aggregation_time_s'] = datetime.utcnow().strftime(
                "%Y-%m-%dT%H:%M:%S")
            item['aggregation_version_f'] = version
            if start_date_str in cycles.keys():
                item['id'] = cycles[start_date_str]['id']

            r = solr_update(config, solr_host, [
                            item], solr_collection_name, r=True)
            if r.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give harvested documents the id of the corresponding cycle document
                if aggregation_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = [
                            'type_s:cycle', f'dataset_s:{dataset_name}', f'filename_s:{filename}']
                        cycle_doc = solr_query(
                            config, solr_host, fq, solr_collection_name)
                        cycle_id = cycle_doc[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    r = solr_update(
                        config, solr_host, cycle_granules, solr_collection_name, r=True)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')
