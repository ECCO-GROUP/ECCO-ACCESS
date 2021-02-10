import os
import sys
import yaml
import requests
import hashlib
import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
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
    date_regex = '%Y-%m-%dT%H:%M:%SZ'

    # Query for all dataset granules
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    remaining_granules = solr_query(
        config, solr_host, fq, solr_collection_name)

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, solr_host, fq, solr_collection_name)

    cycles = {}

    if solr_cycles:
        for cycle in solr_cycles:
            cycles[cycle['start_date_s']] = cycle

    # Generate list of cycle date tuples (start, end)
    cycle_dates = []
    start_date = datetime.strptime('2016-01-01T00:00:00Z', date_regex)
    end_date = datetime.strptime('1992-01-01T00:00:00Z', date_regex)
    delta = timedelta(days=-5)
    curr = start_date
    while curr > end_date:
        cycle_dates.append((curr + delta, curr))
        curr += delta
    cycle_dates.reverse()
    var = 'SLA'

    for (start_date, end_date) in cycle_dates:
        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)

        cycle_granules = [granule for granule in remaining_granules if
                          start_date_str <= granule['date_s'] and
                          granule['date_s'] <= end_date_str]

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
            ds = xr.open_dataset(cycle_granules[0]['granule_file_path_s'])
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            try:

                # Center time

                overall_center_time = start_date + ((end_date - start_date)/2)

                filename_time = datetime.strftime(
                    overall_center_time, '%Y%m%dT%H%M%S')
                filename = f'sla_{filename_time}.nc'

                # Var Attributes
                ds[var].attrs['valid_min'] = np.nanmin(ds[var].values)
                ds[var].attrs['valid_max'] = np.nanmax(ds[var].values)

                # NetCDF encoding
                # encoding_each = {'zlib': True,
                #                  'complevel': 5,
                #                  'shuffle': True,
                #                  '_FillValue': default_fillvals['f8']}

                # coord_encoding = {}
                # for coord in ds.coords:
                #     coord_encoding[coord] = {'_FillValue': None}

                #     if 'time' in coord:
                #         coord_encoding[coord] = {'_FillValue': None,
                #                                  'dtype': 'float64',
                #                                  'zlib': True,
                #                                  'complevel': 6,
                #                                  'contiguous': False,
                #                                  'calendar': 'gregorian',
                #                                  'shuffle': False}
                #         if coord != 'time_step':
                #             coord_encoding[coord]['units'] = "seconds since 2000-01-01 00:00:00.0"
                #     if 'lat' in coord:
                #         coord_encoding[coord] = {'_FillValue': None,
                #                                  'dtype': 'float32'}
                #     if 'lon' in coord:
                #         coord_encoding[coord] = {'_FillValue': None,
                #                                  'dtype': 'float32'}

                # var_encoding = {
                #     var: encoding_each for var in ds.data_vars}

                # encoding = {**coord_encoding, **var_encoding}

                save_dir = f'{output_path}{dataset_name}/aggregated_products/'
                save_path = f'{save_dir}{filename}'

                # If paths don't exist, make them
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save to netcdf
                # ds.to_netcdf(save_path, encoding=encoding)
                ds.to_netcdf(save_path)
                checksum = md5(save_path)
                file_size = os.path.getsize(save_path)
                aggregation_success = True
                exit()

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
            item['end_date_s'] = end_date_str
            item['filename_s'] = filename
            item['filepath_s'] = save_path
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['aggregation_success_b'] = aggregation_success
            item['aggregation_time_s'] = datetime.utcnow().strftime(
                "%Y-%m-%dT%H:%M:%SZ")
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

        # Update loop variables
        remaining_granules = [granule for granule in remaining_granules
                              if granule not in cycle_granules]

        # # Quit before most recent cycle (insufficient data)
        # if current_date < end_date + delta:
        #     print(
        #         f'Insufficient data for complete {start_date + delta} to {end_date + delta} cycle')
        #     break
