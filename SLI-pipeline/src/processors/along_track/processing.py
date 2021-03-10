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
               'sort': 'date_dt asc'}

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

    # Query for dataset metadata
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    ds_metadata = solr_query(config, solr_host, fq, solr_collection_name)[0]

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, solr_host, fq, solr_collection_name)

    cycles = {}

    if solr_cycles:
        for cycle in solr_cycles:
            cycles[cycle['start_date_dt']] = cycle

    # Generate list of cycle date tuples (start, end)
    cycle_dates = []
    current_date = datetime.utcnow()
    start_date = datetime.strptime('1992-01-01T00:00:00', date_regex)
    delta = timedelta(days=10)
    curr = start_date
    while curr < current_date:
        if datetime.strftime(curr, date_regex) > '2016':
            cycle_dates.append((curr, curr + delta))
        curr += delta

    var = 'ssh_smoothed'
    reference_date = datetime(1985, 1, 1, 0, 0, 0)

    for (start_date, end_date) in cycle_dates:

        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)

        query_start = datetime.strftime(start_date, '%Y-%m-%dT%H:%M:%SZ')
        query_end = datetime.strftime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
              f'date_dt:[{query_start} TO {query_end}}}']

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
                prior_time = existing_cycle['aggregation_time_dt']
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
            data_start_time = None
            data_end_time = None

            overall_center_time = start_date + ((end_date - start_date)/2)
            overall_center_time_str = datetime.strftime(
                overall_center_time, '%Y-%m-%dT%H:%M:%S')
            units_time = datetime.strftime(
                overall_center_time, "%Y-%m-%d %H:%M:%S")

            for granule in cycle_granules:
                ds = xr.open_dataset(
                    granule['granule_file_path_s'], group='data')

                if 'gmss' in ds.data_vars:
                    ds = ds.drop(['gmss'])
                    ds = ds.rename_dims({'phony_dim_2': 'Time'})

                else:
                    ds = ds.rename_dims({'phony_dim_1': 'Time'})

                ds = ds.rename_vars({'time': 'Time'})
                ds = ds.rename_vars({var: 'SSHA'})
                ds = ds.rename({'lats': 'Latitude'})
                ds = ds.rename({'lons': 'Longitude'})

                # ds[var].encoding['coordinates'] = 'Longitude Latitude'

                ds = ds.drop([var for var in ds.data_vars if var[0] == '_'])
                ds = ds.drop_vars(['ssh'])
                ds = ds.assign_coords(Time=('Time', ds.Time))
                ds = ds.assign_coords(Latitude=ds.Latitude)
                ds = ds.assign_coords(Longitude=ds.Longitude)

                ds.Time.attrs['long_name'] = 'Time'
                ds.Time.attrs['standard_name'] = 'Time'
                ds = ds.assign_coords(Time=[reference_date +
                                            timedelta(seconds=time) for time in ds.Time.values])

                data_start_time = min(
                    data_start_time, ds.Time.values[0]) if data_start_time else ds.Time.values[0]
                data_end_time = max(
                    data_end_time, ds.Time.values[-1]) if data_end_time else ds.Time.values[-1]

                granules.append(ds)

            try:
                # Merge opened granules
                if len(granules) > 1:

                    merged_cycle_ds = xr.concat((granules), dim='Time')

                else:
                    merged_cycle_ds = granules[0]

                # Time bounds

                # Center time
                data_center_time = data_start_time + \
                    ((data_end_time - data_start_time)/2)
                cycle_center_time = start_date + ((end_date - start_date)/2)

                filename_time = datetime.strftime(
                    cycle_center_time, '%Y%m%dT%H%M%S')

                filename = f'ssha_{filename_time}.nc'

                # Var Attributes
                merged_cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(
                    merged_cycle_ds['SSHA'].values)
                merged_cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(
                    merged_cycle_ds['SSHA'].values)

                # Time Attributes
                merged_cycle_ds.Time.attrs['long_name'] = 'Time'

                # Global Attributes
                merged_cycle_ds.attrs = {}
                merged_cycle_ds.attrs['title'] = 'Ten day aggregated GPSOGDR - Reduced dataset'

                merged_cycle_ds.attrs['cycle_start'] = start_date_str
                merged_cycle_ds.attrs['cycle_center'] = datetime.strftime(
                    cycle_center_time, date_regex)
                merged_cycle_ds.attrs['cycle_end'] = end_date_str

                merged_cycle_ds.attrs['data_time_start'] = str(data_start_time)[
                    :19]
                merged_cycle_ds.attrs['data_time_center'] = str(data_center_time)[
                    :19]
                merged_cycle_ds.attrs['data_time_end'] = str(data_end_time)[
                    :19]

                merged_cycle_ds.attrs['original_dataset_title'] = ds_metadata['original_dataset_title_s']
                merged_cycle_ds.attrs['original_dataset_short_name'] = ds_metadata['original_dataset_short_name_s']
                merged_cycle_ds.attrs['original_dataset_url'] = ds_metadata['original_dataset_url_s']
                merged_cycle_ds.attrs['original_dataset_reference'] = ds_metadata['original_dataset_reference_s']

                # Unify var, dims, coords
                # merged_cycle_ds = merged_cycle_ds.rename({var: 'SSHA'})
                # merged_cycle_ds = merged_cycle_ds.rename_dims({'time': 'Time'})
                # merged_cycle_ds = merged_cycle_ds.rename({'time': 'Time'})

                # NetCDF encoding
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

                    if 'ssha' in coord:
                        coord_encoding[coord] = {
                            '_FillValue': default_fillvals['f8']}

                    if 'time' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'zlib': True,
                                                 'contiguous': False,
                                                 'calendar': 'gregorian',
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
                granule_count = len(granules)

            except Exception as e:
                print(e)
                filename = ''
                save_path = ''
                checksum = ''
                file_size = 0
                granule_count = 0

            # Add cycle to Solr
            item = {}
            item['type_s'] = 'cycle'
            item['dataset_s'] = dataset_name
            item['start_date_dt'] = start_date_str
            # item['center_date_s'] = filename_time
            item['end_date_dt'] = end_date_str
            item['granules_in_cycle_i'] = granule_count
            item['filename_s'] = filename
            item['filepath_s'] = save_path
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['aggregation_success_b'] = aggregation_success
            item['aggregation_time_dt'] = datetime.utcnow().strftime(
                "%Y-%m-%dT%H:%M:%S")
            item['aggregation_version_f'] = version
            if start_date_str in cycles.keys():
                item['id'] = cycles[start_date_str]['id']

            r = solr_update(config, solr_host, [item],
                            solr_collection_name, r=True)
            if r.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give harvested documents the id of the corresponding cycle document
                if aggregation_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle', f'dataset_s:{dataset_name}',
                              f'filename_s:{filename}']
                        cycle_doc = solr_query(
                            config, solr_host, fq, solr_collection_name)
                        cycle_id = cycle_doc[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    r = solr_update(
                        config, solr_host, cycle_granules, solr_collection_name)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

        # Quit before most recent cycle (insufficient data)
        if current_date < end_date + delta:
            print(
                f'Insufficient data for complete {start_date + delta} to {end_date + delta} cycle')
            break
