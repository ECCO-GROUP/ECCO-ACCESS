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


def solr_query(config, fq):
    """
    Queries Solr database using the filter query passed in.
    Returns list of Solr entries that satisfies the query.
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000,
               'sort': 'date_s asc'}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, update_body, r=False):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def processing(config_path='', output_path=''):

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    version = config['version']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    date_regex = '%Y-%m-%dT%H:%M:%S'
    solr_regex = f'{date_regex}Z'

    # Query for dataset metadata
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    ds_metadata = solr_query(config, fq)[0]

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, fq)

    cycles = {}

    if solr_cycles:
        for cycle in solr_cycles:
            cycles[cycle['start_date_dt']] = cycle

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

    # 1812 dataset only uses one granule per cycle
    for (start_date, end_date) in cycle_dates:

        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)
        center_time = start_date + ((end_date - start_date)/2)
        center_time_str = datetime.strftime(center_time, solr_regex)

        # Find the granule with date closest to center of cycle
        # Uses special Solr query function to automatically return granules in proximal order
        query_start = datetime.strftime(start_date, solr_regex)
        query_end = datetime.strftime(end_date, solr_regex)
        fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
              f'date_dt:[{query_start} TO {query_end}}}']
        bf = f'recip(abs(ms({center_time_str},date_dt)),3.16e-11,1,1)'

        getVars = {'q': '*:*',
                   'fq': fq,
                   'bf': bf,
                   'defType': 'edismax',
                   'rows': 300000,
                   'sort': 'date_s asc'}

        url = f'{solr_host}{solr_collection_name}/select?'
        response = requests.get(url, params=getVars)
        cycle_granules = response.json()['response']['docs']

        if not cycle_granules:
            print(f'No granules for cycle {start_date_str} to {end_date_str}')
            continue

        granule = cycle_granules[0]

        updating = False

        # If any single granule in a cycle satisfies any of the following conditions:
        # - has been updated,
        # - previously failed,
        # - has a different version than what is in the config
        # reprocess the entire cycle
        if cycles:
            if start_date_str + 'Z' in cycles.keys():
                existing_cycle = cycles[start_date_str + 'Z']
                prior_time = existing_cycle['processing_time_dt']
                prior_success = existing_cycle['processing_success_b']
                prior_version = existing_cycle['processing_version_f']

                if not prior_success or prior_version != version:
                    updating = True

                if prior_time < granule['modified_time_dt']:
                    updating = True
            else:
                updating = True
        else:
            updating = True

        if updating:
            processing_success = False
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            units_time = datetime.strftime(center_time, "%Y-%m-%d %H:%M:%S")

            ds = xr.open_dataset(granule['granule_file_path_s'])

            # Rename var to 'SSHA'
            ds = ds.rename({var: 'SSHA'})

            try:
                ds.attrs = {}
                ds.attrs['title'] = 'Sea Level Anormaly Estimate based on Altimeter Data'

                ds.attrs['cycle_start'] = start_date_str
                ds.attrs['cycle_center'] = center_time_str
                ds.attrs['cycle_end'] = end_date_str

                data_time_start = ds.Time_bounds.values[0][0]
                data_time_end = ds.Time_bounds.values[-1][1]
                data_time_center = data_time_start + \
                    ((data_time_end - data_time_start)/2)

                ds.attrs['data_time_start'] = np.datetime_as_string(
                    data_time_start, unit='s')
                ds.attrs['data_time_center'] = np.datetime_as_string(
                    data_time_center, unit='s')
                ds.attrs['data_time_end'] = np.datetime_as_string(
                    data_time_end, unit='s')

                ds.attrs['original_dataset_title'] = ds_metadata['original_dataset_title_s']
                ds.attrs['original_dataset_short_name'] = ds_metadata['original_dataset_short_name_s']
                ds.attrs['original_dataset_url'] = ds_metadata['original_dataset_url_s']
                ds.attrs['original_dataset_reference'] = ds_metadata['original_dataset_reference_s']

                # Center time
                filename_time = datetime.strftime(
                    overall_center_time, '%Y%m%dT%H%M%S')
                filename = f'sla_{filename_time}.nc'

                # Var Attributes
                ds['SSHA'].attrs['valid_min'] = np.nanmin(ds['SSHA'].values)
                ds['SSHA'].attrs['valid_max'] = np.nanmax(ds['SSHA'].values)

                encoding_each = {'zlib': True,
                                 'complevel': 5,
                                 'dtype': 'float32',
                                 'shuffle': True,
                                 '_FillValue': default_fillvals['f8']}

                coord_encoding = {}
                for coord in ds.coords:
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

                var_encoding = {var: encoding_each for var in ds.data_vars}

                encoding = {**coord_encoding, **var_encoding}

                save_dir = f'{output_path}{dataset_name}/cycle_products/'
                save_path = f'{save_dir}{filename}'

                # If paths don't exist, make them
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save to netcdf
                ds.to_netcdf(save_path, encoding=encoding)
                checksum = md5(save_path)
                file_size = os.path.getsize(save_path)
                processing_success = True

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
            item['start_date_dt'] = start_date_str
            item['center_date_dt'] = center_time_str
            item['end_date_dt'] = end_date_str
            item['filename_s'] = filename
            item['filepath_s'] = save_path
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['processing_success_b'] = processing_success
            item['processing_time_dt'] = datetime.utcnow().strftime(date_regex)
            item['processing_version_f'] = version
            if start_date_str in cycles.keys():
                item['id'] = cycles[start_date_str]['id']

            r = solr_update(config, [item], r=True)
            if r.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give harvested documents the id of the corresponding cycle document
                if processing_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = [
                            'type_s:cycle', f'dataset_s:{dataset_name}', f'filename_s:{filename}']
                        cycle_doc = solr_query(config, fq)
                        cycle_id = cycle_doc[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    r = solr_update(config, cycle_granules, r=True)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

    # Query for Solr failed harvest documents
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}',
          f'processing_success_b:false']
    failed_processing = solr_query(config, fq)

    if not failed_processing:
        processing_status = f'All cycles successfully processed'
    else:
        # Query for Solr successful harvest documents
        fq = ['type_s:cycle', f'dataset_s:{dataset_name}',
              f'processing_success_b:true']
        successful_processing = solr_query(config, fq)

        if not successful_processing:
            processing_status = f'No cycles successfully processed (either all failed or no granules to process)'
        else:
            processing_status = f'{len(failed_harvesting)} harvested granules failed'

    ds_metadata['processing_status_s'] = {"set": processing_status}
    r = solr_update(config, [ds_metadata], r=True)

    if r.status_code == 200:
        print('Successfully updated Solr dataset document\n')
    else:
        print('Failed to update Solr dataset document\n')
