import os
import sys
import gzip
import yaml
import shutil
import hashlib
import logging
import requests
import pandas as pd
from collections import defaultdict
from requests.auth import HTTPBasicAuth
from netCDF4 import default_fillvals
import numpy as np
import xarray as xr
from pathlib import Path
from xml.etree.ElementTree import parse
from datetime import datetime, timedelta
from urllib.request import urlopen, urlcleanup, urlretrieve

log = logging.getLogger(__name__)


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
               'rows': 300000}

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


def indicators(config_path='', output_path='', s3=None, solr_info=''):
    config_path = 'Sea-Level-Indicators/SLI-pipeline/src/indicators/indicators.yaml'
    output_path = 'sealevel_output'
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    filename = 'indicator.nc'
    output_path = f'{output_path}/indicator/'

    # If target paths don't exist, make them
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    time_format = "%Y-%m-%dT%H:%M:%S"

    chk_time = datetime.utcnow().strftime(time_format)

    gridded_range = ['1992-01-01T00:00:00', '2018-12-31T00:00:00']
    along_track_range = [gridded_range[-1], chk_time]

    # Query for indicator doc on Solr
    fq = [f'type_s:indicator']
    indicator_query = solr_query(config, solr_host, fq, solr_collection_name)

    update = len(indicator_query) == 1

    # ==============================================
    # Create or update indicator netcdf file
    # ==============================================

    if update:
        indicator_metadata = indicator_query[0]
        modified_time = indicator_metadata['modified_time_dt']
    else:
        modified_time = '1992-01-01T:00:00:00'

    # Query for update cycles after modified_time
    fq = [f'type_s:cycle', 'aggregation_success_b:true',
          f'aggregation_time_s:[{modified_time} TO NOW]']
    updated_cycles = solr_query(
        config, solr_host, fq, solr_collection_name)

    # cycle start dates that need modifying
    modified_cycle_start_dates = defaultdict(list)
    for cycle in updated_cycles:
        modified_cycle_start_dates[cycle['start_date_s']].append(cycle)

    # Loop through cycles with modified data
    # If cycle uses along track data, there will be multiple cycle data files to open
    # Use each to calculate indicator value

    indicator_values = []
    for cycle_start_date, cycle_list in modified_cycle_start_dates.items():
        opened_cycles = []
        for cycle in cycle_list:
            filepath = cycle['filepath_s']
            start_date_s = cycle_start_date
            ds = xr.open_dataset(filepath)
            opened_cycles.append(ds)

        # Calculate indicator value
        indicator_value = np.random.random(1)[0]
        indicator_values.append((cycle_start_date, indicator_value))

    indicator_values.sort()

    # Either open existing indicator ds or create new one
    if update:
        indicator_ds = xr.open_dataset(
            indicator_metadata['indicator_file_path_s'])

        times = indicator_ds.Time.values
        data = indicator_ds.Index.values
        data_d = {time: val for time, val in zip(times, data)}

        for time, value in indicator_values:
            data_d[time] = value

        time_vals = [(time, value) for time, value in data_d.items()]
        time_vals.sort()

        new_times = [time[0] for time in time_vals]
        new_data = [vals[1] for vals in time_vals]

        indicator_ds = xr.Dataset(
            {
                'Index': xr.DataArray(
                    data=new_data,
                    dims=['Time'],
                    coords={"Time": new_times},
                    attrs=indicator_ds.Index.attrs
                )
            },
            attrs=indicator_ds.attrs
        )

        print(indicator_ds)

    else:
        times = pd.date_range(start='1992-01-01',
                              end=datetime.today(), freq='1D')
        data = [default_fillvals['f8']]*len(times)

        indicator_ds = xr.Dataset(
            {
                'Index': xr.DataArray(
                    data=data,
                    dims=['Time'],
                    coords={"Time": times},
                    attrs={
                        'comment': 'index dataarray'
                    }
                )
            },
            attrs={'comment': 'global attribute'}
        )

        # Update values in indicator netcdf
        for time, value in indicator_values:

            # indicator_ds.Index.loc[time] = value
            indicator_ds.Index.loc[{"Time": time}] = value

            # dict(time=slice("2000-01-01", "2000-01-02"))

    indicator_ds.to_netcdf(output_path + filename)

    # ==============================================
    # Create or update indicator on Solr
    # ==============================================

    # Create new entry
    if not update:
        indicator_meta = {}
        indicator_meta['type_s'] = 'indicator'
        indicator_meta['filename_s'] = filename
        indicator_meta['indicator_file_path_s'] = output_path + filename
        indicator_meta['modified_time_dt'] = chk_time
        indicator_meta['start_date_dt'] = np.datetime_as_string(
            indicator_ds.Time.values[0], unit='s')

        indicator_meta['end_date_dt'] = np.datetime_as_string(
            indicator_ds.Time.values[-1], unit='s')
        indicator_meta['checksum_s'] = md5(output_path+filename)
        indicator_meta['file_size_l'] = os.path.getsize(output_path)

        # Update Solr with dataset metadata
        r = solr_update(config, solr_host, [indicator_meta],
                        solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

    # Update existing entry
    else:
        indicator_metadata = indicator_query[0]

        update_meta = {}
        update_meta['id'] = indicator_metadata['id']
        update_meta['modified_time_dt'] = {"set": chk_time+'Z'}
        update_meta['start_date_dt'] = {"set": np.datetime_as_string(
            indicator_ds.Time.values[0], unit='s')+'Z'}
        update_meta['end_date_dt'] = {"set": np.datetime_as_string(
            indicator_ds.Time.values[-1], unit='s')+'Z'}
        update_meta['checksum_s'] = {"set": md5(output_path+filename)}
        update_meta['file_size_l'] = {"set": os.path.getsize(output_path)}

        print(update_meta)

        # Update Solr with dataset metadata
        r = solr_update(config, solr_host, [update_meta],
                        solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')


if __name__ == '__main__':
    indicators()
