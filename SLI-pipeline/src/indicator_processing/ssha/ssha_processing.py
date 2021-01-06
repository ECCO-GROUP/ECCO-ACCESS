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


def ssha_processing():
    # 1. Get all harvested granule entries
    # 2. Break into 10 days worth of granule chunks
    # 3. Extract out and merge
    #     1. SSHA
    #     2. surface_type (and other variable names - check files)
    #     3. (and lat/lon/time)
    # 4. Determine start and end time
    #     1. add time_bnds to aggregated file -> HERE!
    #     2. give file center time
    # 5. Add metadata to Solr
    config_path = '/Users/kevinmarlis/Developer/JPL/Sea-Level-Indicators/SLI-pipeline/datasets/ssha_ JASON_3_L2_OST_OGDR_GPS/processing_config.yaml'
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    date_regex = '%Y-%m-%dT%H:%M:%SZ'

    # Query for granules
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    all_granules = solr_query(config, solr_host, fq, solr_collection_name)
    remaining_granules = [granule for granule in all_granules]

    granule_dates = [granule['date_s'] for granule in all_granules]
    earliest_granule = all_granules[0]
    latest_granule = all_granules[-1]

    more_dates_to_parse = True

    start_date = datetime.strptime(earliest_granule['date_s'], date_regex)
    end_date = start_date + timedelta(days=10)

    while more_dates_to_parse:
        start_time = ''
        end_time = ''
        center_time = ''
        if datetime.strftime(end_date, date_regex) < latest_granule['date_s']:

            cycle_granules = [granule for granule in remaining_granules if datetime.strftime(
                start_date, date_regex) <= granule['date_s'] and granule['date_s'] <= datetime.strftime(end_date, date_regex)]

            opened_data = []
            # Process the granules
            for index, granule in enumerate(cycle_granules):
                ds = xr.open_dataset(granule['pre_transformation_file_path_s'])

                if index == 0:
                    start_time = datetime.strptime(
                        f'{ds.attrs["first_meas_time"][:10]}T{ds.attrs["first_meas_time"][11:]}', '%Y-%m-%dT%H:%M:%S.%f')
                if index == len(cycle_granules) - 1:
                    end_time = datetime.strptime(
                        f'{ds.attrs["last_meas_time"][:10]}T{ds.attrs["last_meas_time"][11:]}', '%Y-%m-%dT%H:%M:%S.%f')
                    center_time = start_time + ((end_time - start_time)/2)

                drop_list = [key for key in ds.keys() if key != 'ssha']
                if 'surface_type' in drop_list:
                    drop_list.remove('surface_type')
                if 'surface_classification' in drop_list:
                    drop_list.remove('surface_classification')

                ds = ds.drop(drop_list)

                # Replace nans with fill value
                ds.ssha.values = np.where(np.isnan(ds.ssha.values),
                                          default_fillvals['f8'], ds.ssha.values)

                if 'surface_type' in list(ds.keys()):
                    # Replace non open ocean flags with fill value
                    ds.ssha.values = np.where(ds.surface_type.values == 0,
                                              ds.ssha.values, default_fillvals['f8'])
                    ds = ds.drop('surface_type')
                elif 'surface_classification' in list(ds.keys()):
                    # Replace non open ocean flags with fill value
                    ds.ssha.values = np.where(ds.surface_classification.values == 0,
                                              ds.ssha.values, default_fillvals['f8'])
                    ds = ds.drop('surface_classification')

                opened_data.append(ds)

            merged_cycle = xr.concat((opened_data), dim='time')
            # merged_cycle.attrs = {'first_meas_time': start_time,
            #                       'center_time': center_time,
            #                       'last_meas_time': end_time}

            st = np.array(np.datetime64(start_time))
            et = np.array(np.datetime64(end_time))

            time_bnds = np.array([[st], [et]], dtype='datetime64')

            # ValueError: conflicting sizes for dimension 'time': length 2 on 'time_bnds' and length 854310 on 'time'
            merged_cycle = merged_cycle.assign_coords(
                {'time_bnds': (('time', 'nv'), [[st], [et]])})

            merged_cycle.time.attrs.update(bounds='time_bnds')
            merged_cycle.to_netcdf('merged.nc')

            # Update loop variables
            remaining_granules = [
                granule for granule in remaining_granules if granule not in cycle_granules]

            start_date = datetime.strptime(
                remaining_granules[0]['date_s'], date_regex)

            end_date = start_date + timedelta(days=10)


if __name__ == "__main__":
    ssha_processing()
