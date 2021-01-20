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

# 1. Get all harvested granule entries
# 2. Break into 10 days worth of granule chunks
# 3. Extract out and merge
#     1. SSHA
#     2. surface_type (and other variable names - check files)
#     3. (and lat/lon/time)
# 4. Determine start and end time
#     1. add time_bnds to aggregated file
#           - need to add encoding -> HERE!
#           - time encoding has been weird
#     2. give file center time
# 5. Add metadata to Solr
# 6. Incorporate into pipeline

# Add in other vars (see ~line 99)


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
    # config_path = '/Users/kevinmarlis/Developer/JPL/Sea-Level-Indicators/SLI-pipeline/datasets/ssha_JASON_3_L2_OST_OGDR_GPS/processing_config.yaml'
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    version = config['version']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    date_regex = '%Y-%m-%dT%H:%M:%SZ'

    # 1.  rad_surface_type_flag (rad_surf_type) = 0 (open ocean)
    # 2.  surface_classification_flag (surface_type) = 0 (open ocean)
    # 3.  alt_qual (alt_quality_flag)= 0 (good)
    # 4.  rad_qual (rad_quality_flag) = 0 (good)
    # 5.  geo_qual (geophysical_quality_flag)= 0 (good)
    # 6.  meteo_map_availability_flag (ecmwf_meteo_map_avail) = 0 ('2_maps_nominal')
    # 7.  rain_flag = 0 (no rain)
    # 8.  rad_rain_flag = 0 (no rain)
    # 9.  ice_flag = 0 (no ice)
    # 10. rad_sea_ice_flag = 0 (no ice)
    flags = ['rad_surface_type_flag', 'surface_classification_flag', 'alt_qual',
             'rad_qual', 'geo_qual', 'meteo_map_availability_flag', 'rain_flag',
             'rad_rain_flag', 'ice_flag', 'rad_sea_ice_flag', 'rad_surf_type',
             'surface_type', 'alt_quality_flag', 'rad_quality_flag',
             'geophysical_quality_flag', 'ecmwf_meteo_map_avail']

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

    # Cycles are defined as starting at 2000-01-01 00:00:00.0, and lasting
    # 10 days. Cycle periods are generated, and granules are matched into a
    # specific cycle to be aggregated.

    # Generate list of cycle date tuples (start, end)
    cycle_dates = []
    current_date = datetime.utcnow()
    start_date = datetime.strptime('2000-01-01T00:00:00Z', date_regex)
    delta = timedelta(days=10)
    curr = start_date
    while curr < current_date:
        cycle_dates.append((curr, curr + delta))
        curr += delta

    var = 'gps_ssha'

    for (start_date, end_date) in cycle_dates:
        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)

        cycle_granules = [granule for granule in remaining_granules if
                          start_date_str <= granule['date_s'] and
                          granule['date_s'] <= end_date_str]

        if len(cycle_granules) < 120:
            print(
                f'Not enough granules for complete cycle {start_date_str} to {end_date_str}')
            continue

        updating = False

        # If any single granule in a cycle satisfies any of the following conditions:
        # - has been updated,
        # - previously failed,
        # - has a different version than what is in the config
        # reaggregate the entire cycle
        if cycles:
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

        if updating:
            aggregation_success = False
            print(f'Aggregating cycle {start_date_str} to {end_date_str}')

            opened_data = []
            start_times = []
            end_times = []

            try:

                # Process the granules
                for granule in cycle_granules:
                    uses_groups = False

                    # netCDF granules from 2020-10-29 on contain groups
                    ds = xr.open_dataset(
                        granule['pre_transformation_file_path_s'])
                    if 'lon' in ds.coords:

                        ds[var] = ds[var].assign_coords(
                            {'longitude': ds.lon})
                        ds[var].encoding['coordinates'] = 'longitude latitude'
                        ds = ds.reset_coords(['lon'], drop=True)

                    if 'lat' in ds.coords:
                        ds[var] = ds[var].assign_coords(
                            {'latitude': ds.lat})
                        ds = ds.reset_coords(['lat'], drop=True)
                        ds[var].encoding['coordinates'] = 'longitude latitude'

                    else:
                        uses_groups = True

                        ds = xr.open_dataset(granule['pre_transformation_file_path_s'],
                                             group='data_01/ku')
                        ds_flags = xr.open_dataset(granule['pre_transformation_file_path_s'],
                                                   group='data_01')

                    if uses_groups:
                        ds_keys = list(ds_flags.keys())
                        ds = ds.assign_coords(
                            {"longitude": ds_flags.longitude})
                    else:
                        ds_keys = list(ds.keys())

                    start_times.append(ds.time.values[::])

                    # Replace nans with fill value
                    ds[var].values = np.where(np.isnan(ds[var].values),
                                              default_fillvals['f8'], ds[var].values)

                    # Loop through flags and replace with nans
                    for flag in flags:
                        if flag in ds_keys:
                            if uses_groups:
                                if np.isnan(ds_flags[flag].values).all():
                                    continue

                                ds[var].values = np.where(ds_flags[flag].values == 0,
                                                          ds[var].values, default_fillvals['f8'])
                            else:
                                if np.isnan(ds[flag].values).all():
                                    continue
                                ds[var].values = np.where(ds[flag].values == 0,
                                                          ds[var].values, default_fillvals['f8'])

                            if np.all(ds[var].values == default_fillvals['f8']):
                                print(flag)
                                exit()

                    ds = ds.drop([key for key in ds.keys() if key != var])

                    opened_data.append(ds)

                # Merge
                merged_cycle_ds = xr.concat((opened_data), dim='time')

                # Time bounds
                start_times = np.concatenate(start_times).ravel()
                end_times = start_times[1:]
                end_times = np.append(
                    end_times, end_times[-1] + np.timedelta64(1, 's'))
                time_bnds = np.array([[i, j]
                                      for i, j in zip(start_times, end_times)])

                merged_cycle_ds = merged_cycle_ds.assign_coords(
                    {'time_bnds': (('time', 'nv'), time_bnds)})

                merged_cycle_ds.time.attrs.update(bounds='time_bnds')

                # Center time
                overall_center_time = start_times[0] + \
                    ((end_times[-1] - start_times[0])/2)
                ts = datetime.strptime(str(overall_center_time)[
                    :19], '%Y-%m-%dT%H:%M:%S')
                filename_time = datetime.strftime(ts, '%Y%m%dT%H%M%S')
                filename = f'ssha_{filename_time}.nc'

                # SSHA Attributes
                merged_cycle_ds[var].attrs['valid_min'] = np.nanmin(
                    merged_cycle_ds[var].values)
                merged_cycle_ds[var].attrs['valid_max'] = np.nanmax(
                    merged_cycle_ds[var].values)

                # NetCDF encoding
                encoding_each = {'zlib': True,
                                 'complevel': 5,
                                 'shuffle': True,
                                 '_FillValue': default_fillvals['f8']}

                coord_encoding = {}
                for coord in merged_cycle_ds.coords:
                    coord_encoding[coord] = {'_FillValue': None}

                    if 'time' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float64',
                                                 'zlib': True,
                                                 'complevel': 6,
                                                 'contiguous': False,
                                                 'calendar': 'gregorian',
                                                 'shuffle': False}
                        if coord != 'time_step':
                            coord_encoding[coord]['units'] = "seconds since 2000-01-01 00:00:00.0"
                    if 'lat' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}
                    if 'lon' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}

                var_encoding = {
                    var: encoding_each for var in merged_cycle_ds.data_vars}

                encoding = {**coord_encoding, **var_encoding}

                save_path = f'/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/aggregated_products/{filename}'

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
            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

        # Update loop variables
        remaining_granules = [granule for granule in remaining_granules
                              if granule not in cycle_granules]

        # Quit before most recent cycle (insufficient data)
        if current_date < end_date + delta:
            print(
                f'Insufficient data for complete {start_date + delta} to {end_date + delta} cycle')
            break


if __name__ == "__main__":
    processing()
