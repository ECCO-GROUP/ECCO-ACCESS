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


def testing(ds, var):
    time_da = ds.time
    ssha_da = ds.ssha
    var_da = ds[var]

    vals = ds[var].values
    non_nan_vals = vals[~np.isnan(vals)]

    mean = np.nanmean(ds[var].values)
    rms = np.sqrt(np.mean(non_nan_vals**2))
    std = np.std(non_nan_vals)

    try:
        OFFSET, AMPLITUDE, _, _ = delta_orbit_altitude_offset_amplitude(
            time_da, ssha_da, var_da)
    except Exception as e:
        print(e)
        OFFSET = 100
        AMPLITUDE = 100
    tests = [('Mean', mean), ('RMS', rms), ('STD', std),
             ('Offset', OFFSET), ('Amplitude', AMPLITUDE)]

    for (test, result) in tests:
        test_array = np.full(len(ds[var]), result, dtype='float32')
        ds[test] = xr.DataArray(test_array, ds[var].coords, ds[var].dims)
        ds[test].attrs['comment'] = f'{test} test value from original granule in cycle.'
    return ds


def delta_orbit_altitude_offset_amplitude(time_da, ssha_da, gps_ssha_da):
    # time_da, ssha_da, and gps_ssha_da are xarray DataArray objects
    # least squares fit for an OFFSET and AMPLITUDE of
    # delta orbit altitude between the GPS orbit altitude
    # and DORIS orbit altitude
    # ssh = orbit_altitude - range - corrections
    #
    # in Shailen's SSH files, there is both 'gps_ssha', and 'ssha'
    #
    # gps_ssha - ssha = delta orbit_altitude
    #    because range and corrections are the same for both ssha fields
    #
    # therefore we seek a solution the following equation
    # y = C0 +  C1 cos(omega t) + C2 sin(omega t)
    #
    # where
    # y      : delta GPS orbit altitude
    # C0     : OFFSET
    # C1, C2 : AMPLITUDES of cosine and sine terms comprising a phase shifted oscillation
    # omega  : period of one orbit resolution in seconds
    # t      : time in seconds
    # calculate delta orbit altitude
    delta_orbit_altitude = gps_ssha_da.values - ssha_da.values
    # calculate time (in seconds) from the first to last observations in
    # record
    if type(time_da.values[0]) == np.datetime64:
        td = (time_da.values - time_da[0].values)/1e9
        td = td.astype('float')
    else:
        td = time_da.values
    # plt.plot(td, delta_orbit_altitude, 'k.')
    # plt.xlabel('time delta (s)')
    # plt.grid()
    # plt.title('delta orbit altitude [m]')
    # calculate omega * t
    omega = 2.*np.pi/6745.756
    omega_t = omega * td
    # pull values of omega_t and the delta_orbit_altitude only where
    # the delta_orbit_altitude is not nan (i.e., not missing)
    omega_t_nn = omega_t[~np.isnan(delta_orbit_altitude)]
    delta_orbit_altitude_nn = delta_orbit_altitude[~np.isnan(
        delta_orbit_altitude)]
    # Least squares solution will take the form:
    # c = inv(A.T A) A.T  delta_orbit_altitude.T
    # where *.T indicates transpose
    # inv indicates matrix inverse
    # the three columns of the A matrix
    CONST_TERM = np.ones(len(omega_t_nn))
    COS_TERM = np.cos(omega_t_nn)
    SIN_TERM = np.sin(omega_t_nn)
    # construct A matrix
    A = np.column_stack((CONST_TERM, COS_TERM, SIN_TERM))
    c = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)),
                            A.T), delta_orbit_altitude_nn.T)
    OFFSET = c[0]
    AMPLITUDE = np.sqrt(c[1]**2 + c[2]**2)
    # estimated time series
    y_e = c[0] + c[1]*np.cos(omega_t) + c[2] * np.sin(omega_t)
    # the c vector will have 3 elements
    return OFFSET, AMPLITUDE, delta_orbit_altitude, y_e


def processing(config_path='', output_path=''):

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    version = config['version']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    date_regex = '%Y-%m-%dT%H:%M:%S'
    solr_regex = f'{date_regex}Z'

    """
    Flags
    - DS field names change after a certain date
    - All possible flag names are used by checking if each flag
      is in the keys of a DS
    1.  rad_surface_type_flag (rad_surf_type) = 0 (open ocean)
    2.  surface_classification_flag (surface_type) = 0 (open ocean)
    3.  alt_qual (alt_quality_flag)= 0 (good)
    4.  rad_qual (rad_quality_flag) = 0 (good)
    5.  geo_qual (geophysical_quality_flag)= 0 (good)
    6.  meteo_map_availability_flag (ecmwf_meteo_map_avail) = 0 ('2_maps_nominal')
    7.  rain_flag = 0 (no rain)
    8.  rad_rain_flag = 0 (no rain)
    9.  ice_flag = 0 (no ice)
    10. rad_sea_ice_flag = 0 (no ice)
    """

    flags = ['rad_surface_type_flag', 'surface_classification_flag', 'alt_qual',
             'rad_qual', 'geo_qual', 'meteo_map_availability_flag', 'rain_flag',
             'rad_rain_flag', 'ice_flag', 'rad_sea_ice_flag', 'rad_surf_type',
             'surface_type', 'alt_quality_flag', 'rad_quality_flag',
             'geophysical_quality_flag', 'ecmwf_meteo_map_avail']

    # Query for all dataset granules
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    remaining_granules = solr_query(config, fq)

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, fq)

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

    var = 'gps_ssha'
    tests = ['Mean', 'RMS', 'STD', 'Offset', 'Amplitude']

    for (start_date, end_date) in cycle_dates:

        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)

        query_start = datetime.strftime(start_date, solr_regex)
        query_end = datetime.strftime(end_date, solr_regex)
        fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
              f'date_dt:[{query_start} TO {query_end}]']

        cycle_granules = solr_query(config, fq)

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
            if start_date_str + 'Z' in cycles.keys():
                existing_cycle = cycles[start_date_str + 'Z']
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

            opened_data = []
            start_times = []
            end_times = []

            try:

                # Process the granules
                for granule in cycle_granules:
                    uses_groups = False

                    # netCDF granules from 2020-10-29 on contain groups
                    ds = xr.open_dataset(granule['granule_file_path_s'])

                    if 'lon' in ds.coords:
                        ds = ds.rename({'lon': 'Longitude'})
                        ds = ds.rename({'lat': 'Latitude'})
                        ds[var].encoding['coordinates'] = 'Longitude Latitude'
                        ds_keys = list(ds.keys())
                    else:
                        uses_groups = True

                        ds = xr.open_dataset(granule['granule_file_path_s'],
                                             group='data_01/ku')
                        ds_flags = xr.open_dataset(granule['granule_file_path_s'],
                                                   group='data_01')

                        ds_flags = ds_flags.rename({'longitude': 'Longitude'})
                        ds_flags = ds_flags.rename({'latitude': 'Latitude'})

                        ds_keys = list(ds_flags.keys())

                        ds = ds.assign_coords(
                            {"Longitude": ds_flags.Longitude})
                        ds = ds.assign_coords(
                            {"Latitude": ds_flags.Latitude})

                    start_times.append(ds.time.values[::])

                    # Remove outliers before running tests
                    ds[var].values[np.greater(
                        abs(ds[var].values), 1.5, where=~np.isnan(ds[var].values))] = np.nan

                    # Run tests, returns byte results convert to int
                    ds = testing(ds, var)

                    # Mask out flagged data
                    for flag in flags:
                        if flag in ds_keys:
                            if uses_groups:
                                if np.isnan(ds_flags[flag].values).all():
                                    continue

                                ds[var].values = np.where(ds_flags[flag].values == 0,
                                                          ds[var].values,
                                                          default_fillvals['f8'])
                            else:
                                if np.isnan(ds[flag].values).all():
                                    continue
                                ds[var].values = np.where(ds[flag].values == 0,
                                                          ds[var].values,
                                                          default_fillvals['f8'])

                    # Replace nans with fill value
                    ds[var].values = np.where(np.isnan(ds[var].values),
                                              default_fillvals['f8'],
                                              ds[var].values)

                    keep_keys = tests + [var]
                    ds = ds.drop([key for key in ds.keys()
                                  if key not in keep_keys])

                    opened_data.append(ds)

                # Merge
                merged_cycle_ds = xr.concat((opened_data), dim='time')

                # Time bounds
                start_times = np.concatenate(start_times).ravel()
                end_times = start_times[1:]
                end_times = np.append(
                    end_times, end_times[-1] + np.timedelta64(1, 's'))

                # Center time
                data_center_time = start_times[0] + \
                    ((end_times[-1] - start_times[0])/2)
                center_time = start_date + ((end_date - start_date) / 2)
                center_time_str = datetime.strftime(center_time, date_regex)
                filename_time = datetime.strftime(center_time, '%Y%m%dT%H%M%S')

                filename = f'ssha_{filename_time}.nc'

                # Var Attributes
                merged_cycle_ds[var].attrs['valid_min'] = np.nanmin(
                    merged_cycle_ds[var].values)
                merged_cycle_ds[var].attrs['valid_max'] = np.nanmax(
                    merged_cycle_ds[var].values)

                # Time Attributes
                merged_cycle_ds.time.attrs['long_name'] = 'time'

                # Global Attributes
                merged_cycle_ds.attrs = {}
                merged_cycle_ds.attrs['title'] = 'Ten day aggregated GPSOGDR - Reduced dataset'

                merged_cycle_ds.attrs['cycle_start'] = start_date_str
                merged_cycle_ds.attrs['cycle_center'] = center_time_str
                merged_cycle_ds.attrs['cycle_end'] = end_date_str

                merged_cycle_ds.attrs['data_time_start'] = str(start_times[0])[
                    :19]
                merged_cycle_ds.attrs['data_time_center'] = str(data_center_time)[
                    :19]
                merged_cycle_ds.attrs['data_time_end'] = str(
                    end_times[-1])[:19]

                merged_cycle_ds.attrs['original_dataset_title'] = 'Jason-3 GPS based orbit and SSHA OGDR'
                merged_cycle_ds.attrs['original_dataset_short_name'] = 'JASON_3_L2_OST_OGDR_GPS'
                merged_cycle_ds.attrs['original_dataset_url'] = 'https://podaac.jpl.nasa.gov/dataset/JASON_3_L2_OST_OGDR_GPS?ids=Platforms:Processing%20Levels&values=JASON-3::2%20-%20Geophys.%20Variables,%20Sensor%20Coordinates'
                merged_cycle_ds.attrs['original_dataset_reference'] = 'https://podaac-tools.jpl.nasa.gov/drive/files/allData/jason3/preview/L2/GPS-OGDR/docs/j3_user_handbook.pdf'

                # Unify var, dims, coords
                merged_cycle_ds = merged_cycle_ds.rename({var: 'SSHA'})
                merged_cycle_ds = merged_cycle_ds.rename_dims({'time': 'Time'})
                merged_cycle_ds = merged_cycle_ds.rename({'time': 'Time'})

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

                    if 'SSHA' in coord:
                        coord_encoding[coord] = {
                            '_FillValue': default_fillvals['f8']}

                    if 'Time' in coord:
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
                granule_count = len(opened_data)

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
            item['center_date_dt'] = center_time_str
            item['end_date_dt'] = end_date_str
            item['granules_in_cycle_i'] = granule_count
            item['filename_s'] = filename
            item['filepath_s'] = save_path
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['aggregation_success_b'] = aggregation_success
            item['aggregation_time_dt'] = datetime.utcnow().strftime(date_regex)
            item['aggregation_version_f'] = version
            if start_date_str in cycles.keys():
                item['id'] = cycles[start_date_str]['id']

            r = solr_update(config, [item], r=True)
            if r.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give harvested documents the id of the corresponding cycle document
                if aggregation_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle', f'dataset_s:{dataset_name}',
                              f'filename_s:{filename}']
                        cycle_doc = solr_query(config, fq)
                        cycle_id = cycle_doc[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    r = solr_update(config, cycle_granules)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

        # Quit before most recent cycle (insufficient data)
        if current_date < end_date + delta:
            print(
                f'Insufficient data for complete {start_date + delta} to {end_date + delta} cycle')
            break
