import os
import hashlib
from datetime import datetime, timedelta
import yaml
import requests
import numpy as np
import xarray as xr
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

    query_params = {'q': '*:*',
                    'fq': fq,
                    'rows': 300000,
                    'sort': 'date_dt asc'}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def solr_update(config, update_body):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


def process_along_track(cycle_granules, ds_metadata, cycle_dates):
    """
    Performs all processing required for along track datasets
    """
    var = 'ssh_smoothed'
    reference_date = datetime(1985, 1, 1, 0, 0, 0)
    granules = []
    data_start_time = None
    data_end_time = None

    for granule in cycle_granules:
        ds = xr.open_dataset(granule['granule_file_path_s'], group='data')

        if 'gmss' in ds.data_vars:
            ds = ds.drop(['gmss'])
            ds = ds.rename_dims({'phony_dim_2': 'Time'})

        else:
            ds = ds.rename_dims({'phony_dim_1': 'Time'})

        ds = ds.rename_vars({'time': 'Time'})
        ds = ds.rename_vars({var: 'SSHA'})
        ds = ds.rename({'lats': 'Latitude'})
        ds = ds.rename({'lons': 'Longitude'})

        ds = ds.drop([var for var in ds.data_vars if var[0] == '_'])
        ds = ds.drop_vars(['ssh'])
        ds = ds.assign_coords(Time=('Time', ds.Time))
        ds = ds.assign_coords(Latitude=ds.Latitude)
        ds = ds.assign_coords(Longitude=ds.Longitude)

        ds.Time.attrs['long_name'] = 'Time'
        ds.Time.attrs['standard_name'] = 'Time'
        adjusted_times = [reference_date + timedelta(seconds=time) for
                          time in ds.Time.values]
        ds = ds.assign_coords(Time=adjusted_times)

        data_start_time = min(
            data_start_time, ds.Time.values[0]) if data_start_time else ds.Time.values[0]
        data_end_time = max(
            data_end_time, ds.Time.values[-1]) if data_end_time else ds.Time.values[-1]

        granules.append(ds)

    # Merge opened granules if needed
    cycle_ds = xr.concat((granules), dim='Time') if len(
        granules) > 1 else granules[0]

    # Time bounds

    # Center time
    data_center_time = data_start_time + ((data_end_time - data_start_time)/2)

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    # Time Attributes
    cycle_ds.Time.attrs['long_name'] = 'Time'

    # Global Attributes
    cycle_ds.attrs = {}
    cycle_ds.attrs['title'] = 'Sea Level Anormaly Estimate based on Altimeter Data'

    cycle_ds.attrs['cycle_start'] = cycle_dates[0]
    cycle_ds.attrs['cycle_center'] = cycle_dates[1]
    cycle_ds.attrs['cycle_end'] = cycle_dates[2]

    cycle_ds.attrs['data_time_start'] = str(data_start_time)[:19]
    cycle_ds.attrs['data_time_center'] = str(data_center_time)[:19]
    cycle_ds.attrs['data_time_end'] = str(data_end_time)[:19]

    cycle_ds.attrs['original_dataset_title'] = ds_metadata['original_dataset_title_s']
    cycle_ds.attrs['original_dataset_short_name'] = ds_metadata['original_dataset_short_name_s']
    cycle_ds.attrs['original_dataset_url'] = ds_metadata['original_dataset_url_s']
    cycle_ds.attrs['original_dataset_reference'] = ds_metadata['original_dataset_reference_s']

    return cycle_ds, len(granules)


def process_1812(cycle_granules, ds_metadata, cycle_dates):
    """
    Performs all processing required for the 1812 dataset
    """
    var = 'SLA'
    granule = cycle_granules[0]

    ds = xr.open_dataset(granule['granule_file_path_s'])

    cycle_ds = ds.rename({var: 'SSHA'})

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    # Global attributes
    cycle_ds.attrs = {}

    cycle_ds.attrs['title'] = 'Sea Level Anormaly Estimate based on Altimeter Data'

    cycle_ds.attrs['cycle_start'] = cycle_dates[0]
    cycle_ds.attrs['cycle_center'] = cycle_dates[1]
    cycle_ds.attrs['cycle_end'] = cycle_dates[2]

    data_time_start = ds.Time_bounds.values[0][0]
    data_time_end = ds.Time_bounds.values[-1][1]
    data_time_center = data_time_start + ((data_time_end - data_time_start)/2)

    cycle_ds.attrs['data_time_start'] = np.datetime_as_string(
        data_time_start, unit='s')
    cycle_ds.attrs['data_time_center'] = np.datetime_as_string(
        data_time_center, unit='s')
    cycle_ds.attrs['data_time_end'] = np.datetime_as_string(
        data_time_end, unit='s')

    cycle_ds.attrs['original_dataset_title'] = ds_metadata['original_dataset_title_s']
    cycle_ds.attrs['original_dataset_short_name'] = ds_metadata['original_dataset_short_name_s']
    cycle_ds.attrs['original_dataset_url'] = ds_metadata['original_dataset_url_s']
    cycle_ds.attrs['original_dataset_reference'] = ds_metadata['original_dataset_reference_s']

    return cycle_ds, 1


def process_shalin(cycle_granules, ds_metadata, cycle_dates):
    """
    Performs all processing required for Shalin's GPS dataset
    """
    def testing(ds, var):
        non_nan_vals = ds[var].values[~np.isnan(ds[var].values)]

        mean = np.nanmean(ds[var].values)
        rms = np.sqrt(np.mean(non_nan_vals**2))
        std = np.std(non_nan_vals)

        try:
            offset, amplitude, _, _ = delta_orbit_altitude_offset_amplitude(
                ds.time, ds.ssha, ds[var])
        except Exception as e:
            print(e)
            offset = 100
            amplitude = 100

        tests = [('Mean', mean), ('RMS', rms), ('STD', std),
                 ('Offset', offset), ('Amplitude', amplitude)]

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
            time_data = (time_da.values - time_da[0].values)/1e9
            time_data = time_data.astype('float')
        else:
            time_data = time_da.values
        # plt.plot(td, delta_orbit_altitude, 'k.')
        # plt.xlabel('time delta (s)')
        # plt.grid()
        # plt.title('delta orbit altitude [m]')
        # calculate omega * t
        omega = 2.*np.pi/6745.756
        omega_t = omega * time_data
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
        offset = c[0]
        amplitude = np.sqrt(c[1]**2 + c[2]**2)
        # estimated time series
        y_e = c[0] + c[1]*np.cos(omega_t) + c[2] * np.sin(omega_t)
        # the c vector will have 3 elements
        return offset, amplitude, delta_orbit_altitude, y_e

    # Flags
    # - DS field names change after a certain date
    # - All possible flag names are used by checking if each flag
    #   is in the keys of a DS
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
    var = 'gps_ssha'
    tests = ['Mean', 'RMS', 'STD', 'Offset', 'Amplitude']
    granules = []
    data_start_time = None
    data_end_time = None

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

        ds = ds.rename_dims({'time': 'Time'})
        ds = ds.rename({'time': 'Time'})
        ds = ds.rename_vars({var: 'SSHA'})

        data_start_time = min(
            data_start_time, ds.Time.values[0]) if data_start_time else ds.Time.values[0]
        data_end_time = max(
            data_end_time, ds.Time.values[-1]) if data_end_time else ds.Time.values[-1]

        granules.append(ds)

     # Merge
    cycle_ds = xr.concat((granules), dim='Time')

    # Center time
    data_center_time = data_start_time + ((data_end_time - data_start_time)/2)

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    # Time Attributes
    cycle_ds.Time.attrs['long_name'] = 'Time'

    # Global Attributes
    cycle_ds.attrs = {}
    cycle_ds.attrs['title'] = 'Ten day aggregated GPSOGDR - Reduced dataset'

    cycle_ds.attrs['cycle_start'] = cycle_dates[0]
    cycle_ds.attrs['cycle_center'] = cycle_dates[1]
    cycle_ds.attrs['cycle_end'] = cycle_dates[2]

    cycle_ds.attrs['data_time_start'] = str(data_start_time)[:19]
    cycle_ds.attrs['data_time_center'] = str(data_center_time)[:19]
    cycle_ds.attrs['data_time_end'] = str(data_end_time)[:19]

    cycle_ds.attrs['original_dataset_title'] = ds_metadata['original_dataset_title_s']
    cycle_ds.attrs['original_dataset_short_name'] = ds_metadata['original_dataset_short_name_s']
    cycle_ds.attrs['original_dataset_url'] = ds_metadata['original_dataset_url_s']
    cycle_ds.attrs['original_dataset_reference'] = ds_metadata['original_dataset_reference_s']

    return cycle_ds, len(granules)


def processing(config_path='', output_path=''):

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    version = config['version']
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    processor = config['processor']
    date_regex = '%Y-%m-%dT%H:%M:%S'
    solr_regex = f'{date_regex}Z'

    # Query for dataset metadata
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    ds_metadata = solr_query(config, fq)[0]

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{dataset_name}']
    solr_cycles = solr_query(config, fq)

    cycles = {cycle['start_date_dt']: cycle for cycle in solr_cycles}

    # Generate list of cycle date tuples (start, end)
    cycle_dates = []

    start_date = datetime.strptime('1992-01-01T00:00:00', date_regex)
    end_date = datetime.utcnow()
    delta = timedelta(days=10)

    curr = start_date
    while curr < end_date:

        if datetime.strftime(curr + delta, date_regex) >= ds_metadata['start_date_dt']:
            cycle_dates.append((curr, curr + delta))
        if datetime.strftime(curr + delta, date_regex) > ds_metadata['end_date_dt']:
            break

        curr += delta

    for (start_date, end_date) in cycle_dates:

        # Make strings for cycle start, center, and end dates
        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)
        center_date = start_date + ((end_date - start_date)/2)
        center_date_str = datetime.strftime(center_date, date_regex)

        date_strs = (start_date_str, center_date_str, end_date_str)

        # Find the granule with date closest to center of cycle
        # Uses special Solr query function to automatically return granules in proximal order
        if '1812' in dataset_name:
            query_start = datetime.strftime(start_date, solr_regex)
            query_end = datetime.strftime(end_date, solr_regex)
            fq = ['type_s:harvested', f'dataset_s:{dataset_name}', 'harvest_success_b:true',
                  f'date_dt:[{query_start} TO {query_end}}}']
            boost_function = f'recip(abs(ms({center_date_str}Z,date_dt)),3.16e-11,1,1)'

            query_params = {'q': '*:*',
                            'fq': fq,
                            'bf': boost_function,
                            'defType': 'edismax',
                            'rows': 300000,
                            'sort': 'date_s asc'}

            url = f'{solr_host}{solr_collection_name}/select?'
            response = requests.get(url, params=query_params)
            cycle_granules = response.json()['response']['docs']

        # Get granules within start_date and end_date
        else:
            query_start = datetime.strftime(start_date, solr_regex)
            query_end = datetime.strftime(end_date, solr_regex)
            fq = ['type_s:harvested', f'dataset_s:{dataset_name}', 'harvest_success_b:true',
                  f'date_dt:[{query_start} TO {query_end}}}']

            cycle_granules = solr_query(config, fq)

        # Skip cycle if no granules harvested
        if not cycle_granules:
            print(f'No granules for cycle {start_date_str} to {end_date_str}')
            continue

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

                for granule in cycle_granules:
                    if prior_time < granule['modified_time_dt']:
                        updating = True
                        continue
            else:
                updating = True
        else:
            updating = True

        if updating:
            processing_success = False
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            funcs = {'1812': process_1812,
                     'along_track': process_along_track,
                     'shalin': process_shalin}

            try:
                cycle_ds, granule_count = funcs[processor](cycle_granules,
                                                           ds_metadata,
                                                           date_strs)

                # NetCDF encoding
                encoding_each = {'zlib': True,
                                 'complevel': 5,
                                 'dtype': 'float32',
                                 'shuffle': True,
                                 '_FillValue': default_fillvals['f8']}

                coord_encoding = {}
                for coord in cycle_ds.coords:

                    if 'Time' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'zlib': True,
                                                 'contiguous': False,
                                                 'calendar': 'gregorian',
                                                 'shuffle': False}

                    # To account for time bounds in 1812 dataset
                    if 'Time' in coord and '1812' in dataset_name:
                        units_time = datetime.strftime(
                            center_date, "%Y-%m-%d %H:%M:%S")
                        coord_encoding[coord]['units'] = f'days since {units_time}'

                    if 'Lat' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}
                    if 'Lon' in coord:
                        coord_encoding[coord] = {'_FillValue': None,
                                                 'dtype': 'float32'}

                var_encoding = {
                    var: encoding_each for var in cycle_ds.data_vars}

                encoding = {**coord_encoding, **var_encoding}

                filename_time = datetime.strftime(center_date, '%Y%m%dT%H%M%S')

                filename = f'ssha_{filename_time}.nc'

                save_dir = f'{output_path}{dataset_name}/cycle_products/'
                save_path = f'{save_dir}{filename}'

                # If paths don't exist, make them
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save to netcdf
                cycle_ds.to_netcdf(save_path, encoding=encoding)
                checksum = md5(save_path)
                file_size = os.path.getsize(save_path)
                processing_success = True

            except Exception as e:
                print(e)
                filename = ''
                save_path = ''
                checksum = ''
                file_size = 0
                granule_count = 0

            # Add or update Solr cycle
            # Update cycle entry
            if start_date_str + 'Z' in cycles.keys():
                item = cycles[start_date_str + 'Z']
                item['granules_in_cycle_i'] = {"set": granule_count}
                item['filename_s'] = {"set": filename}
                item['filepath_s'] = {"set": save_path}
                item['checksum_s'] = {"set": checksum}
                item['file_size_l'] = {"set": file_size}
                item['processing_success_b'] = {"set": processing_success}
                item['processing_time_dt'] = {
                    "set": datetime.utcnow().strftime(solr_regex)}
                item['processing_version_f'] = {"set": version}
            # Make new entry
            else:
                item = {}
                item['type_s'] = 'cycle'
                item['dataset_s'] = dataset_name
                item['start_date_dt'] = start_date_str
                item['center_date_dt'] = center_date_str
                item['end_date_dt'] = end_date_str
                item['granules_in_cycle_i'] = granule_count
                item['filename_s'] = filename
                item['filepath_s'] = save_path
                item['checksum_s'] = checksum
                item['file_size_l'] = file_size
                item['processing_success_b'] = processing_success
                item['processing_time_dt'] = datetime.utcnow().strftime(date_regex)
                item['processing_version_f'] = version

            resp = solr_update(config, [item])
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give harvested documents the id of the corresponding cycle document
                if processing_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle', f'dataset_s:{dataset_name}',
                              f'filename_s:{filename}']
                        cycle_doc = solr_query(config, fq)
                        cycle_id = cycle_doc[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    resp = solr_update(config, cycle_granules)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

    # Query for Solr failed harvest documents
    fq = ['type_s:cycle',
          f'dataset_s:{dataset_name}', 'processing_success_b:false']
    failed_processing = solr_query(config, fq)

    if not failed_processing:
        processing_status = 'All cycles successfully processed'
    else:
        # Query for Solr successful harvest documents
        fq = ['type_s:cycle',
              f'dataset_s:{dataset_name}', 'processing_success_b:true']
        successful_processing = solr_query(config, fq)

        if not successful_processing:
            processing_status = 'No cycles successfully processed (either all failed or no granules to process)'
        else:
            processing_status = f'{len(failed_processing)} cycles failed'

    ds_metadata['processing_status_s'] = {"set": processing_status}
    resp = solr_update(config, [ds_metadata])

    if resp.status_code == 200:
        print('Successfully updated Solr dataset document\n')
    else:
        print('Failed to update Solr dataset document\n')
