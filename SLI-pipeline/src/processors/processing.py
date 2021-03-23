import os
import hashlib
from datetime import datetime, timedelta
import yaml
import requests
import numpy as np
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module


def md5(fpath):
    """
    Creates md5 checksum from file

    Params:
        fpath (str): path of the file

    Returns:
        hash_md5.hexdigest (str): double length string containing only hexadecimal digits
    """
    hash_md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq):
    """
    Queries Solr database using the filter query passed in.

    Params:
        config (dict): the dataset specific config file
        fq (List[str]): the list of filter query arguments

    Returns:
        response.json()['response']['docs'] (List[dict]): the Solr docs that satisfy the query
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
    Updates Solr database with list of docs. If a doc contains an existing id field,
    Solr will update or replace that existing doc with the new doc.

    Params:
        config (dict): the dataset specific config file
        update_body (List[dict]): the list of docs to update on Solr

    Returns:
        requests.post(url, json=update_body) (Response): the Response object from the post call
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


def process_along_track(cycle_granules, ds_meta, dates):
    """
    Processes and aggregates individual granules that fall within a cycle's date range for
    a non GPS along track dataset.

    Params:
        cycle_granules (List[dict]): the dataset specific config file
        ds_meta (dict): the list of docs to update on Solr
        dates (Tuple[str, str, str]):

    Returns:
        cycle_ds (Dataset): the processed cycle Dataset object
        len(granules) (int): the number of granules within the processed cycle Dataset object
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

        ds = ds.rename_vars({'time': 'Time', var: 'SSHA'})
        ds = ds.rename({'lats': 'Latitude', 'lons': 'Longitude'})

        ds = ds.drop([var for var in ds.data_vars if var[0] == '_'])
        ds = ds.drop_vars(['ssh'])
        ds = ds.assign_coords(Time=('Time', ds.Time))
        ds = ds.assign_coords(Latitude=ds.Latitude)
        ds = ds.assign_coords(Longitude=ds.Longitude)

        ds.Time.attrs['long_name'] = 'Time'
        ds.Time.attrs['standard_name'] = 'Time'
        adjusted_times = [reference_date + timedelta(seconds=time) for time in ds.Time.values]
        ds = ds.assign_coords(Time=adjusted_times)

        data_start_time = min(
            data_start_time, ds.Time.values[0]) if data_start_time else ds.Time.values[0]
        data_end_time = max(
            data_end_time, ds.Time.values[-1]) if data_end_time else ds.Time.values[-1]

        granules.append(ds)

    # Merge opened granules if needed
    cycle_ds = xr.concat((granules), dim='Time') if len(granules) > 1 else granules[0]

    # Time bounds

    # Center time
    data_center_time = data_start_time + ((data_end_time - data_start_time)/2)

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    # Time Attributes
    cycle_ds.Time.attrs['long_name'] = 'Time'

    # Global Attributes
    cycle_ds.attrs = {
        'title': 'Sea Level Anormaly Estimate based on Altimeter Data',
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'data_time_start': str(data_start_time)[:19],
        'data_time_center': str(data_center_time)[:19],
        'data_time_end': str(data_end_time)[:19],
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, len(granules)


def process_measures_grids(cycle_granules, ds_meta, dates):
    """
    Processes and aggregates individual granules that fall within a cycle's date range for
    measures grids datasets (1812).

    Params:
        cycle_granules (List[dict]): the dataset specific config file
        ds_meta (dict): the list of docs to update on Solr
        dates (Tuple[str, str, str]):

    Returns:
        cycle_ds (Dataset): the processed cycle Dataset object
        1 (int): the number of granules within the processed cycle Dataset object
    """
    var = 'SLA'
    granule = cycle_granules[0]

    ds = xr.open_dataset(granule['granule_file_path_s'])

    cycle_ds = ds.rename({var: 'SSHA'})

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    data_time_start = cycle_ds.Time_bounds.values[0][0]
    data_time_end = cycle_ds.Time_bounds.values[-1][1]
    data_time_center = data_time_start + ((data_time_end - data_time_start)/2)

    # Global attributes
    cycle_ds.attrs = {
        'title': 'Sea Level Anormaly Estimate based on Altimeter Data',
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'data_time_start': np.datetime_as_string(data_time_start, unit='s'),
        'data_time_center': np.datetime_as_string(data_time_center, unit='s'),
        'data_time_end': np.datetime_as_string(data_time_end, unit='s'),
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, 1


def process_gps(cycle_granules, ds_meta, dates):
    """
    Processes and aggregates individual granules that fall within a cycle's date range for
    Shalin's gps along track dataset.

    Dataset's netCDF files change format from 2020-10-29 onward. Groups are used beyond that date.

    Params:
        cycle_granules (List[dict]): the dataset specific config file
        ds_meta (dict): the list of docs to update on Solr
        dates (Tuple[str, str, str]):

    Returns:
        cycle_ds (Dataset): the processed cycle Dataset object
        1 (int): the number of granules within the processed cycle Dataset object
    """
    def testing(ds, var):
        """
        Performs mean, rms, std, offset, and amplitude tests for a granule and includes
        their results as individual Dataarrays within the Dataset object.

        Params:
            ds (Dataset): data granule
            var (str): the datavar on which to perform the tests

        Returns:
            ds (Dataset): Dataset object with testing Dataarrays added

        """
        non_nan_vals = ds[var].values[~np.isnan(ds[var].values)]

        mean = np.nanmean(ds[var].values)
        rms = np.sqrt(np.mean(non_nan_vals**2))
        std = np.std(non_nan_vals)

        try:
            offset, amplitude = delta_orbit_altitude_offset_amplitude(ds.time, ds.ssha, ds[var])

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
        """
        delta orbit altitude between the GPS orbit altitude
        and DORIS orbit altitude
        ssh = orbit_altitude - range - corrections

        in Shailen's SSH files, there is both 'gps_ssha', and 'ssha'

        gps_ssha - ssha = delta orbit_altitude
           because range and corrections are the same for both ssha fields

        therefore we seek a solution the following equation
        y = C0 +  C1 cos(omega t) + C2 sin(omega t)

        where
        y      : delta GPS orbit altitude
        C0     : OFFSET
        C1, C2 : AMPLITUDES of cosine and sine terms comprising a phase shifted oscillation
        omega  : period of one orbit resolution in seconds
        t      : time in seconds

        Params:
            time_da (DataArray): time DataArray from the Dataset object being tested
            ssha_da (DataArray): ssha DataArray from the Dataset object being tested
            gps_ssha_da (DataArray): gps adjusted ssha DataArray from the Dataset object being tested

        Returns:
            offset (float):
            amplitude (float):

        """

        # calculate delta orbit altitude
        delta_orbit_altitude = gps_ssha_da.values - ssha_da.values

        # calculate time (in seconds) from the first to last observations in record
        if isinstance(time_da.values[0], np.datetime64):
            time_data = (time_da.values - time_da[0].values)/1e9
            time_data = time_data.astype('float')
        else:
            time_data = time_da.values

        # calculate omega * t
        omega = 2.*np.pi/6745.756
        omega_t = omega * time_data

        # pull values of omega_t and the delta_orbit_altitude only where
        # the delta_orbit_altitude is not nan (i.e., not missing)
        omega_t_nn = omega_t[~np.isnan(delta_orbit_altitude)]
        delta_orbit_altitude_nn = delta_orbit_altitude[~np.isnan(delta_orbit_altitude)]

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
        c = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), delta_orbit_altitude_nn.T)

        offset = c[0]
        amplitude = np.sqrt(c[1]**2 + c[2]**2)

        return offset, amplitude

    # List of flags to use for data masking
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

        ds = xr.open_dataset(granule['granule_file_path_s'])

        if 'lon' in ds.coords:
            ds = ds.rename({'lon': 'Longitude', 'lat': 'Latitude'})
            ds[var].encoding['coordinates'] = 'Longitude Latitude'
            ds_keys = list(ds.keys())
        else:
            uses_groups = True

            ds = xr.open_dataset(granule['granule_file_path_s'], group='data_01/ku')
            ds_flags = xr.open_dataset(granule['granule_file_path_s'], group='data_01')
            ds_flags = ds_flags.rename({'longitude': 'Longitude', 'latitude': 'Latitude'})
            ds = ds.assign_coords({"Longitude": ds_flags.Longitude, "Latitude": ds_flags.Latitude})

            ds_keys = list(ds_flags.keys())

        # Remove outliers before running tests
        ds[var].values[np.greater(abs(ds[var].values), 1.5, where=~
                                  np.isnan(ds[var].values))] = np.nan

        # Run tests, returns byte results convert to int
        ds = testing(ds, var)

        # Mask out flagged data
        for flag in flags:
            if flag in ds_keys:
                if uses_groups:
                    if np.isnan(ds_flags[flag].values).all():
                        continue

                    ds[var].values = np.where(ds_flags[flag].values == 0, ds[var].values,
                                              default_fillvals['f8'])
                else:
                    if np.isnan(ds[flag].values).all():
                        continue
                    ds[var].values = np.where(ds[flag].values == 0, ds[var].values,
                                              default_fillvals['f8'])

        # Replace nans with fill value
        ds[var].values = np.where(np.isnan(ds[var].values), default_fillvals['f8'], ds[var].values)

        ds = ds.drop([key for key in ds.keys() if key not in tests + [var]])

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
    cycle_ds.attrs = {
        'title': 'Ten day aggregated GPSOGDR - Reduced dataset',
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'data_time_start': str(data_start_time)[:19],
        'data_time_center': str(data_center_time)[:19],
        'data_time_end': str(data_end_time)[:19],
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, len(granules)


def collect_granules(ds_name, dates, date_strs, config):
    """
    Collects granules that fall within a cycle's date range.
    The measures gridded dataset (1812) needs to only select the single granule closest
    to the center datetime of the cycle.

    Params:
        ds_name (str): the name of the dataset
        dates (Tuple[datetime, datetime, datetime]): the start, center, and end of the cycle 
                                                     in datetime format
        date_strs (Tuple[str, str, str]): the start, center, and end of the cycle in string format
        config (dict): the dataset specific config file


    Returns:
        cycle_granules (List[dict]): the Solr docs that satisfy the query
    """
    solr_regex = '%Y-%m-%dT%H:%M:%SZ'
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    # Find the granule with date closest to center of cycle
    # Uses special Solr query function to automatically return granules in proximal order
    if '1812' in ds_name:
        query_start = datetime.strftime(dates[0], solr_regex)
        query_end = datetime.strftime(dates[2], solr_regex)
        fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true',
              f'date_dt:[{query_start} TO {query_end}}}']
        boost_function = f'recip(abs(ms({date_strs[1]}Z,date_dt)),3.16e-11,1,1)'

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
        query_start = datetime.strftime(dates[0], solr_regex)
        query_end = datetime.strftime(dates[2], solr_regex)
        fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true',
              f'date_dt:[{query_start} TO {query_end}}}']

        cycle_granules = solr_query(config, fq)

    return cycle_granules


def check_updating(cycles, date_strs, cycle_granules, version):
    """
    Checks whether a cycle requires reprocessing based on three conditions:
    - If the prior processing attempt failed
    - If the prior processing version differs from the current processing version
    - If any of the granules within the cycle date range have been modified since
        the prior processing attempt
    If the cycle has not previously been processed, check_updating returns True.

    Params:
        cycles (dict): the existing cycles on Solr in dictionary format where the key is
                        the start date string
        date_strs (Tuple[str, str, str]): the start, center, and end of the cycle in string format
        cycle_granules (List[dict]): the granules that make up the cycle
        version (float): the processing version number as defined in the dataset's config file


    Returns:
        (bool): whether or not the cycle requires reprocessing
    """

    # Cycles dict uses the PODAAC date format (with a trailing 'Z')
    if date_strs[0] + 'Z' in cycles.keys():
        existing_cycle = cycles[date_strs[0] + 'Z']

        prior_time = existing_cycle['processing_time_dt']
        prior_success = existing_cycle['processing_success_b']
        prior_version = existing_cycle['processing_version_f']

        if not prior_success or prior_version != version:
            return True

        for granule in cycle_granules:
            if prior_time < granule['modified_time_dt']:
                return True

        return False

    return True


def cycle_ds_encoding(cycle_ds, ds_name, center_date):
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    The measures gridded dataset (1812) has additional units encoding requirements.

    Params:
        cycle_ds (Dataset): the Dataset object
        ds_name (str): the name of the dataset (used to check if dataset is 1812)
        center_date (datetime): used to set the units encoding in the 1812 dataset

    Returns:
        encoding (dict): the encoding dictionary for the cycle_ds Dataset object
    """

    var_encoding = {'zlib': True,
                    'complevel': 5,
                    'dtype': 'float32',
                    'shuffle': True,
                    '_FillValue': default_fillvals['f8']}
    var_encodings = {var: var_encoding for var in cycle_ds.data_vars}

    coord_encoding = {}
    for coord in cycle_ds.coords:
        if 'Time' in coord:
            coord_encoding[coord] = {'_FillValue': None,
                                     'zlib': True,
                                     'contiguous': False,
                                     'calendar': 'gregorian',
                                     'shuffle': False}
            # To account for time bounds in 1812 dataset
            if '1812' in ds_name:
                units_time = datetime.strftime(center_date, "%Y-%m-%d %H:%M:%S")
                coord_encoding[coord]['units'] = f'days since {units_time}'

        if 'Lat' in coord or 'Lon' in coord:
            coord_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}

    encoding = {**coord_encoding, **var_encodings}
    return encoding


def post_process_solr_update(config, ds_metadata):
    """
    Determines processing status by number of failed and successful cycle documents on Solr.
    Updates dataset document on Solr with status message

    Params:
        config (dict): the dataset's config file
        ds_metadata (dict): the dataset metadata document from Solr
    """
    ds_name = config['ds_name']

    processing_status = 'All cycles successfully processed'

    # Query for failed cycle documents
    fq = ['type_s:cycle', f'dataset_s:{ds_name}', 'processing_success_b:false']
    failed_processing = solr_query(config, fq)

    if failed_processing:
        # Query for successful cycle documents
        fq = ['type_s:cycle', f'dataset_s:{ds_name}', 'processing_success_b:true']
        successful_processing = solr_query(config, fq)

        processing_status = 'No cycles successfully processed (all failed or no granules to process)'

        if successful_processing:
            processing_status = f'{len(failed_processing)} cycles failed'

    ds_metadata['processing_status_s'] = {"set": processing_status}
    resp = solr_update(config, [ds_metadata])

    if resp.status_code == 200:
        print('Successfully updated Solr dataset document\n')
    else:
        print('Failed to update Solr dataset document\n')


def processing(config_path, output_path, reprocess):
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    The measures gridded dataset (1812) has additional units encoding requirements.

    Params:
        config_path (str): path to the dataset's config file
        output_path (str): path to the pipeline's output directory
        reprocess (bool): denotes if all cycles should be reprocessed
    """

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    ds_name = config['ds_name']
    version = config['version']
    processor = config['processor']
    date_regex = '%Y-%m-%dT%H:%M:%S'

    # Query for dataset metadata
    ds_metadata = solr_query(config, ['type_s:dataset', f'dataset_s:{ds_name}'])[0]

    # Query for all existing cycles in Solr
    solr_cycles = solr_query(config, ['type_s:cycle', f'dataset_s:{ds_name}'])

    cycles = {cycle['start_date_dt']: cycle for cycle in solr_cycles}

    # Generate list of cycle date tuples (start, end)
    delta = timedelta(days=10)
    start_date = datetime.strptime('1992-01-01T00:00:00', date_regex)
    end_date = start_date + delta

    while True:
        # Make strings for cycle start, center, and end dates
        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)
        center_date = start_date + ((end_date - start_date)/2)
        center_date_str = datetime.strftime(center_date, date_regex)

        dates = (start_date, center_date, end_date)
        date_strs = (start_date_str, center_date_str, end_date_str)

        # End cycle processing if cycles are outside of dataset date range
        if start_date_str > ds_metadata['end_date_dt']:
            break

        # Move to next cycle date range if end of cycle is before start of dataset
        if end_date_str < ds_metadata['start_date_dt']:
            start_date = end_date
            end_date = start_date + delta
            continue

        # ======================================================
        # Collect granules within cycle
        # ======================================================

        cycle_granules = collect_granules(ds_name, dates, date_strs, config)

        # Skip cycle if no granules harvested
        if not cycle_granules:
            print(f'No granules for cycle {start_date_str} to {end_date_str}')
            continue

        # ======================================================
        # Determine if cycle requires processing
        # ======================================================

        if reprocess or check_updating(cycles, date_strs, cycle_granules, version):
            processing_success = False
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            funcs = {'measures_grids': process_measures_grids,
                     'along_track': process_along_track,
                     'gps': process_gps}

            # ======================================================
            # Process the cycle
            # ======================================================

            try:
                # Dataset specific processing of cycle
                cycle_ds, granule_count = funcs[processor](cycle_granules, ds_metadata, date_strs)

                # Create netcdf encoding for cycle
                encoding = cycle_ds_encoding(cycle_ds, ds_name, center_date)

                # Save to netcdf
                filename_time = datetime.strftime(center_date, '%Y%m%dT%H%M%S')
                filename = f'ssha_{filename_time}.nc'

                save_dir = f'{output_path}{ds_name}/cycle_products/'
                save_path = f'{save_dir}{filename}'

                # If paths don't exist, make them
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cycle_ds.to_netcdf(save_path, encoding=encoding)

                # Determine checksum and file size
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
            item = {
                'type_s': 'cycle',
                'dataset_s': ds_name,
                'start_date_dt': start_date_str,
                'center_date_dt': center_date_str,
                'end_date_dt': end_date_str,
                'granules_in_cycle_i': granule_count,
                'filename_s': filename,
                'filepath_s': save_path,
                'checksum_s': checksum,
                'file_size_l': file_size,
                'processing_success_b': processing_success,
                'processing_time_dt': datetime.utcnow().strftime(date_regex),
                'processing_version_f': version
            }

            if start_date_str + 'Z' in cycles.keys():
                item['id'] = cycles[start_date_str + 'Z']['id']

            resp = solr_update(config, [item])
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give granule documents the id of the corresponding cycle document
                if processing_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle', f'dataset_s:{ds_name}', f'filename_s:{filename}']
                        cycle_id = solr_query(config, fq)[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    resp = solr_update(config, cycle_granules)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

        start_date = end_date
        end_date = start_date + delta

    # ======================================================
    # Update dataset document with overall processing status
    # ======================================================
    post_process_solr_update(config, ds_metadata)
