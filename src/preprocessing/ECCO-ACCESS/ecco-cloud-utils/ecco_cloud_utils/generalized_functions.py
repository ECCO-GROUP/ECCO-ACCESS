# -*- coding: utf-8 -*-

import ecco_cloud_utils as ea
import numpy as np
import pyresample as pr
import xarray as xr
from pathlib import Path
from datetime import datetime
import sys
import os

# ECCO ACCESS LIBRARY
# https://github.com/ECCO-GROUP/ECCO-ACCESS
ecco_cloud_util_path = Path('../')
sys.path.append(str(ecco_cloud_util_path))


# %%
# return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)
def generalized_grid_product(product_name,
                             data_res,
                             data_max_lat,
                             area_extent,
                             dims,
                             proj_info):

    # minimum Length of data product grid cells (km)
    source_grid_min_L = np.cos(np.deg2rad(data_max_lat))*data_res*112e3

    # maximum length of data roduct grid cells (km)
    # data product at equator has grid spacing of data_res*112e3 m
    source_grid_max_L = data_res*112e3

    areaExtent = (area_extent[0], area_extent[1],
                  area_extent[2], area_extent[3])

    # Corressponds to resolution of grid from data
    cols = dims[0]
    rows = dims[1]

    # USE PYRESAMPLE TO GENERATE THE LAT/LON GRIDS
    # -- note we do not have to use pyresample for this, we could
    # have created it manually using the np.meshgrid or some other method
    # if we wanted.
    tmp_data_grid = pr.area_config.get_area_def(proj_info['area_id'], proj_info['area_name'],
                                                proj_info['proj_id'], proj_info['proj4_args'],
                                                cols, rows, areaExtent)

    data_grid_lons, data_grid_lats = tmp_data_grid.get_lonlats()

    # Changes longitude bounds from 0-360 to -180-180, doesnt change if its already -180-180
    data_grid_lons, data_grid_lats = pr.utils.check_and_wrap(data_grid_lons,
                                                             data_grid_lats)

    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the source observation grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    source_grid = \
        pr.geometry.SwathDefinition(lons=data_grid_lons.ravel(),
                                    lats=data_grid_lats.ravel())

    return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)
# %%

# %%
# return dates_in_year_iso, daily_paths


def generalized_get_data_filepaths_for_year(year, data_dir, data_file_suffix,
                                            data_time_scale, date_format):

    # check data_time_scale and date_format are supported
    supported_time_scales = ['monthly', 'daily']
    supported_date_formats = ['yyyymm', 'yyyy_mm', 'yyyyddd', 'yyyymmdd']
    supported_pairings = ['monthly-yyyymm', 'monthly-yyyy_mm', 'monthly-yyyyddd',
                          'daily-yyyyddd', 'daily-yyyymmdd']
    input_is_unsupported = False
    if data_time_scale not in supported_time_scales:
        print(f'unsupported data time scale: {data_time_scale}')
        print(f'supported data time scales: {supported_time_scales}')
        input_is_unsupported = True
    if date_format not in supported_date_formats:
        print(f'unsupported date format: {date_format}')
        print(f'supported date formats: {supported_date_formats}')
        input_is_unsupported = True
    if not input_is_unsupported:
        if f'{data_time_scale}-{date_format}' not in supported_pairings:
            print(f'unsupported date pairing: {data_time_scale}-{date_format}')
            print(f'supported date pairings: {supported_pairings}')

    # make empty dictionary
    daily_paths = dict()

    # find all etcdf files in this directory that have the year and suffix
    all_netcdf_files_year = np.sort(
        list(data_dir.glob(f'**/*{year}*{data_file_suffix}')))

    dates_in_year = []
    if data_time_scale == 'monthly':
        for i in range(1, 13):
            if date_format == 'yyyymm':
                dates_in_year.append(str(year) + f'{i:02d}')
            elif date_format == 'yyyy_mm':
                dates_in_year.append(str(year) + '_' + f'{i:02d}')
            elif date_format == 'yyyyddd':
                dates_in_year.append(str(year) + f'{i:02d}')
    elif data_time_scale == 'daily':
        if date_format == 'yyyyddd':
            for i in range(1, 367):
                dates_in_year.append(str(year) + f'{i:03d}')
        elif date_format == 'yyyymmdd':
            dates_in_year = \
                np.arange(str(year) + '-01-01', str(year+1) +
                          '-01-01', dtype='datetime64[D]')

    # make empty list that will contain the dates in this year in iso format
    # yyyy-mm-dd
    dates_in_year_iso = []
    # loop through every day in the year
    for date in dates_in_year:
        if data_time_scale == 'monthly':
            if date_format == 'yyyymm':
                date_str_iso_obj = datetime.strptime(date, '%Y%m')
                date_str = date
            elif date_format == 'yyyy_mm':
                date_str_iso_obj = datetime.strptime(date, '%Y_%m')
                date_str = date
            elif date_format == 'yyyyddd':
                date_str_iso_obj = datetime.strptime(date, '%Y%m')
                date_str = datetime.strftime(date_str_iso_obj, '%Y%j')
        elif data_time_scale == 'daily':
            if date_format == 'yyyyddd':
                print('TODO: implement daily yyyyddd date support')
                date_str = date
            elif date_format == 'yyyymmdd':
                date_str = str(date.tolist().year) + \
                    str(date.tolist().month).zfill(2) + \
                    str(date.tolist().day).zfill(2)
                date_str_iso_obj = datetime.strptime(date_str, '%Y%m%d')

        # add iso format date to dates_in_year_iso
        date_str_iso = datetime.strftime(date_str_iso_obj, '%Y-%m-%d')
        dates_in_year_iso.append(date_str_iso)

        # find the path that matches this day
        paths_date = []
        # Extracting date from path
        # Need to isolate date from the data name and excess information at the end
        for netcdf_path in all_netcdf_files_year:
            if str.find(str(netcdf_path), date_str) >= 0:
                paths_date = netcdf_path

        # add this path to the dictionary with the date_str_iso as the key
        daily_paths[date_str_iso] = paths_date

        # post to solr

    # return the dates in iso format, and the paths
    return dates_in_year_iso, daily_paths
# %%

# %%
# return data_DA


def generalized_transform_to_model_grid_solr(data_field_info, record_date, model_grid,
                                             model_grid_type, array_precision,
                                             record_file_name, original_dataset_metadata,
                                             extra_information, ds, factors,
                                             time_zone_included_with_time,
                                             model_grid_name):
    # initialize notes for this record
    record_notes = ''

    source_indices_within_target_radius_i, \
        num_source_indices_within_target_radius_i, \
        nearest_source_index_to_target_index_i = factors

    # set data info values
    data_field = data_field_info['name_s']
    standard_name = data_field_info['standard_name_s']
    long_name = data_field_info['long_name_s']
    units = data_field_info['units_s']

    # create empty data array
    data_DA = ea.make_empty_record(standard_name, long_name, units,
                                   record_date,
                                   model_grid, model_grid_type,
                                   array_precision, variable_name=data_field)

    print(data_DA)

    # add some metadata to the newly formed data array object
    data_DA.attrs['original_filename'] = record_file_name
    data_DA.attrs['original_field_name'] = data_field
    data_DA.attrs['interplation_parameters'] = 'bin averaging'
    data_DA.attrs['interplation_code'] = 'pyresample'
    data_DA.attrs['interpolation_date'] = \
        str(np.datetime64(datetime.now(), 'D'))

    data_DA.time.attrs['long_name'] = 'center time of averaging period'

    data_DA.attrs['original_dataset_title'] = original_dataset_metadata['original_dataset_title_s']
    data_DA.attrs['original_dataset_short_name'] = original_dataset_metadata['original_dataset_short_name_s']
    data_DA.attrs['original_dataset_url'] = original_dataset_metadata['original_dataset_url_s']
    data_DA.attrs['original_dataset_reference'] = original_dataset_metadata['original_dataset_reference_s']
    data_DA.attrs['original_dataset_doi'] = original_dataset_metadata['original_dataset_doi_s']
    data_DA.attrs['interpolated_grid_id'] = model_grid_name
    data_DA.name = f'{data_field}_interpolated_to_{model_grid_name}'

    # load the file, do the mapping and update the record times
    print('reading ', record_file_name)

    if 'transpose' in extra_information:
        orig_data = ds[data_field].values[0, :].T
    else:
        orig_data = ds[data_field].values

    # see if we have any valid data
    if np.sum(~np.isnan(orig_data)) > 0:

        data_model_projection = ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                            num_source_indices_within_target_radius_i,
                                                            nearest_source_index_to_target_index_i,
                                                            orig_data, model_grid.XC.shape)

        # put the new data values into the data_DA array.
        # --where the mapped data are not nan, replace the original values
        # --where they are nan, just leave the original values alone
        data_DA.values = np.where(~np.isnan(data_model_projection),
                                  data_model_projection, data_DA.values)

    else:
        print('file loaded but empty')
        record_notes = record_notes + ' -- empty record -- '

    # update time values
    if 'time_bounds_var' in extra_information:
        data_DA.time_start.values[0] = ds.Time_bounds[0][0].values
        data_DA.time_end.values[0] = ds.Time_bounds[0][1].values
    elif 'no_time' in extra_information:
        data_DA.time_start.values[0] = record_date
        data_DA.time_end.values[0] = record_date
    elif 'no_time_dashes' in extra_information:
        new_start_time = f'{ds.time_coverage_start[0:4]}-{ds.time_coverage_start[4:6]}-{ds.time_coverage_start[6:8]}'
        new_end_time = f'{ds.time_coverage_end[0:4]}-{ds.time_coverage_end[4:6]}-{ds.time_coverage_end[6:8]}'
        data_DA.time_start.values[0] = new_start_time
        data_DA.time_end.values[0] = new_end_time
    elif time_zone_included_with_time:
        data_DA.time_start.values[0] = ds.time_coverage_start[:-1]
        data_DA.time_end.values[0] = ds.time_coverage_end[:-1]
    else:
        data_DA.time_start.values[0] = ds.time_coverage_start
        data_DA.time_end.values[0] = ds.time_coverage_end

    if 'time_var' in extra_information:
        data_DA.time.values[0] = ds.Time[0].values
    else:
        data_DA.time.values[0] = record_date

    data_DA.attrs['notes'] = record_notes

    return data_DA


# %%
# return num_files_saved
def generalized_transform_to_model_grid(source_indices_within_target_radius_i,
                                        num_source_indices_within_target_radius_i,
                                        nearest_source_index_to_target_index_i,
                                        model_grid, model_grid_type,
                                        record_date, record_filepath,
                                        data_fields,
                                        array_precision,
                                        time_zone_included_with_time,
                                        extra_information,
                                        new_filename,
                                        fill_values,
                                        output_dirs,
                                        binary_dtype,
                                        new_data_attr,
                                        save_binary,
                                        save_netcdf):

    num_files_saved = 0

    for data_field_info in data_fields:
        # initialize notes for this record
        record_notes = ''

        # set data info values
        data_field = data_field_info['name']
        standard_name = data_field_info['standard_name']
        long_name = data_field_info['long_name']
        units = data_field_info['units']

        # create empty data array
        data_DA = ea.make_empty_record(standard_name, long_name, units,
                                       record_date,
                                       model_grid, model_grid_type,
                                       array_precision)

        # add some metadata to the newly formed data array object
        data_DA.attrs['original_filename'] = record_filepath.name
        data_DA.attrs['original_field_name'] = data_field
        data_DA.attrs['interplation_parameters'] = 'bin averaging'
        data_DA.attrs['interplation_code'] = 'pyresample'
        data_DA.attrs['interpolation_date'] = \
            str(np.datetime64(datetime.now(), 'D'))

        data_DA.time.attrs['long_name'] = 'center time of averaging period'

        data_DA.attrs['original_dataset_title'] = new_data_attr['original_dataset_title']
        data_DA.attrs['original_dataset_url'] = new_data_attr['original_dataset_url']
        data_DA.attrs['original_dataset_reference'] = new_data_attr['original_dataset_reference']
        data_DA.attrs['original_dataset_product_id'] = new_data_attr['original_dataset_product_id']
        data_DA.attrs['interpolated_grid_id'] = new_data_attr['interpolated_grid_id']
        model_grid_id = new_data_attr['interpolated_grid_id']
        data_DA.name = f'{data_field}_interpolated_to_{model_grid_id}'

        # load the file, do the mapping and update the record times
        if record_filepath.is_file() == True:

            print('reading ', record_filepath.name)

            ds = xr.open_dataset(record_filepath, decode_times=True)

            if 'transpose' in extra_information:
                orig_data = ds[data_field].values[0, :].T
            else:
                orig_data = ds[data_field].values

            # see if we have any valid data
            if np.sum(~np.isnan(orig_data)) > 0:

                data_model_projection = ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                                    num_source_indices_within_target_radius_i,
                                                                    nearest_source_index_to_target_index_i,
                                                                    orig_data, model_grid.XC.shape)

                # put the new data values into the data_DA array.
                # --where the mapped data are not nan, replace the original values
                # --where they are nan, just leave the original values alone
                data_DA.values = np.where(~np.isnan(data_model_projection),
                                          data_model_projection, data_DA.values)

            else:
                print('file loaded but empty')
                record_notes = record_notes + ' -- empty record -- '

            # update time values
            if 'time_bounds_var' in extra_information:
                data_DA.time_start.values[0] = ds.Time_bounds[0][0].values
                data_DA.time_end.values[0] = ds.Time_bounds[0][1].values
            elif 'no_time' in extra_information:
                data_DA.time_start.values[0] = record_date
                data_DA.time_end.values[0] = record_date
            elif 'no_time_dashes' in extra_information:
                new_start_time = f'{ds.time_coverage_start[0:4]}-{ds.time_coverage_start[4:6]}-{ds.time_coverage_start[6:8]}'
                new_end_time = f'{ds.time_coverage_end[0:4]}-{ds.time_coverage_end[4:6]}-{ds.time_coverage_end[6:8]}'
                data_DA.time_start.values[0] = new_start_time
                data_DA.time_end.values[0] = new_end_time
            elif time_zone_included_with_time:
                data_DA.time_start.values[0] = ds.time_coverage_start[:-1]
                data_DA.time_end.values[0] = ds.time_coverage_end[:-1]
            else:
                data_DA.time_start.values[0] = ds.time_coverage_start
                data_DA.time_end.values[0] = ds.time_coverage_end

            if 'time_var' in extra_information:
                data_DA.time.values[0] = ds.Time[0].values
            else:
                data_DA.time.values[0] = record_date

            data_DA.attrs['notes'] = record_notes

            save_filename = f'{new_filename}_{data_field}'

            # TODO: dont save, just return data_DA for a list of transformed data arrays
            ea.save_to_disk(data_DA,
                            save_filename,
                            fill_values['binary'], fill_values['netcdf'],
                            output_dirs['netcdf'], output_dirs['binary'],
                            binary_dtype, model_grid_type, save_binary=save_binary,
                            save_netcdf=save_netcdf)

            num_files_saved += int(save_binary) + int(save_netcdf)

        else:
            print(f'no file for {record_date}, file path: {record_filepath}')
            record_notes = record_notes + ' -- file not found! '

    return num_files_saved
# %%

# return num_files_saved


def generalized_process_loop(data_fields,
                             iso_dates_for_year,
                             paths,
                             source_indices_within_target_radius_i,
                             num_source_indices_within_target_radius_i,
                             nearest_source_index_to_target_index_i,
                             model_grid, model_grid_type,
                             array_precision,
                             time_zone_included_with_time,
                             extra_information,
                             filenames,
                             fill_values,
                             output_dirs,
                             binary_dtype,
                             new_data_attr,
                             save_binary,
                             save_netcdf):

    num_files_saved = 0

    # Process each date of the year
    for record_date in iso_dates_for_year:
        if paths[record_date] != []:
            record_filepath = paths[record_date]
        else:
            record_filepath = Path('')

        dashless_date = record_date.replace('-', '')
        new_filename = f'{filenames["shortest"]}_{dashless_date}'

        # send filename to the transformation routine, return data array
        # that contains the original values mapped to new grid
        num_files_saved += generalized_transform_to_model_grid(source_indices_within_target_radius_i,
                                                               num_source_indices_within_target_radius_i,
                                                               nearest_source_index_to_target_index_i,
                                                               model_grid, model_grid_type,
                                                               record_date, record_filepath,
                                                               data_fields,
                                                               array_precision,
                                                               time_zone_included_with_time,
                                                               extra_information,
                                                               new_filename,
                                                               fill_values,
                                                               output_dirs,
                                                               binary_dtype,
                                                               new_data_attr,
                                                               save_binary,
                                                               save_netcdf)

    return num_files_saved
# %%

# return assimilated_data_DA_year_merged


def open_and_merge(data_field_info, iso_dates_for_year, assimilated_paths,
                   model_grid, model_grid_type, array_precision, remove_nan_days_from_data):

    assimilated_data_DA_year = []

    # set data info values
    data_field = data_field_info['name']
    standard_name = data_field_info['standard_name']
    long_name = data_field_info['long_name']
    units = data_field_info['units']

    for record_date in iso_dates_for_year:
        if assimilated_paths[record_date] != []:
            assimilated_filepath = assimilated_paths[record_date]
        else:
            assimilated_filepath = Path('')

        # create empty data array
        assimilated_DA = ea.make_empty_record(standard_name, long_name, units,
                                              record_date,
                                              model_grid, model_grid_type,
                                              array_precision)

        if assimilated_filepath.is_file():
            print('opening ', assimilated_filepath.name)
            assimilated_DA = xr.open_dataarray(assimilated_filepath)

        assimilated_data_DA_year.append(assimilated_DA)

    if remove_nan_days_from_data:
        nonnan_days = []
        for i in range(len(assimilated_data_DA_year)):
            if(np.count_nonzero(~np.isnan(assimilated_data_DA_year[i].values)) > 0):
                nonnan_days.append(assimilated_data_DA_year[i])
        assimilated_data_DA_year_merged = xr.concat((nonnan_days), dim='time')
    else:
        assimilated_data_DA_year_merged = xr.concat(
            (assimilated_data_DA_year), dim='time')

    return assimilated_data_DA_year_merged
# %%

# %%
# returns nothing


def generalized_aggregate_and_save(DA_year_merged,
                                   new_data_attr,
                                   do_monthly_aggregation,
                                   year,
                                   skipna_in_mean,
                                   filenames,
                                   fill_values,
                                   output_dirs,
                                   binary_dtype,
                                   model_grid_type,
                                   save_binary=True,
                                   save_netcdf=True):

    # if everything comes back nans it means there were no files
    # to load for the entire year.  don't bother saving the
    # netcdf or binary files for this year
    if np.sum(~np.isnan(DA_year_merged.values)) == 0:
        print('Empty year not writing to disk', year)
    else:

        # update the dataset attributes
        DA_year_merged.attrs['original_dataset_title'] = new_data_attr['original_dataset_title']
        DA_year_merged.attrs['original_dataset_url'] = new_data_attr['original_dataset_url']
        DA_year_merged.attrs['original_dataset_reference'] = new_data_attr['original_dataset_reference']
        DA_year_merged.attrs['original_dataset_doi'] = new_data_attr['original_dataset_doi']
        DA_year_merged.attrs['interpolated_grid_id'] = new_data_attr['interpolated_grid_id']
        DA_year_merged.name = new_data_attr['new_name']

        if do_monthly_aggregation:
            mon_DA_year = []
            for month in range(1, 13):
                # to find the last day of the month, we go up one month,
                # and back one day
                #   if Jan-Nov, then we'll go forward one month to Feb-Dec
                if month < 12:
                    cur_mon_year = np.datetime64(str(year) + '-' +
                                                 str(month+1).zfill(2) +
                                                 '-' + str(1).zfill(2), 'ns')
                    # for december we go up one year, and set month to january
                else:
                    cur_mon_year = np.datetime64(str(year+1) + '-' +
                                                 str('01') +
                                                 '-' + str(1).zfill(2), 'ns')

                mon_str = str(year) + '-' + str(month).zfill(2)

                mon_DA = \
                    DA_year_merged.sel(time=mon_str).mean(axis=0,
                                                          skipna=skipna_in_mean, keep_attrs=True)

                tb, ct = ea.make_time_bounds_from_ds64(cur_mon_year, 'AVG_MON')

                mon_DA = mon_DA.assign_coords({'time': ct})
                mon_DA = mon_DA.expand_dims('time', axis=0)

                avg_start_time = mon_DA.time.copy(deep=True)
                avg_start_time.values[0] = tb[0]

                avg_end_time = mon_DA.time.copy(deep=True)
                avg_end_time.values[0] = tb[1]

                avg_center_time = mon_DA.time.copy(deep=True)
                avg_center_time.values[0] = ct

                # we'll make the center of the averaging time
                mon_DA = mon_DA.assign_coords(
                    {'time_start': ('time', avg_start_time)})
                mon_DA = mon_DA.assign_coords(
                    {'time_end': ('time', avg_end_time)})

                # halfway through the approx 1M averaging period.
                mon_DA.time.values[0] = ct
                mon_DA.time.attrs['long_name'] = 'center time of 1M averaging period'

                mon_DA_year.append(mon_DA)

            mon_DA_year_merged = xr.concat((mon_DA_year), dim='time')

        #######################################################
        ## BEGIN SAVE TO DISK                                ##

        ea.save_to_disk(DA_year_merged,
                        filenames['shortest'],
                        fill_values['binary'], fill_values['netcdf'],
                        output_dirs['netcdf'], output_dirs['binary'],
                        binary_dtype, model_grid_type, save_binary=save_binary,
                        save_netcdf=save_netcdf)

        if do_monthly_aggregation:
            ea.save_to_disk(mon_DA_year_merged,
                            filenames['monthly'],
                            fill_values['binary'], fill_values['netcdf'],
                            output_dirs['netcdf'], output_dirs['binary'],
                            binary_dtype, model_grid_type, save_binary=save_binary,
                            save_netcdf=save_netcdf)

        ## END   SAVE TO DISK                                ##
        #######################################################
 # %%
