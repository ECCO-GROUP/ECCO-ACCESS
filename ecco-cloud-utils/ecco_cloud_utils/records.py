# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:24:13 2020

@author: Ian
"""
import xarray as xr
import numpy as np
from pathlib import Path
from .llc_array_conversion import llc_tiles_to_compact

# %%


def make_empty_record(standard_name, long_name, units,
                      record_date,
                      model_grid, model_grid_type,
                      array_precision):

    # make an empty data array to hold the interpolated 2D field
    # all values are nans.
    # dimensions are the same as model_grid.XC
    data_DA = xr.DataArray(np.ones(np.shape(model_grid.XC.values),
                                   dtype=array_precision),
                           dims=model_grid.XC.dims)*np.nan

    data_DA = data_DA.assign_coords(time=np.datetime64(record_date, 'ns'))
    data_DA = data_DA.expand_dims(dim='time', axis=0)

    # add start and end time records. default is same value as record date
    data_DA = data_DA.assign_coords(
        time_start=('time', data_DA.time.copy(deep=True)))
    data_DA = data_DA.assign_coords(
        time_end=('time', data_DA.time.copy(deep=True)))

    for dim in model_grid.XC.dims:
        data_DA = data_DA.assign_coords({dim: model_grid[dim]})

    # create XC and YC coordinates of empty record from XC YC of model_grid
    if model_grid_type == 'llc':
        # llc grid has 'tile dimension'
        data_DA = data_DA.assign_coords(
            {'XC': (('tile', 'j', 'i'), model_grid.XC)})
        data_DA = data_DA.assign_coords(
            {'YC': (('tile', 'j', 'i'), model_grid.YC)})

    # some grids only have j i dimensions
    elif model_grid_type == 'latlon':
        data_DA = data_DA.assign_coords({'XC': (('j', 'i'), model_grid.XC)})
        data_DA = data_DA.assign_coords({'YC': (('j', 'i'), model_grid.YC)})

    else:
        print('invalid grid type!')
        return []

    # copy over the attributes from XC and YC to the dataArray
    data_DA.XC.attrs = model_grid.XC.attrs
    data_DA.YC.attrs = model_grid.YC.attrs

    # add some metadata
    data_DA.attrs = []
    if 'title' in model_grid:
        data_DA.attrs['interpolated_grid'] = model_grid.title
    else:
        data_DA.attrs['interpolated_grid'] = model_grid.name
    data_DA.attrs['model_grid_type'] = model_grid_type
    data_DA.attrs['long_name'] = long_name
    data_DA.attrs['standard_name'] = standard_name
    data_DA.attrs['units'] = units

    return data_DA

# %%


def save_to_disk(data_DA,
                 output_filename,
                 binary_fill_value, netcdf_fill_value,
                 netcdf_output_dir, binary_output_dir, binary_output_dtype,
                 model_grid_type, save_binary=True, save_netcdf=True):

    if save_binary:
        # define binary file output filetype
        dt_out = np.dtype(binary_output_dtype)

        # create directory
        binary_output_dir.mkdir(exist_ok=True)

        # define binary output filename
        binary_output_filename = binary_output_dir / output_filename

        # replace nans with the binary fill value (something like -9999)
        tmp_fields = np.where(np.isnan(data_DA.values),
                              binary_fill_value, data_DA.values)

        # SAVE FLAT BINARY
        # loop through each record of the year, save binary fields one at a time
        # appending each record as we go
        fd1 = open(str(binary_output_filename), 'wb')
        fd1 = open(str(binary_output_filename), 'ab')

        for i in range(len(data_DA.time)):
            print('saving binary record: ', str(i))

            # if we have an llc grid, then we have to reform to compact
            if model_grid_type == 'llc':
                tmp_field = llc_tiles_to_compact(
                    tmp_fields[i, :], less_output=True)

            # otherwise assume grid is x,y (2 dimensions)
            elif model_grid_type == 'latlon':
                tmp_field = tmp_fields[i, :]

            else:
                print('unknown model grid type!')
                tmp_field = []
                return []

            # make sure we have something to save...
            if len(tmp_field) > 0:
                # if this is the first record, create new binary file
                tmp_field.astype(dt_out).tofile(fd1)

        # close the file at the end of the operation
        fd1.close()

    if save_netcdf:
        print('saving netcdf record')

        # create directory
        netcdf_output_dir.mkdir(exist_ok=True)

        # define netcdf output filename
        netcdf_output_filename = netcdf_output_dir / \
            Path(output_filename + '.nc')

        # SAVE NETCDF
        # replace the binary fill value (-9999) with the netcdf fill value
        # which is much more interesting

        # replace nans with the binary fill value (something like -9999)
        data_DA.values = \
            np.where(np.isnan(data_DA.values),
                     netcdf_fill_value, data_DA.values)

        encoding_each = {'zlib': True,
                         'complevel': 9,
                         'fletcher32': True,
                         '_FillValue': netcdf_fill_value}

        data_DS = data_DA.to_dataset()

        encoding = {var: encoding_each for var in data_DS.data_vars}

        # the actual saving (so easy with xarray!)
        data_DS.to_netcdf(netcdf_output_filename,  encoding=encoding)
        data_DS.close()
