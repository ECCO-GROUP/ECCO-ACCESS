#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:00:14 2020

@author: ifenty
"""
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import netCDF4 as nc4
import xarray as xr
import datetime
from pprint import pprint

#%%




#%%
def calculate_valid_minmax_for_dataset(dataset_base_dir):
    print(dataset_base_dir)
    groupings = list(dataset_base_dir.glob('*'))

    pprint(groupings)
    tmp = []
    file = []

    for grouping in groupings:
    #grouping = groupings[grouping_to_process]

        print('\n --- grouping', grouping)

        glob_str = grouping.name + "*.nc"
        files_in_grouping = np.sort(list(grouping.glob(glob_str)))

        print('\n ----- files ', files_in_grouping)
        data_vars = dict()

        print('processing ', grouping.name)
        for file_i, file in enumerate(files_in_grouping):

            print(file)
            tmp = xr.open_dataset(grouping / file, chunks='auto')

            print(tmp.data_vars)
            if file_i == 0:
                for data_var in tmp.data_vars:
                    data_vars[data_var] = dict()
                    data_vars[data_var]['valid_max'] = []
                    data_vars[data_var]['valid_min'] = []

            for data_var in tmp.data_vars:
                data_vars[data_var]['valid_max'].append(tmp[data_var].attrs['valid_max'])
                data_vars[data_var]['valid_min'].append(tmp[data_var].attrs['valid_min'])

        # finish looping through files, calculate min and max
        new_dict = dict()
        for data_var in data_vars.keys():
            print ('\n==== ', data_var)
            print('\t ------ valid mins')
            pprint(data_vars[data_var]['valid_min'])
            print('\t ------ valid maxs')
            pprint(data_vars[data_var]['valid_max'])

            data_vars[data_var]['valid_min'] = np.nanmin(data_vars[data_var]['valid_min'])
            data_vars[data_var]['valid_max'] = np.nanmax(data_vars[data_var]['valid_max'])

            # create an array with min and max
            tmp_arr = [data_vars[data_var]['valid_min'], data_vars[data_var]['valid_max']]

            # create a dictionary with the min and max for this data_var
            new_dict[data_var] = dict()
            new_dict[data_var]['dims'] = 'nb'
            new_dict[data_var]['data'] = tmp_arr

        # convert dictionary to a datast
        valid_minmax_ds = xr.Dataset.from_dict(new_dict)

        # current time and date
        current_time = datetime.datetime.now().isoformat()[0:19]
        valid_minmax_ds.attrs['date_created'] = current_time

        valid_minmax_ds.attrs['origin'] = str(grouping)
        valid_minmax_ds.attrs['comment'] = 'valid_min and valid_max for data variable'

        valid_minmax_ds.to_netcdf(grouping / 'valid_minmax.nc')

        print('\n --- final for ', grouping.name)
        pprint(data_vars)


#%%

def apply_valid_minmax_for_dataset(dataset_base_dir, scaling_factor=1.0, valid_minmax_prec=32):

    print(dataset_base_dir)
    groupings = list(dataset_base_dir.glob('*'))

    pprint(groupings)

    for grouping in groupings:
        print('\n\n --- grouping', grouping)

        valid_minmax_ds = xr.open_dataset(grouping / 'valid_minmax.nc')
        for data_var in valid_minmax_ds:
            print('\n ', data_var)
            pprint(valid_minmax_ds[data_var].values)

        glob_str = grouping.name + "*.nc"

        # find list of files
        files_in_grouping = np.sort(list(grouping.glob(glob_str)))

        print('\n ----- files ', files_in_grouping)

        print('processing ', grouping.name)

        # loop through files
        for file_i, file in enumerate(files_in_grouping):

            print(file)
            # open netcdf
            tmp_ds = nc4.Dataset(grouping / file, 'a')

            # loop through data variables in the valid_minmax_ds
            for data_var in valid_minmax_ds.data_vars:
                print('\n BEFORE', data_var)
                print(tmp_ds[data_var].getncattr('valid_min'))
                print(tmp_ds[data_var].getncattr('valid_max'))

                # replace the values of the valid min and max

                if valid_minmax_prec == 32:
                    tmp_min = np.float32(valid_minmax_ds[data_var].values[0]*scaling_factor)
                    tmp_max = np.float32(valid_minmax_ds[data_var].values[1]*scaling_factor)
                elif valid_minmax_prec ==64:
                    tmp_min = np.float64(valid_minmax_ds[data_var].values[0]*scaling_factor)
                    tmp_max = np.float64(valid_minmax_ds[data_var].values[1]*scaling_factor)

                tmp_ds[data_var].setncattr('valid_min', tmp_min)
                tmp_ds[data_var].setncattr('valid_max', tmp_max)

                print('\n AFTER', data_var)
                print(tmp_ds[data_var].getncattr('valid_min'))
                print(tmp_ds[data_var].getncattr('valid_max'))

            # update date of modified metadata
            current_time = datetime.datetime.now().isoformat()[0:19]
            tmp_ds.setncattr('date_metadata_modified', current_time)

            # close the file
            tmp_ds.close()

#%%

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', required=True, type=str,\
                        help='directory containing dataset grouping subdirectories')

    parser.add_argument('--calculate_valid_minmax', type=bool, nargs='?',\
                        const=True, default=True,\
                            help='calculate the valid minmax for the groupings in dataset_base_dir')

    parser.add_argument('--apply_valid_minmax', type=bool, nargs='?',\
                        const=True, default=False,\
                            help='apply the new minmax for the groupings in dataset_base_dir')

    parser.add_argument('--valid_scaling_factor', required=False, type=float, default=1.0,\
                       help='scaling factor by which to inflate the valid min and valid max')

    parser.add_argument('--valid_minmax_prec', required=False, type=int, default=32, choices=[32, 64],\
                       help='32 or 64 bit precision for valid min and max')

    return parser



#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    dataset_base_dir = Path(args.dataset_base_dir)

    apply_valid_minmax = args.apply_valid_minmax

    calculate_valid_minmax = args.calculate_valid_minmax

    valid_minmax_prec = args.valid_minmax_prec

    if valid_minmax_prec == 32:
        valid_scaling_factor = np.float32(args.valid_scaling_factor)
    elif valid_minmax_prec == 64:
        valid_scaling_factor = np.float64(args.valid_scaling_factor)


    print('\n\n===================================')
    print('starting update_valid_minmax')
    print('dataset_base_dir', dataset_base_dir)
    print('apply_valid_minmax', apply_valid_minmax)
    print('calculate_valid_minmax', calculate_valid_minmax)
    print('valid_scaling_factor', valid_scaling_factor)

    #dataset_base_dir = Path('/home/ifenty/tmp/v4r4_nc_output_20201215_native/native/mon_mean_x')


    if calculate_valid_minmax or apply_valid_minmax:
        calculate_valid_minmax_for_dataset(dataset_base_dir)

    if apply_valid_minmax:
        apply_valid_minmax_for_dataset(dataset_base_dir, valid_scaling_factor, valid_minmax_prec)
