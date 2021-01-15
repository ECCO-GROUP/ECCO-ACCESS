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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_files_to_process(all_files, n_jobs, job_id):
    file_chunks = list(split(all_files, n_jobs))

    print('total number of files in this grouping ', len(all_files))
    print('total number of files in this job ', len(file_chunks[job_id]))

    return file_chunks[job_id]


def fix_Zul_kul(dataset_base_dir,\
                grouping_to_process, n_jobs, job_id, dry_run=False):

    # new attributes
    Zl_attrs = {'long_name': "depth of the top face of tracer grid cells", 'units': 'm', 'positive': 'up', 'comment': "First element is 0m, the depth of the top face of the first tracer grid cell (ocean surface). Last element is the depth of the top face of the deepest grid cell. The use of 'l' in the variable name follows the MITgcm convention for ocean variables in which the lower (l) face of a tracer grid cell on the logical grid corresponds to the top face of the grid cell on the physical grid. In other words, the logical vertical grid of MITgcm ocean variables is inverted relative to the physical vertical grid.", 'coverage_content_type': 'coordinate', 'standard_name': 'depth'}

    Zu_attrs = {'long_name': "depth of the bottom face of tracer grid cells", 'units': 'm', 'positive': 'up', 'comment': "First element is -10m, the depth of the bottom face of the first tracer grid cell. Last element is the depth of the bottom face of the deepest grid cell. The use of 'u' in the variable name follows the MITgcm convention for ocean variables in which the upper (u) face of a tracer grid cell on the logical grid corresponds to the bottom face of the grid cell on the physical grid. In other words, the logical vertical grid of MITgcm ocean variables is inverted relative to the physical vertical grid.", 'coverage_content_type': 'coordinate', 'standard_name': 'depth'}

    k_l_attrs = {'axis': 'Z', 'long_name': "grid index in z corresponding to the top face of tracer grid cells ('w' locations)", 'c_grid_axis_shift': -0.5, 'swap_dim': 'Zl', 'comment': "First index corresponds to the top surface of the uppermost tracer grid cell. The use of 'l' in the variable name follows the MITgcm convention for ocean variables in which the lower (l) face of a tracer grid cell on the logical grid corresponds to the top face of the grid cell on the physical grid.", 'coverage_content_type': 'coordinate'}

    k_u_attrs = {'axis': 'Z', 'long_name': "grid index in z corresponding to the bottom face of tracer grid cells ('w' locations)", 'c_grid_axis_shift': 0.5, 'swap_dim': 'Zu', 'comment': "First index corresponds to the bottom surface of the uppermost tracer grid cell. The use of 'u' in the variable name follows the MITgcm convention for ocean variables in which the upper (u) face of a tracer grid cell on the logical grid corresponds to the bottom face of the grid cell on the physical grid.", 'coverage_content_type': 'coordinate'}


    print('\n\n===========================')
    print('fix Zul Kul for dataset')

    # grouping
    print('dataset_base_dir ', dataset_base_dir)
    groupings = np.sort(list(dataset_base_dir.glob('*')))

    #pprint(groupings)
    print('\n... found groupings ')
    for grouping_i, grouping in enumerate(groupings):
        print(grouping_i, grouping.name)

    for g_i in grouping_to_process:
        grouping = groupings[g_i] 
        print('\n --- processing', g_i, grouping.name)

        glob_str = grouping.name + "*ECCO*.nc"

        # find list of files
        files_in_grouping = np.sort(list(grouping.glob(glob_str)))

        files_to_process = get_files_to_process(files_in_grouping, n_jobs, job_id)

        # loop through files
        for file_i, file in enumerate(files_to_process):
            print(file)
            
            if not dry_run:
                # open netcdf
                tmp_ds_xr = xr.open_dataset(grouping / file, chunks={})
                tmp_ds_xr.close()

                tmp_ds = nc4.Dataset(grouping / file, 'a')

                tmp_Zu = tmp_ds.variables["Zl"][:]
                tmp_Zl = tmp_ds.variables["Zu"][:]

                print('first Zu ', tmp_ds["Zu"][0])
                print('first Zl ', tmp_ds["Zl"][0])

                # before swapping the arrays of Zl and Zu, check to make sure
                # that we already haven't done so.
                # if the first value of Zl is negative then we've already swapped
                if tmp_ds["Zu"][0] < 0.0:
                    print('first Zu is already negative, not changing Zu Zl values')
                else:           
                    # replaces depth values
                    tmp_ds["Zu"][:] = tmp_Zu 
                    tmp_ds["Zl"][:] = tmp_Zl 
                 
                # replace attributes 
                tmp_ds["Zu"].setncatts(Zu_attrs)
                tmp_ds["Zl"].setncatts(Zl_attrs)
                tmp_ds["k_u"].setncatts(k_u_attrs)
                tmp_ds["k_l"].setncatts(k_l_attrs)

                # update date of modified metadata
                current_time = datetime.datetime.now().isoformat()[0:19]
                tmp_ds.setncattr('date_modified', current_time)
                tmp_ds.setncattr('date_modified', current_time)

                # close the file
                tmp_ds.close()
            else:
                print('dry run!')
#%%

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', type=str, required=True,\
                        help='directory containing dataset grouping subdirectories')

    parser.add_argument('--show_grouping_numbers', type=str2bool, nargs='?', const=True,\
                        help='show id of each grouping (dataset)')

    parser.add_argument('--fix_ZuZlkukl', type=str2bool, nargs='?', const=True,\
                        help='flag whether or not to update coordinates')

    parser.add_argument('--dry_run', type=str2bool, nargs='?', const=True,\
                        help='flag whether or not to apply changes')

    # list of integers
    parser.add_argument('--grouping_to_process', nargs='+', type=int, required=True)

    parser.add_argument('--n_jobs', type=int, required=True)
    parser.add_argument('--job_id', type=int, required=True)

    return parser


# show id number for each grouping in datase_base_dir
def grouping_numbers(dataset_base_dir):

    print('\n\n===========================')
    print('show groupings')

    # grouping
    print('dataset_base_dir ', dataset_base_dir)
    groupings = np.sort(list(dataset_base_dir.glob('*')))

    #pprint(groupings)
    print('\n... found groupings ')
    for grouping_i, grouping in enumerate(groupings):
        print(grouping_i, grouping.name)


#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    dataset_base_dir = Path(args.dataset_base_dir)

    grouping_to_process = args.grouping_to_process
    print(type(grouping_to_process))
    print(grouping_to_process)

    show_grouping_numbers = args.show_grouping_numbers
    fix_ZuZlkukl = args.fix_ZuZlkukl 

    dry_run = args.dry_run 

    n_jobs = args.n_jobs
    job_id = args.job_id

    print('\n\n===================================')
    print('starting update_valid_minmax')
    print('\n')
    print('dataset_base_dir', dataset_base_dir)
    print('grouping_to_process ', grouping_to_process)
    print('job_id', job_id)
    print('n_jobs', n_jobs)
    print('show_grouping_numbers', show_grouping_numbers)
    print('fix_ZuZlkukl', fix_ZuZlkukl)
    print('dry_run ', dry_run)

    print('\n')

    if show_grouping_numbers:
        grouping_numbers(dataset_base_dir)

    if fix_ZuZlkukl:
        fix_Zul_kul(dataset_base_dir, grouping_to_process, n_jobs, job_id, dry_run=dry_run)
