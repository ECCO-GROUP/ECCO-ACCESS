#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import netCDF4 as nc4
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
import time
import warnings
import xarray as xr
import datetime
from collections import OrderedDict
from copy import deepcopy

warnings.filterwarnings('ignore')

def get_groupings(base_dir, grid_type, time_type):
    groupings = dict()
    tmp = base_dir / grid_type/ time_type

    if tmp.exists():
        gdirs = np.sort(list( tmp.iterdir()))
        for pi, p in enumerate(gdirs):
            grouping = str(p).split('/')[-1]
            groupings[pi] = dict()
            groupings[pi]['name'] = grouping
            groupings[pi]['grid'] = grid_type
            groupings[pi]['time_type'] = time_type
            groupings[pi]['directory'] = p
    else:
        print('-- grouping directory does not exist')

    return groupings

def sort_attrs(attrs):
    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    #print('sorted keys')
    #pprint(keys)

    for k in keys:
        od[k] = deepcopy(attrs[k])

    #pprint(od)

    return od


def apply_fixes(ecco_filename):

    print('\nApplying fixes for ', ecco_filename.name)
    try:
        # open a dataset
        with nc4.Dataset(ecco_filename, mode='r+') as tmp_ds:

            # alphabetically sort all attributes
            print ('\n>> sorting attributes')
            sorted_attr_dict = sort_attrs(tmp_ds.__dict__)

            print ('\n>> removing global attributes')
            # delete all attributes one at a time
            for attr in tmp_ds.ncattrs():
                tmp_ds.delncattr(attr)

        with nc4.Dataset(ecco_filename, mode='r+') as tmp_ds:

            # replace all one at a time (in alphabetical order)
            for attr in sorted_attr_dict:
                tmp_ds.setncattr(attr, sorted_attr_dict[attr])

            print(f"\n+ SUCCESS: changes applied {ecco_filename.name}\n")
  
        return 1

    except Exception as e:
        raise e

    print('could not open file!')
    return -1


def f1(ecco_files):
    results = []
    for ecco_filename in ecco_files:
        result = apply_fixes(ecco_filename)
        results.append(result)
    return results



#############
def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', type=str, required=True,\
                        help='directory containing dataset grouping subdirectories')

    parser.add_argument('--grid_type', type=str, required=True,\
                        choices=['native','lat-lon'],\
                        help='')

    parser.add_argument('--time_type', type=str, required=True,\
                        choices=['day_inst','mon_mean','day_mean'],\
                        help='')

    parser.add_argument('--grouping_id', type=int, required=True,\
                        help='which grouping num to process (int)')


    parser.add_argument('--debug', help='only process 2 ecco files', action="store_true")

    return parser


#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    dataset_base_dir = Path(args.dataset_base_dir)
    grid_type = args.grid_type
    time_type = args.time_type
    grouping_id = args.grouping_id

    if args.debug:
        debug = True
    else:
        debug = False

    print('\n===================================')
    print('starting valid_minmax')
    print('\n')
    print('dataset_base_dir', dataset_base_dir)
    print('time_type', time_type)
    print('grid_type', grid_type)
    print('grouping_id', grouping_id)
    print('debug', debug)
    print('\n')

    print("get groupings")
    groupings = get_groupings(dataset_base_dir, grid_type, time_type)

    for gi in groupings.keys():
        print(gi, groupings[gi]['name'])

    if grouping_id >= 0:
        grouping_ids = [grouping_id]
    else:
        grouping_ids = list(range(len(groupings)))

    print('\ngrouping ids to process: ', grouping_ids)

    glob_name = '**/*ECCO_V4r4*nc'

    start_group_loop = time.time()
    print('beginning the loop\n')
    for gi in grouping_ids:
        grouping_info = groupings[gi]
        print('-------------------------------------')
        print('Processing Grouping: ', gi)
        print(grouping_info)

        print('globbing files')
        start_time = time.time()
        ecco_files = np.sort(list(grouping_info['directory'].glob(glob_name)))
        print('...time to glob files ', time.time() - start_time)

        print(f'# files found {len(ecco_files)}')
        if debug:
            ecco_files = ecco_files[0:2]
        print(f'# files to process { len(ecco_files)} ')

        print('computing')
        start_time = time.time()
        f1_out = f1(ecco_files)
        delta_time = time.time() - start_time
        time_per = delta_time / len(ecco_files)

        print('GROUPING FINISHED', gi)
        print(grouping_info)
        print('len ecco_files ', len(ecco_files))
        print('unique results: ', np.unique(f1_out))
        print('len results ', len(f1_out))
        print('time to compute ',delta_time)
        print('time per ', time_per)

    print('ALL FINISHED: total time ', time.time() - start_group_loop)
