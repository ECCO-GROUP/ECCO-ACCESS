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

import dask
from dask.distributed import Client, progress
from dask import delayed

warnings.filterwarnings('ignore')

def get_groupings(base_dir, grid_type, time_type):
    groupings = dict()
    tmp = Path(f'{base_dir}/{grid_type}/{time_type}')
    print(tmp)
    if tmp.exists():
        for pi, p in enumerate(tmp.iterdir()):
            grouping = str(p).split('/')[-1]
            groupings[pi] = dict()
            groupings[pi]['name'] = grouping
            groupings[pi]['grid'] = grid_type
            groupings[pi]['time_type'] = time_type
            groupings[pi]['directory'] = p
            
    return groupings

@delayed
def load_ecco_file(filename):
    time_start=time.time()
    print(filename.name)
    ecco_field = xr.open_dataset(filename).load()
    return ecco_field

@delayed
def get_minmax(ecco_field):
    results_da = dict()
    for dv in ecco_field.data_vars:
        results_da[dv] = dict()
        tmp_min = ecco_field[dv].min()
        tmp_max = ecco_field[dv].max()
        
        results_da[dv]['valid_min'] = tmp_min.values
        results_da[dv]['valid_max'] = tmp_max.values
    
    return results_da   

def construct_DS(results, grouping_info, ds_title, ds_id, delta_time):
    dvs = list(results[0].keys())

    X = dict()
    DAs = []

    # loop through all data varaibles
    for dv in dvs:
        print(dv)
        X[dv] = dict()
        X[dv]['valid_max'] = []
        X[dv]['valid_min'] = []

        # loop through all records
        for r in results:
            X[dv]['valid_min'].append(r[dv]['valid_min'])
            X[dv]['valid_max'].append(r[dv]['valid_max'])

        # final min max for all records
        valid_min = np.array(X[dv]['valid_min']).min() 
        valid_max = np.array(X[dv]['valid_max']).max() 
        
        # construct data array with valid min and max
        tmp = xr.DataArray([valid_min, valid_max], dims=['valid_min_max'])
        tmp.name = dv
        DAs.append(tmp)

    DS = xr.merge(DAs)
    DS.attrs['title']     = ds_title
    DS.attrs['name']      = grouping_info['name']
    DS.attrs['grid']      = grouping_info['grid']
    DS.attrs['time_type'] = grouping_info['time_type']
    DS.attrs['id']        = ds_id
    DS.attrs['shortname'] = ds_id.split('/')[1]
    DS.attrs['directory'] = str(grouping_info['directory'])
    DS.attrs['calc_time_seconds'] = delta_time

    return DS

def save_to_disk(DS, output_dir):
  filename = f"valid_minmax_{DS.attrs['name']}_{DS.attrs['grid']}_{DS.attrs['time_type']}_{DS.attrs['shortname']}.nc"
  print(filename)

  if not output_dir.exists():#parents=True, exist_ok=True):
    output_dir.mkdir(parents=True, exist_ok=True)

  DS.to_netcdf(output_dir / filename)
  DS.close()


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

    parser.add_argument('--n_workers', type=int, required=True,\
                        help='n_workers (int)')

    parser.add_argument('--threads_per_worker', type=int, required=True,\
                        help='threads_per_worker (int)')

    parser.add_argument('--grouping_id', type=int, required=True,\
                        help='which grouping num to process (int)')

    parser.add_argument('--output_dir', type=str, required=True,\
                        help='output_directory')

    return parser


def f(ecco_files):
    results = []
    
    for file in ecco_files:
        ecco_field = load_ecco_file(file)
        result = get_minmax(ecco_field)
        results.append(result)
        
    return results

#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    dataset_base_dir = Path(args.dataset_base_dir)
    grid_type = args.grid_type
    time_type = args.time_type
    grouping_id = args.grouping_id
    n_workers = args.n_workers
    threads_per_worker = args.threads_per_worker
    output_dir = Path(args.output_dir)

    print('\n\n===================================')
    print('starting valid_minmax')
    print('\n')
    print('dataset_base_dir', dataset_base_dir)
    print('time_type', time_type)
    print('grid_type', grid_type)
    print('grouping_id', grouping_id)
    print('n_workers', n_workers)
    print('threads_per_worker', threads_per_worker)
    print('output_dir', output_dir)
    print('\n')

    client = Client(processes=False, n_workers=n_workers, threads_per_worker=threads_per_worker)#,memory_limit='128GB')

    glob_name = '*ECCO*nc'

    print('getting groupings')
    groupings = get_groupings(dataset_base_dir, grid_type, time_type)

    for gi in groupings:
        print(gi, groupings[gi]['name'])

    if grouping_id >= 0:
        grouping_ids = [grouping_id]
    else:
        grouping_ids = list(range(len(groupings)))

    print('beginning the loop\n')
    for gi in grouping_ids:
        grouping_info = groupings[gi]
        print('groupings : ', gi)
        print(grouping_info)
        print('\n')

        print('globbing files')
        start_time = time.time()
        ecco_files = np.sort(list(grouping_info['directory'].glob(glob_name)))
        print('...time to glob files ', time.time() - start_time)

        print('number of ecco files ', len(ecco_files))

        print('computing')
        start_time = time.time()
        results_da_compute = dask.compute(f(ecco_files))[0]
        delta_time = time.time() - start_time
        print('...time to compute ', time.time() - start_time)

        print('constructing DS')
        start_time = time.time()

        tmp_file = xr.open_dataset(ecco_files[0])
        ds_title = tmp_file.attrs['title']
        ds_id    = tmp_file.attrs['id']

        DS = construct_DS(results_da_compute, grouping_info, ds_title, ds_id, delta_time)
        print('time to construct DS ', time.time() - start_time)

        pprint(DS.attrs)    
        save_to_disk(DS, output_dir)
