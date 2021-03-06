#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from collections import OrderedDict
import uuid
import pickle

##################3
def generate_json(dataset_base_dir):
    datasets = list((dataset_base_dir.glob('*')))
    pprint(datasets)

    dataset_names = list(map(lambda x: x.name, datasets))
    pprint(dataset_names)

    for dataset in dataset_names:
        print('\n\n\n', dataset)
        print('------------------------')
        netcdf_files = np.sort(list((dataset_base_dir / dataset).glob('**/*ECCO*nc')))
       
        file_metadata = dict()
        for netcdf_file in netcdf_files:
            print(netcdf_file.name);
            try:
                tmp = xr.open_dataset(netcdf_file, chunks={})
                file_metadata[netcdf_file.name] = dict()
                file_metadata[netcdf_file.name]['title'] = tmp.title
                file_metadata[netcdf_file.name]['time_coverage_start'] = tmp.time_coverage_start
                file_metadata[netcdf_file.name]['time_coverage_end'] = tmp.time_coverage_end
                file_metadata[netcdf_file.name]['time_coverage_duration'] = tmp.time_coverage_duration
                file_metadata[netcdf_file.name]['uuid'] = tmp.uuid
                file_metadata[netcdf_file.name]['date_created'] = tmp.date_created
                file_metadata[netcdf_file.name]['date_metadata_modified'] = tmp.date_metadata_modified

            except:
                print('some error with ', netcdf_file)
                sys.exit()
                
        filename = 'dataset_dictionary_' + dataset + '.json'
        try:
            with open(dataset_base_dir / filename, 'w') as outfile:
                json.dump(file_metadata, outfile, indent=4)
        except:
            print('could not save')

#####################
def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', type=str, required=True,\
                       help='directory containing dataset grouping subdirectories')
    return parser

#####################
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    dataset_base_dir = Path(args.dataset_base_dir)

    generate_json(dataset_base_dir) 
