#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
from importlib import reload
sys.path.append('/home5/ifenty/ECCOv4-py')
import ecco_v4_py as ecco

sys.path.append('/home5/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
import ecco_cloud_utils as ea

from pathlib import Path
import pandas as pd
import netCDF4 as nc4
import xarray as xr
import datetime
from pprint import pprint
from collections import OrderedDict
import pyresample as pr
import uuid
import pickle
from pandas import read_csv
reload(ecco)                               

#%%

#products_root_dir = Path('/home5/ifenty/podaac/lat-lon/mon_mean/')
products_root_dir = Path('/home5/ifenty/podaac/native/mon_mean/')

datasets = list((products_root_dir.glob('*')))
pprint(datasets)

dataset_names = list(map(lambda x: x.name, datasets))
pprint(dataset_names)

for dataset in dataset_names:
    print('\n\n\n', dataset)
    print('------------------------')
    netcdf_files = np.sort(list((products_root_dir / dataset).glob('**/*nc')))
    
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
            
        except:
            print('some error with ', netcdf_file)
            sys.exit()
            
    
    filename = 'dataset_dictionary_' + dataset + '.json'
    try:
        with open(products_root_dir / filename, 'w') as outfile:
            json.dump(file_metadata, outfile, indent=4)
    except:
        print('could not save')

    
