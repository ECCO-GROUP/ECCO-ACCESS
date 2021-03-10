#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import dask
import json
import netCDF4 as nc4
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
import time
import warnings
import xarray as xr

warnings.filterwarnings('ignore')


# In[6]:


def load_ecco_fields(data_dir, glob_name):
    time_start=time.time()

    ecco_fields = []
    # opening 312 monthly mean files takes about 40s using parallel & dask
    
    ecco_files = list(data_dir.glob(glob_name))
    print(ecco_files[0:5])
    ecco_fields = xr.open_mfdataset(ecco_files, parallel=True, data_vars='minimal',                                  coords='minimal',compat='override')
    
    tt = time.time() - time_start    
    print(tt / len(ecco_fields))
    print(time.time() - time_start)
    return ecco_fields


# In[21]:


def get_groupings(base_dir, grid_type, time_avg):
    groupings = dict()
    tmp = Path(f'{base_dir}/{grid_type}/{time_avg}')
    print(tmp)
    if tmp.exists():
        for pi, p in enumerate(tmp.iterdir()):
            grouping = str(p).split('/')[-1]
            groupings[pi] = dict()
            groupings[pi]['name'] = grouping
            groupings[pi]['grid'] = grid_type
            groupings[pi]['time_avg'] = time_avg
            groupings[pi]['directory'] = p
            
    return groupings


# In[88]:


def calc_valid_minmax(ecco_fields):
    t0 = time.time()
    results_da = dict()
    for dv in ecco_fields.data_vars:
        print(dv)
        results_da[dv] = dict()
        results_da[dv]['valid_max'] = ecco_fields[dv].max()
        results_da[dv]['valid_min'] = ecco_fields[dv].min()

    results_da_compute = dask.compute(results_da)[0]
    delta_time = time.time()-t0
    
    DAs = []
    for dv in ecco_fields.data_vars:
        print(dv)
        valid_max = results[dv]['valid_max'].values
        valid_min = results[dv]['valid_min'].values
        print(valid_max, valid_min)
        tmp = xr.DataArray([valid_min, valid_max], dims=['valid_min_max'])
        tmp.name = dv
        DAs.append(tmp)

    DS = xr.merge(DAs)
    DS.attrs['title']     = ecco_fields.attrs['title']
    DS.attrs['name']      = groupings[gi]['name']
    DS.attrs['grid']      = groupings[gi]['grid']
    DS.attrs['time_avg']  = groupings[gi]['time_avg']
    DS.attrs['id']        = ecco_fields.attrs['id']
    DS.attrs['shortname'] = ecco_fields.attrs['id'].split('/')[1]
    DS.attrs['directory'] = str(groupings[gi]['directory'])
    DS.attrs['calc_time_seconds'] = delta_time
    
    return DS


# ## Inputs

# In[89]:


gi = 0;


# In[90]:


dataset_base_dir = Path('/home/ifenty/ian1/ifenty/ECCOv4/Version4/Release4/podaac/')


# In[91]:


grids = ['native','latlon']
time_avgs = ['day_inst', 'day_mean','mon_mean']


# ## Calc

# In[92]:


groupings = get_groupings(dataset_base_dir, grids[0], time_avgs[2])
groupings


# In[93]:


ecco_fields = load_ecco_fields(groupings[gi]['directory'], '*ECCO*nc')


# In[94]:


DS = calc_valid_minmax(ecco_fields)


# In[95]:


pprint(DS.attrs)


# In[96]:


DS


# In[97]:


filename = f"valid_minmax_{DS.attrs['name']}_{DS.attrs['grid']}_{DS.attrs['time_avg']}_{DS.attrs['shortname']}.nc"
filename


# In[98]:


output_dir = Path('/home/ifenty/ian1/ifenty/ECCOv4/Version4/Release4/podaac/valid_minmax')
if not output_dir.exists():
    output_dir.mkdir()


# In[99]:


DS.to_netcdf(output_dir / filename)

