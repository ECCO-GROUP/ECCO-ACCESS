#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:06:49 2020

@author: ifenty
"""

import sys
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCOv4-py')

from importlib import reload
import ecco_v4_py as ecco
import ecco_cloud_utils as ea
import copy as copy
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import netCDF4 as nc4
import xarray as xr
import datetime
from pprint import pprint
import pyresample as pr
import uuid
import pickle
from collections import OrderedDict
from pandas import read_csv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

def even_cax(cmin, cmax, fac=1.0):
    tmp = np.max([np.abs(cmin), np.abs(cmax)])
    cmin = -tmp*fac
    cmax =  tmp*fac
    return cmin, cmax


#%%
def create_thumbnails(dataset_base_dir,\
                      product_type,\
                      grouping_to_process,\
                      thumbnail_height,\
                      thumbnail_width_to_height_ratio,\
                      output_dir):


    #%%
    #dataset_base_dir=Path('/home/ifenty/tmp/v4r4_nc_output_20201223/native/mon_mean')
    #product_type='native'
    #thumbnail_height = 9.11
    #thumbnail_width_to_height_ratio = 1.58
    #output_dir=Path('/home/ifenty/tmp/v4r4_nc_output_20201223/thumbnails/native/mon_mean')
    #grouping_to_process=0
    #%%


    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True,exist_ok=True)
        except:
            print ('cannot make %s ' % output_dir)


    # grouping
    print('dataset_base_dir ', dataset_base_dir)
    groupings = list(dataset_base_dir.glob('*/'))

    #pprint(groupings)
    print('\n... found groupings ')
    for grouping_i, grouping in enumerate(groupings):
        print(grouping_i, grouping.name)

    if grouping_to_process != 'all':
        grouping_to_process = int(grouping_to_process)

        groupings = [groupings[grouping_to_process]]

    plt.close('all')
    for grouping_i, grouping in enumerate(groupings):
        print('\n --- grouping to process', grouping.name)

        glob_str = grouping.name + "*ECCO*.nc"

        # find list of files
        files_in_grouping = np.sort(list(grouping.glob(glob_str)))

        print('\n ----- files ', files_in_grouping)

        print('processing ', grouping.name)
        print('\n')
        # loop through files
        #for file_i, file in enumerate(files_in_grouping):
        #    print(file_i, file)

        tmp_ds = xr.open_dataset(files_in_grouping[0])

        shortname_tmp = tmp_ds.metadata_link.split('ShortName=')[1]
        print(shortname_tmp)

        tb_filename = shortname_tmp + '.jpg'
        tb_filenameb = shortname_tmp + '_b.jpg'

        data_vars = list(tmp_ds.data_vars)
        print(data_vars)

        which_data_var = 0
        which_k_level = 0

        data_var = data_vars[0]

        if 'FRESH_FLUX' in shortname_tmp:
            data_var = 'oceFWflx'
        elif 'HEAT_FLUX' in shortname_tmp:
            data_var = 'oceQnet'
        elif  'BOLUS_STREAMFUNCTION' in shortname_tmp:
            which_k_level = 1
        elif 'SALT_PLUME' in shortname_tmp:
            which_k_level = 1
        elif 'DENS_STRAT' in shortname_tmp:
            data_var = 'RHOAnoma'


        tmp_ds_dims = set(tmp_ds.dims)


        # arbitrarily pick the first one


        if product_type == 'native':

            dims_3D = {'k','k_u','k_l'}
            dims_2D = {'i','i_g','j','j_g'}

            if len(dims_3D.intersection(tmp_ds_dims)) > 0:
                ds_dim = 3

            elif len(dims_2D.intersection(tmp_ds_dims)) > 0:
                ds_dim = 2

        else:
            dims_3D = {'k','k_u','k_l'}
            dims_2D = {'latitude','longitude'}

            if len(dims_3D.intersection(tmp_ds_dims)) > 0:
                ds_dim = 3

            elif len(dims_2D.intersection(tmp_ds_dims)) > 0:
                ds_dim = 2


        print(ds_dim)

        fname = output_dir / tb_filename

        print(data_var, shortname_tmp)
        if ds_dim == 3:
            tmp = tmp_ds[data_var][0,which_k_level,:]

        elif ds_dim == 2:
            tmp = tmp_ds[data_var][0,:]


        cmin = np.nanmin(tmp)
        cmax = np.nanmax(tmp)
        print('color lims')
        print(shortname_tmp, cmin, cmax)


        # default
        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_bad(color='dimgray')

        if ('STRESS' in shortname_tmp) or \
           ('FLUX' in shortname_tmp) or\
           ('VEL' in shortname_tmp) or\
           ('TEND' in shortname_tmp) or\
           ('BOLUS' in shortname_tmp):

            fac = 0.8
            if ('FRESH_FLUX' in shortname_tmp) or\
                ('MOMENTUM_TEND' in shortname_tmp):
                fac = 0.25

            print('even_cax')
            cmin, cmax= even_cax(cmin, cmax, fac)
            print(cmin, cmax)

            cmap = copy.copy(cmocean.cm.balance)#plt.get_cmap('bwr'))
            cmap.set_bad(color='dimgray')

        elif ('TEMP_SALINITY' in shortname_tmp):
            cmap = copy.copy(cmocean.cm.thermal)
            cmap.set_bad(color='dimgray')

        elif ('MIXED_LAYER' in shortname_tmp):

            cmap = copy.copy(cmocean.cm.deep)
            cmap.set_bad(color='dimgray')
            cmin = 0
            cmax = 250

        elif ('SEA_ICE_CONC' in shortname_tmp):

            cmap = copy.copy(cmocean.cm.ice)
            cmap.set_bad(color='dimgray')
            cmin = 0
            cmax = 1

        elif 'DENS_STRAT' in shortname_tmp:
            cmap = copy.copy(cmocean.cm.dense)
            cmap.set_bad(color='dimgray')

        if product_type == 'native':

            tb = ecco.plot_tiles(tmp, cmap=cmap, fig_num=grouping_i, \
                                 show_tile_labels=False,\
                                     cmin=cmin, cmax=cmax)

            fig_ref = tb[0]
        elif product_type == 'latlon':
            fig_ref = plt.figure(num=grouping_i, figsize=[8,8]);
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=-66.5))
            ecco.plot_global(tmp_ds[data_var].longitude,\
                             tmp_ds[data_var].latitude,\
                             tmp,\
                             4326, cmin, cmax, ax,\
                             show_grid_labels=False, cmap=cmap)

        fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)

#            axs = tb[0].axes
#            for ax in axs:
#                print(type(ax))
#                #ax.set_axis_off()
#                ax.set_frame_on(True)

        print (fname)
        #plt.savefig(fname, dpi=175, facecolor='w', pad_inches=0, format='jpg')
        plt.savefig(fname, dpi=100, facecolor='w', bbox_inches='tight', pad_inches = 0.05)


            #%%
def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', type=str, required=True,\
                        help='directory containing dataset grouping subdirectories')

    parser.add_argument('--product_type', type=str, required=True,\
                        choices=['native','latlon'],\
                        help='')

    parser.add_argument('--grouping_to_process', type=str, required=True,\
                        default='all',\
                        help='which grouping num to process, or "all"')

    parser.add_argument('--thumbnail_height', type=float,\
                        default = 9.11,\
                        help='')

    parser.add_argument('--thumbnail_width_to_height_ratio', type=float,\
                        default = 1,
                        help='')

    parser.add_argument('--output_dir', type=str, required=True,\
                        help='output_directory')

    return parser


#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    dataset_base_dir = Path(args.dataset_base_dir)

    product_type = args.product_type

    grouping_to_process = args.grouping_to_process

    thumbnail_height = args.thumbnail_height

    thumbnail_width_to_height_ratio = args.thumbnail_width_to_height_ratio

    output_dir = Path(args.output_dir)


    print('\n\n===================================')
    print('starting create_thumbnails.py')
    print('\n')
    print('dataset_base_dir', dataset_base_dir)
    print('product_type', product_type)
    print('grouping_to_process', grouping_to_process)

    print('thumbnail_height', thumbnail_height)
    print('thumbnail_width_to_height_ratio', thumbnail_width_to_height_ratio)

    print('\n')

    create_thumbnails(dataset_base_dir,\
                      product_type,\
                      grouping_to_process,\
                      thumbnail_height,\
                      thumbnail_width_to_height_ratio,\
                      output_dir)