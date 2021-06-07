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
import random
from collections import OrderedDict

warnings.filterwarnings('ignore')

def get_groupings(base_dir, grid_type, time_type):
    groupings = dict()
    #print('base_dir ', base_dir)
    #print('grid_type ', grid_type)
    #print('time_type ', time_type)
    tmp = base_dir / grid_type/ time_type

    #print(tmp)
    #print('\n')
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

    for k in keys:
        od[k] = attrs[k]

    return od


def load_valid_minmax(valid_minmax_dir):
    valid_minmax_files = list(valid_minmax_dir.glob('**/valid_minmax*.nc'))

    minmax = dict()
    for mmf in valid_minmax_files:
        tmp = xr.open_dataset(mmf)
        ds_id = tmp.attrs['id'].split('/')[1]

        minmax[ds_id] = dict()
        for dv in tmp.data_vars:
            minmax[ds_id][dv] = dict()
            minmax[ds_id][dv]['valid_min'] = tmp[dv].values[0]
            minmax[ds_id][dv]['valid_max'] = tmp[dv].values[1]
    return minmax


def apply_fixes(ecco_filename, minmax, comment_fix, summary_fix, qc_prob):

    comment_keys = list(comment_fix.keys())
    summary_fix_keys = list(summary_fix.keys())

    print('\nApplying fixes for ', ecco_filename.name)
    try:
        # open a dataset
        with nc4.Dataset(ecco_filename, mode='r+') as tmp_ds:

            # get variables in this dataset
            nc_dvs = list(tmp_ds.variables)

            # get ID, shortname, title
            ds_id        = tmp_ds.id.split('/')[1]
            ds_shortname = tmp_ds.metadata_link.split('ShortName=')[1]
            ds_title     = tmp_ds.title

            metadata_updated = False
            ds_shortname_base = "https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName="

            # fix metadata link and summary
            print('\n>> fixing shortname and summary')
            # look for ds_id in summary_fix keys
            if ds_id not in summary_fix_keys:

                print(f'\n+ FAILURE: {ds_id} not in summary_fix.keys(): {ecco_filename.name}')
                return -1

            else: # ds_id found in summary_fix

                summary_fix_title     = summary_fix[ds_id]['title']
                summary_fix_shortname = summary_fix[ds_id]['shortname']
                summary_fix_summary   = summary_fix[ds_id]['summary']

                # require that the title and ds_id match
                if ds_title != summary_fix_title:
                    print(f'\n+ FAILURE: title mismatch in summary_fix[{ds_id}]: {ecco_filename.name}')
                    return -1
                else:
                    print('... title matches', ds_title)

                # fix shortname mismatch (in metadata link)
                if ds_shortname != summary_fix_shortname:
                    print('... shortname mismatch (file, summary_fix):', ds_shortname, summary_fix_shortname)
                    print('... old metadata link: ', tmp_ds.metadata_link)
                    tmp_ds.metadata_link = f'{ds_shortname_base}{summary_fix_shortname}'
                    print('... new metadata link: ', tmp_ds.metadata_link)
                    metadata_updated = True
                else:
                    print('... shortname matches', ds_shortname)

                # fix summary
                if tmp_ds.summary != summary_fix_summary:
                    print('... new summary != old summary')
                    print('\n... old summary: ', tmp_ds.summary)
                    tmp_ds.setncattr('summary', summary_fix_summary)
                    print('\n... new summary: ', tmp_ds.summary)
                    metadata_updated = True
                else:
                    print('... new summary == old summary: not updating summary')


            # fix units on select variables
            print('\n>> fixing units')
            if 'EXFatemp' in nc_dvs:
                v = tmp_ds.variables['EXFatemp']

                if v.units != 'degree_K':
                    print (f'... fixing units for EXFatemp')
                    print (f'... old: {v.units}')
                    v.setncattr("units", 'degree_K')
                    print (f'... new: {v.units}')
                    metadata_updated = True

                else:
                    print (f'... units are identical {v.units} -- not fixing units!')
            else:
                print (f'... EXFatemp not in granule -- not fixing units!')


            # fix comment on select variables
            print('\n>> fixing comment')
            vars_to_fix = set(nc_dvs).intersection(set(comment_keys))
            if len(vars_to_fix) > 0:
                for dv in vars_to_fix:
                    v = tmp_ds.variables[dv]
                    print(f'... fixing comment for {dv}')
                    v.setncattr("comment", comment_fix[dv]["comment"])
                    print(f'... new {v.comment}')
                    metadata_updated = True
            else:
                print(f'... granule has no variables with updated comments -- not updating')


            # fix minmax
            print('\n>> fixing minmax:')
            if ds_id in minmax.keys():
                # get variables in the minmax dataset
                minmax_dvs = list(minmax[ds_id].keys())

                print('minmax keys: ', minmax_dvs)

                # loop through all variables in the minmax dictionary
                for minmax_dv in minmax_dvs:
                    if minmax_dv in nc_dvs:
                        print(f'\n{minmax_dv} found in nc_dvs')
                        # get a pointer to the this data variable
                        v = tmp_ds.variables[minmax_dv]

                        # pull out the current valid min max attributes
                        old_valid_min = v.valid_min
                        old_valid_max = v.valid_max

                        # get the valid min and max for this variable
                        new_valid_min = minmax[ds_id][minmax_dv]['valid_min']
                        new_valid_max = minmax[ds_id][minmax_dv]['valid_max']

                        # QC section
                        # roll the dice
                        qc_rand = random.random()

                        if qc_rand < qc_prob:
                            # load actual vmin vmax here (expensive IO)
                            v_min = np.nanmin(v[:])
                            v_max = np.nanmax(v[:])

                            print(f'   QC act min/max        : {v_min:.12} {v_max:.12}')
                            print(f'   QC old valid_min/max  : {old_valid_min:.12} {old_valid_max:.12}')
                            print(f'   QC new valid_min/max  : {new_valid_min:.12} {new_valid_max:.12}')

                            if (old_valid_min > v_min) or (old_valid_max < v_max):
                                print('   QC old valid min/max was wrong')
                                print(f'     1. old valid min >= vmin {old_valid_min:.12} {v_min:.12}\
                                    {old_valid_min > v_min}')
                                print(f'     2. old valid max <= vmax {old_valid_max:.12} {v_max:.12}\
                                    {old_valid_max < v_max}')
                            else:
                                print('   QC ... old valid min/max was ok')

                            if (new_valid_min > v_min) or (new_valid_max < v_max):
                                print(f'\n+ FAILURE: new valid min/max is wrong in {ds_id} {minmax_dv}')
                                print(f'   1. new valid min >= vmin {new_valid_min:.12} {v_min:.12} \
                                    {new_valid_min > v_min}')
                                print(f'   2. new valid max <= vmax {new_valid_max:.12} {v_max:.12} \
                                    {new_valid_max < v_max}')
                                return -1

                            else:
                                print('   QC ... new valid min/max is ok')
                            print('\n')

                        if (new_valid_min == old_valid_min) and (new_valid_max == old_valid_max):
                            print('... new and old valid min/max are identical!')
                            print('... not updating valid min/max!')

                        else:
                            v.setncattr("valid_min", new_valid_min)
                            v.setncattr("valid_max", new_valid_max)

                            print('... new and old valid min/max are different!')
                            print(f'... old/new valid min {old_valid_min:.12} {v.valid_min:.12}')
                            print(f'... old/new valid max {old_valid_max:.12} {v.valid_max:.12}')


                    else:
                        print(f'\n+ FAILURE: minmax key {minmax_dv} not in granule variables {nc_dvs}: {ecco_filename.name}')
                        return -1

            else:
                print(f'\n!!!! granule id not found in minmax keys {ds_id}')
                return -1


            # update the reference to the URS approved version
            print ('\n>> fixing references')
            tmp_ds.setncattr('references', \
                             'ECCO Consortium, Fukumori, I., Wang, O., Fenty, I., Forget, G., Heimbach, P., & Ponte, R. M. 2020. Synopsis of the ECCO Central Production Global Ocean and Sea-Ice State Estimate (Version 4 Release 4). doi:10.5281/zenodo.3765928')
            print ('\n>> fixing source')
            tmp_ds.setncattr('source', \
                             'The ECCO V4r4 state estimate was produced by fitting a free-running solution of the MITgcm (checkpoint 66g) to satellite and in situ observational data in a least squares sense using the adjoint method')

            # fix coordinate comment typo
            print ('\n>> fixing coordinates comment')
            tmp_ds.setncattr('coordinates_comment', "Note: the global 'coordinates' attribute describes auxillary coordinates.")


            # update date of modified metadata
            print ('\n>> updating date modified')
            current_time = datetime.datetime.now().isoformat()[0:19]
            tmp_ds.setncattr('date_modified', current_time)
            tmp_ds.setncattr('date_metadata_modified', current_time)

            # alphabetically sort all attributes
            print ('\n>> sorting attributes')
            sorted_attr_dict = sort_attrs(tmp_ds.__dict__)

            print ('\n>> removing/replacing global attributes')
            # delete all attributes one at a time
            for attr in tmp_ds.ncattrs():
                tmp_ds.delncattr(attr)

            # replace all one at a time (in alphabetical order)
            for attr in sorted_attr_dict:
                tmp_ds.setncattr(attr, sorted_attr_dict[attr])

            print(f"\n+ SUCCESS: changes applied {ecco_filename.name}\n")
            return 1


    except Exception as e:
        raise e

    print('could not open file!')
    return -1


def f1(ecco_files, minmax, comment_fix, summary_fix, qc_prob):
    results = []
    for ecco_filename in ecco_files:
        result = apply_fixes(ecco_filename, minmax, comment_fix, summary_fix, qc_prob)
        results.append(result)
    return results




#############
def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_base_dir', type=str, required=True,\
                        help='directory containing dataset grouping subdirectories')

    parser.add_argument('--grid_type', type=str, required=True,\
                        choices=['native','lat-lon', '1D'],\
                        help='')

    parser.add_argument('--time_type', type=str, required=True,\
                        choices=['day_inst','mon_mean','day_mean','hourly','time_invariant','grid_type'],\
                        help='')

    parser.add_argument('--summary_fix_dir', type=str, required=True,\
                        help='directory containing summary dictionary')


    parser.add_argument('--grouping_id', type=int, required=True,\
                        help='which grouping num to process (int)')


    parser.add_argument('--valid_minmax_dir', type=str, required=True,\
                        help='valid minmax dir')


    parser.add_argument('--qc_prob', type=float, required=False, default=0.0,\
                        help='probability [0..1] of running a quality control check on minmax')


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
    valid_minmax_dir = Path(args.valid_minmax_dir)
    summary_fix_dir = Path(args.summary_fix_dir)
    qc_prob = args.qc_prob

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
    print('summary_fix_dir', summary_fix_dir)
    print('valid_minmax_dir', valid_minmax_dir)
    print('qc_prob', qc_prob)
    print('debug', debug)
    print('\n')


    comment_fix = dict()
    comment_fix["oceSPDep"] = dict()
    comment_fix["oceSPDep"]['filename'] = "SEA_ICE_SALT_PLUME_FLUX"
    comment_fix["oceSPDep"]['comment'] = "Depth of parameterized salt plumes formed due to brine rejection during sea-ice formation."

    comment_fix["RHOAnoma"] = dict()
    comment_fix["RHOAnoma"]['filename'] = "OCEAN_DENS_STRAT_PRESS"
    comment_fix["RHOAnoma"]['comment'] = "In-situ seawater density anomaly relative to the reference density, rhoConst. rhoConst = 1029 kg m-3"

    comment_fix["SALT"] = dict()
    comment_fix["SALT"]['filename'] = "OCEAN_TEMPERATURE_SALINITY"
    comment_fix["SALT"]['comment'] = "Defined using CF convention 'Sea water salinity is the salt content of sea water, often on the Practical Salinity Scale of 1978. However, the unqualified term 'salinity' is generic and does not necessarily imply any particular method of calculation. The units of salinity are dimensionless and the units attribute should normally be given as 1e-3 or 0.001 i.e. parts per thousand.' see https://cfconventions.org/Data/cf-standard-names/73/build/cf-standard-name-table.html"

    comment_fix["SIsnPrcp"] = dict()
    comment_fix["SIsnPrcp"]['filename'] = "OCEAN_AND_ICE_SURFACE_FW_FLUX"
    comment_fix["SIsnPrcp"]['comment'] = "Snow precipitation rate over sea-ice, averaged over the entire model grid cell."


    #%%
    print("get groupings")
    groupings = get_groupings(dataset_base_dir, grid_type, time_type)

    for gi in groupings.keys():
        print(gi, groupings[gi]['name'])

    if grouping_id >= 0:
        grouping_ids = [grouping_id]
    else:
        grouping_ids = list(range(len(groupings)))

    print('\ngrouping ids to process: ', grouping_ids)


    print('loading summary fix')
    with open(summary_fix_dir / 'ECCOv4r4_dataset_summary.json') as f:
        summary_fix = json.load(f)
        #pprint(len(summary_fix.keys()))
        #pprint(list(summary_fix.keys()))

    # load minmax
    print('loading minmax')
    minmax = load_valid_minmax(valid_minmax_dir)
    #print(len(minmax.keys()))
    #pprint(list(minmax.keys()))

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
            ecco_files = ecco_files[2:4]
        print(f'# files to process { len(ecco_files)} ')

        print('computing')
        start_time = time.time()
        f1_out = f1(ecco_files, minmax, comment_fix, summary_fix, qc_prob)
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
