"""
Created on Fri May 10 17:27:09 2019

@author: ifenty"""

import sys
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
sys.path.append('/home/ifenty/git_repos_others/ECCO-GROUP/ECCOv4-py')

from importlib import reload
import ecco_v4_py as ecco
import ecco_cloud_utils as ea

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


def find_podaac_metadata(podaac_dataset_table, filename, debug=False):
    """Return revised file metadata based on an input ECCO `filename`.

    This should consistently parse a filename that conforms to the
    ECCO filename conventions and match it to a row in my metadata
    table.

    """
    # Use filename components to find metadata row from podaac_dataset_table.
    if "_snap_" in filename:
        head, tail = filename.split("_snap_")
        head = f"{head}_snap"
    elif "_day_mean_" in filename:
        head, tail = filename.split("_day_mean_")
        head = f"{head}_day_mean"
    elif "_mon_mean_" in filename:
        head, tail = filename.split("_mon_mean_")
        head = f"{head}_mon_mean"
    else:
        raise Exception("Error: filename may not conform to ECCO V4r4 convention.")

    if debug:
        print('split filename into head', head)
        print('split filename into tail', tail)

    tail = tail.split("ECCO_V4r4_")[1]

    if debug:
        print('further split tail into ',  tail)

    # Get the filenames column from my table as a list of strings.
    names = podaac_dataset_table['DATASET.FILENAME']

    # Find the index of the row with a filename with matching head, tail chars.
    index = names.apply(lambda x: all([x.startswith(head), x.endswith(tail)]))

    # Select that row from podaac_dataset_table table and make a copy of it.
    metadata = podaac_dataset_table[index].iloc[0].to_dict()

    podaac_metadata = {
        'id': metadata['DATASET.PERSISTENT_ID'].replace("PODAAC-","10.5067/"),
        'metadata_link': f"https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName={metadata['DATASET.SHORT_NAME']}",
        'title': metadata['DATASET.LONG_NAME'],
    }
    if debug:
        print('\n... podaac metadata:')
        pprint(podaac_metadata)

    return podaac_metadata

#%%
def apply_podaac_metadata(xrds, podaac_metadata):
    """Apply attributes podaac_metadata to ECCO dataset and its variables.

    Attributes that are commented `#` are retained with no modifications.
    Attributes that are assigned `None` are dropped.
    New attributes added to dataset.

    """
    # REPLACE GLOBAL ATTRIBUTES WITH NEW/UPDATED DICTIONARY.
    atts = xrds.attrs
    for name, modifier in podaac_metadata.items():
        if callable(modifier):
            atts.update({name: modifier(x=atts[name])})

        elif modifier is None:
            if name in atts:
                del atts[name]
        else:
            atts.update({name: modifier})

    xrds.attrs = atts

    # MODIFY VARIABLE ATTRIBUTES IN A LOOP.
    for v in xrds.variables:
        if "gmcd_keywords" in xrds.variables[v].attrs:
            del xrds.variables[v].attrs['gmcd_keywords']

    return xrds  # Return the updated xarray Dataset.
    #return apply_podaac_metadata  # Return the function.


def sort_attrs(attrs):
    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    for k in keys:
        od[k] = attrs[k]

    return od


def sort_all_attrs(G, print_output=False):
    for coord in list(G.coords):
        if print_output:
            print(coord)
        new_attrs = sort_attrs(G[coord].attrs)
        G[coord].attrs = new_attrs

    for dv in list(G.data_vars):
        if print_output:
            print(dv)
        new_attrs = sort_attrs(G[dv].attrs)
        G[dv].attrs = new_attrs

    new_attrs = sort_attrs(G.attrs)
    G.attrs = new_attrs

    return G

#%%
def generate_netcdfs(output_freq_code,
                     product_type,
                     grouping_to_process,
                     output_dir_base,
                     diags_root,
                     metadata_json_dir,
                     podaac_dir,
                     ecco_grid_dir,
                     ecco_grid_filename,
                     array_precision = np.float32,
                     debug_mode=False):


    #%%
    # output_freq_codes  one of either 'AVG_DAY' or 'AVG_MON'
    #
    # time_steps_to_process : one of
    # 'all'
    # a list [1,3,4,...]
    # 'by_job'

    # grouping to process:
    #   can be either
    #   a single number
    #   'by_job' determine from job id and num_jobs


    # ECCO FIELD INPUT DIRECTORY
    # model diagnostic output
    # subdirectories must be
    #  'diags_all/diag_mon'
    #  'diags_all/diag_mon'
    #  'diags_all/snap'

    print('\nBEGIN: generate_netcdfs')
    print('OFC',output_freq_code)
    print('PDT', product_type)
    print('GTP',grouping_to_process)
    print('DBG',debug_mode)
    print('\n')


    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999

    ecco_start_time = np.datetime64('1992-01-01T12:00:00')
    ecco_end_time   = np.datetime64('2017-12-31T12:00:00')

    filename_tail_latlon = '_ECCO_V4r4_latlon_0p50deg.nc'
    filename_tail_native = '_ECCO_V4r4_native_llc0090.nc'
    filename_tail_1D = '_ECCO_V4r4_1D.nc'

    metadata_fields = ['ECCOv4r4_global_metadata_for_all_datasets',
                       'ECCOv4r4_global_metadata_for_latlon_datasets',
                       'ECCOv4r4_global_metadata_for_native_datasets',
                       'ECCOv4r4_global_metadata_for_1D_datasets',
                       'ECCOv4r4_coordinate_metadata_for_1D_datasets',
                       'ECCOv4r4_coordinate_metadata_for_latlon_datasets',
                       'ECCOv4r4_coordinate_metadata_for_native_datasets',
                       'ECCOv4r4_geometry_metadata_for_latlon_datasets',
                       'ECCOv4r4_geometry_metadata_for_native_datasets',
                       'ECCOv4r4_groupings_for_1D_datasets',
                       'ECCOv4r4_groupings_for_latlon_datasets',
                       'ECCOv4r4_groupings_for_native_datasets',
                       'ECCOv4r4_variable_metadata',
                       'ECCOv4r4_variable_metadata_for_latlon_datasets',
                       'ECCOv4r4_variable_metadata_for_1D_datasets']


    #%% -- -program start


    # Define fill values for binary and netcdf
    # ---------------------------------------------
    if array_precision == np.float32:
        binary_output_dtype = '>f4'
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif array_precision == np.float64:
        binary_output_dtype = '>f8'
        netcdf_fill_value = nc4.default_fillvals['f8']

    print('\nLOADING METADATA')
    # load METADATA
    metadata = dict()

    for mf in metadata_fields:
        mf_e = mf + '.json'
        print(mf_e)
        with open(str(metadata_json_dir / mf_e), 'r') as fp:
            metadata[mf] = json.load(fp)


    # metadata for different variables
    global_metadata_for_all_datasets = metadata['ECCOv4r4_global_metadata_for_all_datasets']
    global_metadata_for_1D_datasets = metadata['ECCOv4r4_global_metadata_for_1D_datasets']
    global_metadata_for_latlon_datasets = metadata['ECCOv4r4_global_metadata_for_latlon_datasets']
    global_metadata_for_native_datasets = metadata['ECCOv4r4_global_metadata_for_native_datasets']

    coordinate_metadata_for_1D_datasets     = metadata['ECCOv4r4_coordinate_metadata_for_1D_datasets']
    coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
    coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']

    geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
    geometry_metadata_for_native_datasets = metadata['ECCOv4r4_geometry_metadata_for_native_datasets']

    groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
    groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
    groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

    variable_metadata_1D = metadata['ECCOv4r4_variable_metadata_for_1D_datasets']

    variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']
    variable_metadata = metadata['ECCOv4r4_variable_metadata']

    #%%
    # load PODAAC fields
    podaac_dataset_table = read_csv(podaac_dir / 'PODAAC_datasets-revised_20210226.4.csv')


    # load ECCO grid
    ecco_grid = xr.open_dataset(ecco_grid_dir / ecco_grid_filename)

    print(ecco_grid)

    #%%
    print('\nproduct type', product_type)

    if product_type == '1D':
        #dataset_description_tail = dataset_description_tail_1D
        filename_tail = filename_tail_1D
        groupings = groupings_for_1D_datasets
        output_dir_type = output_dir_base / '1D'
        global_metadata = global_metadata_for_all_datasets + global_metadata_for_1D_datasets

    # show groupings
    print('\nAll groupings')
    for gi, gg in enumerate(groupings_for_1D_datasets):
        print('\t', gi, gg['name'])

    #%%
    print('\nGetting directories for group variables')
    if output_freq_code == 'AVG_DAY':
        period_suffix = 'day_mean'

    elif output_freq_code == 'AVG_MON':
        period_suffix = 'mon_mean'

    elif output_freq_code == 'SNAPSHOT':
        period_suffix = 'day_inst'
    else:
        print('valid options are AVG_DAY, AVG_MON, SNAPSHOT')
        print('you provided ', output_freq_code)
        sys.exit()

    print('...output_freq_code ', output_freq_code)

    output_dir_freq = output_dir_type / period_suffix
    print('...making output_dir freq ', output_dir_freq)

    # make output directory
    if not output_dir_freq.exists():
        try:
            output_dir_freq.mkdir()
            print('... made %s ' % output_dir_freq)
        except:
            print('...cannot make %s ' % output_dir_freq)


    print('... using provided grouping ', grouping_to_process)
    grouping_num = grouping_to_process


    ######################################################
    # all super custom for v4r4.
    # load's ou 1D NeCDF files and trusts their time levels
    # summary text is explicitly provided instead of made
    # on the fly
    if grouping_to_process == 0: # Pressure
        G = xr.open_dataset(diags_root / 'Pa_global.nc')
        G_orig = G.copy(deep=True)

        G = G.Pa_global
        # nan out the first two misisng values
        G.values[0:2] = np.nan
        G.attrs.clear()

        if output_freq_code == 'AVG_MON':
            G.resample(time='1m').mean()
        if output_freq_code == 'AVG_DAY':
            G.resample(time='1D').mean()

        G = G.to_dataset()
        summary_text = 'This dataset provides instantaneous hourly global mean atmospheric surface pressure over the ocean and sea-ice from the ECCO Version 4 Release 4 (V4r4) ocean and sea-ice state estimate. Estimating the Circulation and Climate of the Ocean (ECCO) state estimates are data-constrained, dynamically and kinematically-consistent reconstructions of the three-dimensional, time-evolving ocean, sea-ice, and surface atmospheric states and fluxes. ECCO V4r4 is a free-running solution of a global, nominally 1-degree configuration of the MIT general circulation model (MITgcm) that has been fit to observations in a least-squares sense. Observational data constraints used in V4r4 include sea surface height (SSH) from satellite altimeters [ERS-1/2, TOPEX/Poseidon, GFO, ENVISAT, Jason-1,2,3, CryoSat-2, and SARAL/AltiKa]; sea surface temperature (SST) from satellite radiometers [AVHRR], sea surface salinity (SSS) from the Aquarius satellite radiometer/scatterometer, ocean bottom pressure (OBP) from the GRACE satellite gravimeter; sea ice concentration from satellite radiometers [SSM/I and SSMIS], and in-situ ocean temperature and salinity measured with conductivity-temperature-depth (CTD) sensors and expendable bathythermographs (XBTs) from several programs [e.g., WOCE, GO-SHIP, Argo, and others] and platforms [e.g., research vessels, gliders, moorings, ice-tethered profilers, and instrumented pinnipeds]. V4r4 covers the period 1992-01-01T12:00:00 to 2018-01-01T00:00:00.'

    if grouping_to_process == 1: # GMSL
        if output_freq_code == 'AVG_MON':
            G = xr.open_dataset(diags_root / 'GMSL.nc')
            G_orig = G.copy(deep=True)

            summary_text = 'This dataset provides monthly-averaged global mean sea-level anomalies including barystatic and sterodynamic terms from the ECCO Version 4 Release 4 (V4r4) ocean and sea-ice state estimate. Estimating the Circulation and Climate of the Ocean (ECCO) state estimates are data-constrained, dynamically and kinematically-consistent reconstructions of the three-dimensional, time-evolving ocean, sea-ice, and surface atmospheric states and fluxes. ECCO V4r4 is a free-running solution of a global, nominally 1-degree configuration of the MIT general circulation model (MITgcm) that has been fit to observations in a least-squares sense. Observational data constraints used in V4r4 include sea surface height (SSH) from satellite altimeters [ERS-1/2, TOPEX/Poseidon, GFO, ENVISAT, Jason-1,2,3, CryoSat-2, and SARAL/AltiKa]; sea surface temperature (SST) from satellite radiometers [AVHRR], sea surface salinity (SSS) from the Aquarius satellite radiometer/scatterometer, ocean bottom pressure (OBP) from the GRACE satellite gravimeter; sea ice concentration from satellite radiometers [SSM/I and SSMIS], and in-situ ocean temperature and salinity measured with conductivity-temperature-depth (CTD) sensors and expendable bathythermographs (XBTs) from several programs [e.g., WOCE, GO-SHIP, Argo, and others] and platforms [e.g., research vessels, gliders, moorings, ice-tethered profilers, and instrumented pinnipeds]. V4r4 covers the period 1992-01-01T12:00:00 to 2018-01-01T00:00:00.'

        if output_freq_code == 'AVG_DAY':
            G = xr.open_dataset(diags_root / 'GMSL_day.nc')
            G_orig = G.copy(deep=True)

            summary_text = 'This dataset provides daily-averaged global mean sea-level anomalies including barystatic and sterodynamic terms from the ECCO Version 4 Release 4 (V4r4) ocean and sea-ice state estimate. Estimating the Circulation and Climate of the Ocean (ECCO) state estimates are data-constrained, dynamically and kinematically-consistent reconstructions of the three-dimensional, time-evolving ocean, sea-ice, and surface atmospheric states and fluxes. ECCO V4r4 is a free-running solution of a global, nominally 1-degree configuration of the MIT general circulation model (MITgcm) that has been fit to observations in a least-squares sense. Observational data constraints used in V4r4 include sea surface height (SSH) from satellite altimeters [ERS-1/2, TOPEX/Poseidon, GFO, ENVISAT, Jason-1,2,3, CryoSat-2, and SARAL/AltiKa]; sea surface temperature (SST) from satellite radiometers [AVHRR], sea surface salinity (SSS) from the Aquarius satellite radiometer/scatterometer, ocean bottom pressure (OBP) from the GRACE satellite gravimeter; sea ice concentration from satellite radiometers [SSM/I and SSMIS], and in-situ ocean temperature and salinity measured with conductivity-temperature-depth (CTD) sensors and expendable bathythermographs (XBTs) from several programs [e.g., WOCE, GO-SHIP, Argo, and others] and platforms [e.g., research vessels, gliders, moorings, ice-tethered profilers, and instrumented pinnipeds]. V4r4 covers the period 1992-01-01T12:00:00 to 2018-01-01T00:00:00.'
            # nan out the first time level because the barystatic sea level array as a discontinuity at t=0
            for gdv in G.data_vars:
                G[gdv].values[0] = np.nan

        for gdv in G.data_vars:
            G[gdv] = G[gdv] - G[gdv].mean()


    if grouping_to_process == 2: #SBO
        G = xr.open_dataset(diags_root / 'SBO_global.nc')
        # nan out the first time level because the barystatic sea level array as a discontinuity at t=0

        summary_text = 'This dataset provides instantaneous hourly core products of the IERS Special Bureau for the Oceans from the ECCO Version 4 Release 4 (V4r4) ocean and sea-ice state estimate. Estimating the Circulation and Climate of the Ocean (ECCO) state estimates are data-constrained, dynamically and kinematically-consistent reconstructions of the three-dimensional, time-evolving ocean, sea-ice, and surface atmospheric states and fluxes. ECCO V4r4 is a free-running solution of a global, nominally 1-degree configuration of the MIT general circulation model (MITgcm) that has been fit to observations in a least-squares sense. Observational data constraints used in V4r4 include sea surface height (SSH) from satellite altimeters [ERS-1/2, TOPEX/Poseidon, GFO, ENVISAT, Jason-1,2,3, CryoSat-2, and SARAL/AltiKa]; sea surface temperature (SST) from satellite radiometers [AVHRR], sea surface salinity (SSS) from the Aquarius satellite radiometer/scatterometer, ocean bottom pressure (OBP) from the GRACE satellite gravimeter; sea ice concentration from satellite radiometers [SSM/I and SSMIS], and in-situ ocean temperature and salinity measured with conductivity-temperature-depth (CTD) sensors and expendable bathythermographs (XBTs) from several programs [e.g., WOCE, GO-SHIP, Argo, and others] and platforms [e.g., research vessels, gliders, moorings, ice-tethered profilers, and instrumented pinnipeds]. V4r4 covers the period 1992-01-01T12:00:00 to 2018-01-01T00:00:00.'
        G_orig = G.copy(deep=True)

        for gdv in G.data_vars:
            if ('com' in gdv) or ('amp' in gdv) or ('mass' in gdv):
                G[gdv].values[0:1] = np.nan


    ######################################################
    for dv in G.data_vars:
        G[dv].attrs = dict()

    G.attrs = dict()

    grouping = groupings[grouping_num]
    print('... grouping to use ', grouping['name'])
    print('... fields in grouping ', grouping['fields'])

    # dimension of dataset
    dataset_dim = grouping['dimension']
    print('... grouping dimension', dataset_dim)


    # define empty list of gcmd keywords pertaining to this dataset
    grouping_gcmd_keywords = []

#%%

    # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
    print('\n... adding metadata specific to the variable')
    G, grouping_gcmd_keywords = \
        ecco.add_variable_metadata(variable_metadata, G, grouping_gcmd_keywords)

    print('\n... using 1D dataseta metadata specific to the variable')
    G, grouping_gcmd_keywords = \
        ecco.add_variable_metadata(variable_metadata_1D, G, grouping_gcmd_keywords)

    for dv in G.data_vars:
        print(G[dv])

    pprint(G)
#%%

    print('\n... adding coordinate metadata for 1D dataset')
    G = ecco.add_coordinate_metadata(coordinate_metadata_for_1D_datasets,G)
    pprint(G)

#%%
    # ADD GLOBAL METADATA
    print("\n... adding global metadata for all datasets")
    G = ecco.add_global_metadata(global_metadata_for_all_datasets, G,\
                            dataset_dim)
    print('\n... adding global metadata for 1D dataset')
    G = ecco.add_global_metadata(global_metadata_for_1D_datasets, G,\
                                    dataset_dim)

    pprint(G)
    #%%

    # ADD GLOBAL METADATA ASSOCIATED WITH TIME AND DATE
    print('\n... adding time / data global attrs')
    G.attrs['time_coverage_start'] = str(ecco_start_time)[0:19]
    G.attrs['time_coverage_end'] = str(ecco_end_time)[0:19]

    pprint(G)

    #%%
    # current time and date
    current_time = datetime.datetime.now().isoformat()[0:19]
    G.attrs['date_created'] = current_time
    G.attrs['date_modified'] = current_time
    G.attrs['date_metadata_modified'] = current_time
    G.attrs['date_issued'] = current_time
    pprint(G)

    #%%
    # add coordinate attributes to the variables
    dv_coordinate_attrs = dict()

    for dv in list(G.data_vars):
        dv_coords_orig = set(list(G[dv].coords))

        # REMOVE TIME STEP FROM LIST OF COORDINATES (PODAAC REQUEST)
        #coord_attrs[coord] = coord_attrs[coord].split('time_step')[0].strip()
        #data_var_coorcoord_attrs[coord].split()
        set_intersect = dv_coords_orig.intersection(set(['XC','YC','XG','YG','Z','Zp1','Zu','Zl','time'])) #

        dv_coordinate_attrs[dv] = " ".join(set_intersect)

    #%%
    print('\n... creating variable encodings')
    # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
    dv_encoding = dict()
    for dv in G.data_vars:
        dv_encoding[dv] =  {'zlib':True, \
                            'complevel':5,\
                            'shuffle':True,\
                            '_FillValue':netcdf_fill_value}

        # overwrite default coordinates attribute (PODAAC REQUEST)
        G[dv].encoding['coordinates'] = dv_coordinate_attrs[dv]

    #%%
    # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
    print('\n... creating coordinate encodings')
    coord_encoding = dict()

    for coord in G.coords:
        # default encoding: no fill value, float32
        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}

        if (G[coord].values.dtype == np.int32) or \
           (G[coord].values.dtype == np.int64) :
            coord_encoding[coord]['dtype'] ='int32'

        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in G[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = G[coord].attrs['units']
                # delete from the attributes list
                del G[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # MERGE ENCODINGS for coordinates and variables
    encoding = {**dv_encoding, **coord_encoding}

    if 'keywords' not in G.attrs:
        G.attrs['keywords'] = ""
    pprint(G.keywords)
    #%%
    # MERGE GCMD KEYWORDS
    print('\n... merging GCMD keywords')
    common_gcmd_keywords = G.keywords.split(',')
    gcmd_keywords_list = set(grouping_gcmd_keywords + common_gcmd_keywords)

    gcmd_keyword_str = ''
    for gcmd_keyword in gcmd_keywords_list:
        if len(gcmd_keyword_str) == 0:
            gcmd_keyword_str = gcmd_keyword
        else:
            gcmd_keyword_str += ', ' + gcmd_keyword

    #print(gcmd_keyword_str)
    G.attrs['keywords'] = gcmd_keyword_str
    pprint(G.attrs)

    #%%
    ## ADD FINISHING TOUCHES

    # uuic
    print('\n... adding uuid')
    G.attrs['uuid'] = str(uuid.uuid1())

    # add any dataset grouping specific comments.
    if 'comment' in grouping:
        G.attrs['comment'] = G.attrs['comment'] + ' ' \
        + grouping['comment']

    # set the long name of the time attribute
    if 'AVG' in output_freq_code:
        G.time.attrs['long_name'] = 'center time of averaging period'
    else:
        G.time.attrs['long_name'] = 'snapshot time'

    # set averaging period duration and resolution
    print('\n... setting time coverage resolution')
    print(output_freq_code)
    # --- AVG DAY
    if output_freq_code == 'AVG_MON':
        G.attrs['time_coverage_duration'] = 'P1M'
        G.attrs['time_coverage_resolution'] = 'P1M'

        #date_str = str(np.datetime64(G.time.values[0],'M'))
        ppp_tttt = 'mon_mean'

    # --- AVG DAY
    elif output_freq_code == 'AVG_DAY':
        G.attrs['time_coverage_duration'] = 'P1D'
        G.attrs['time_coverage_resolution'] = 'P1D'

        #date_str = str(np.datetime64(G.time.values[0],'D'))
        ppp_tttt = 'day_mean'

    # --- SNAPSHOT
    elif output_freq_code == 'SNAPSHOT':
        G.attrs['time_coverage_duration'] = 'P0S'
        G.attrs['time_coverage_resolution'] = 'P0S'

        if 'bounds' in G.time.attrs:
            G.time.attrs.pop('bounds')

        # convert from oroginal
        #   '1992-01-16T12:00:00.000000000'
        # to new format
        # '1992-01-16T120000'
        #date_str = str(G.time.values[0])[0:19].replace(':','')
        ppp_tttt = 'snap'

    print(ppp_tttt)
    #%%
    ## construct filename
    print('\n... creating filename')

    filename = grouping['filename'] + '_' + ppp_tttt + \
        filename_tail

    print(filename)
    # make subdirectory for the grouping
    output_dir = output_dir_freq / grouping['filename']
    print('\n... creating output_dir', output_dir)

    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True,exist_ok=True)
        except:
            print ('cannot make %s ' % output_dir)

    #%%
    # create full pathname for netcdf file
    netcdf_output_filename = output_dir / filename

    # add product name attribute = filename
    G.attrs['product_name'] = filename

    # add summary attribute = description of dataset

    G.attrs['summary'] = summary_text
    # get podaac metadata based on filename

    #print('\n... getting PODAAC metadata')
    podaac_metadata = \
       find_podaac_metadata(podaac_dataset_table, filename, debug=True)

    # apply podaac metadata based on filename
    print('\n... applying PODAAC metadata')
    pprint(podaac_metadata)
    G = apply_podaac_metadata(G, podaac_metadata)

    # sort comments alphabetically
    print('\n... sorting global attributes')
    G.attrs = sort_attrs(G.attrs)

    # add one final comment (PODAAC request)
    G.attrs["coordinates_comment"] = \
        "Note: the global 'coordinates' attribute descibes auxillary coordinates."

    #%%
    # SAVE
    print('\n... saving to netcdf ', netcdf_output_filename)
    G.to_netcdf(netcdf_output_filename, encoding=encoding)
    G.close()

    print('\n... checking existence of new file: ', netcdf_output_filename.exists())
    print('\n')

    GG = xr.open_dataset(netcdf_output_filename)
    GG.close()
        #%%
    return G,G_orig




#%%
if __name__ == "__main__":

    reload(ecco)

    #%%
    mapping_factors_dir = Path('/home/ifenty/tmp/ecco-v4-podaac-mapping-factors')

    diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/diags_all')

    ## METADATA
    metadata_json_dir = Path('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')
    podaac_dir = metadata_json_dir #Path('/home/ifenty/git_repos_others/ecco-data-pub/metadata')

    ecco_grid_dir = Path('/home/ifenty/data/grids/grid_ECCOV4r4')
    ecco_grid_dir_mds = Path('/home/ifenty/data/grids/grid_ECCOV4r4')
    ecco_grid_filename = 'ECCO_V4r4_llc90_grid_geometry.nc'

    #%%
    # Define precision of output files, float32 is standard
    # ------------------------------------------------------
    array_precision = np.float32


    # 1D Groupings
    #    0 Pressure
    # 	 1 GMSL
    # 	 2 SBO




    #%%
    G = []

    product_type = '1D'
    output_dir_base = Path('/home/ifenty/tmp/1D_20210306')
    if not output_dir_base.exists():
        output_dir_base.mkdir()


    Gs = []
    G_os = []

    for    product in  range(4):

        if product == 0:
            # pressure
            output_freq_code = 'SNAPSHOT'
            grouping_to_process = 0

        elif product== 1:
            #GMSL
            output_freq_code = 'AVG_MON'
            grouping_to_process = 1

        elif product == 2:
            output_freq_code = 'AVG_DAY'
            grouping_to_process = 1

        elif product == 3:
            output_freq_code = 'SNAPSHOT'
            grouping_to_process = 2

        debug_mode = True


        diags_root = Path('/home/ifenty/ian1/ifenty/ECCOv4/binary_output/scalar_time_series/netcdf')



        print('\n\n===================================')
        print('output_freq_code', output_freq_code)
        print('product_type', product_type)

        g, go = generate_netcdfs(output_freq_code,
                             product_type,
                             grouping_to_process,
                             output_dir_base,
                             diags_root,
                             metadata_json_dir,
                             podaac_dir,
                             ecco_grid_dir,
                             ecco_grid_filename,
                             array_precision,
                             debug_mode)

        Gs.append(g)
        G_os.append(go)

#%%
import matplotlib.pyplot as plt

thumb_dir = output_dir_base / 'thumbs'
try:
    thumb_dir.mkdir()
except:
    print('already exists')
    print(thumb_dir.exists())


thumbnail_height = 5.5
thumbnail_width_to_height_ratio = 2


fig_ref = plt.figure(1,clear=True)

legs = []
for dv in Gs[0].data_vars:
    legs.append(dv)
    plt.plot(Gs[0].time, Gs[0][dv])
plt.grid()
plt.title(Gs[0].title)
plt.ylabel('Pressure [N m-2]')


name = Gs[0].id.split('/')[1]
fname = name + '.jpg'
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)
fpath = thumb_dir / fname
plt.savefig(fpath, dpi=150, facecolor='w', bbox_inches='tight', pad_inches = 0.05)
print(fpath)

#%%
fig_ref = plt.figure(2,clear=True)

legs = []
for dv in Gs[1].data_vars:
    legs.append(dv)
    plt.plot(Gs[1].time, Gs[1][dv])
plt.legend(legs)
plt.grid()
plt.title(Gs[1].title)
plt.ylabel('Sea-Level Anomaly [m]')

name = Gs[1].id.split('/')[1]
fname = name + '.jpg'
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)
fpath = thumb_dir / fname
plt.savefig(fpath, dpi=150, facecolor='w', bbox_inches='tight', pad_inches = 0.05)

print(fpath)

fig_ref = plt.figure(3,clear=True)
legs = []
for dv in Gs[2].data_vars:
    legs.append(dv)
    plt.plot(Gs[2].time, Gs[2][dv])

plt.legend(legs)
plt.grid()
plt.title(Gs[2].title)
plt.ylabel('Sea-Level Anomaly [m]')

name = Gs[2].id.split('/')[1]
fname = name + '.jpg'
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)
fpath = thumb_dir / fname
plt.savefig(fpath, dpi=150, facecolor='w', bbox_inches='tight', pad_inches = 0.05)
print(fpath)
#%%

fig_ref = plt.figure(4,clear=True)
legs = []
fs = ['xoamp']
for dv in Gs[3].data_vars:
    if dv in fs:
        plt.plot(Gs[3].time, Gs[3][dv])
        legs.append(dv)

plt.legend(legs)
plt.grid()
plt.title(Gs[3].title)
plt.ylabel('component of oam due to freshwater flux [kg m2 s-1]')


name = Gs[3].id.split('/')[1]
fname = name + '.jpg'
fig_ref.set_size_inches(thumbnail_width_to_height_ratio*thumbnail_height, thumbnail_height)
fpath = thumb_dir / fname
plt.savefig(fpath, dpi=150, facecolor='w', bbox_inches='tight', pad_inches = 0.05)
print(fpath)


#%%


for dv in Gs[1].data_vars:
    print(dv)
    print(Gs[1][dv].mean())
    Gs[1][dv][0:12].plot()
plt.legend(legs)

plt.figure(4,clear=True)
for dv in Gs[2].data_vars:
    print(dv)
    print(Gs[2][dv].mean())
    Gs[2][dv][0:365].plot()
plt.legend(legs)

#%%

plt.figure(5,clear=True)
tmp = Gs[2]
tmp2 = tmp['global_mean_sea_level_anomaly'] -\
    tmp['global_mean_sterodynamic_sea_level_anomaly'] - tmp['global_mean_barystatic_sea_level_anomaly']
tmp2.plot()
plt.title('residual : GMSL - stereo - bary')


plt.figure(6,clear=True)
tmp = Gs[1]
tmp2 = tmp['global_mean_sea_level_anomaly'] -\
    tmp['global_mean_sterodynamic_sea_level_anomaly'] - tmp['global_mean_barystatic_sea_level_anomaly']
tmp2.plot()
plt.title('residual : GMSL - stereo - bary')




#%%
plt.close(10)
fig, axs = plt.subplots(6,5,num=10,figsize=(26,20), sharex=True, clear=True)
axs = axs.ravel()
tmp = Gs[3]
for dvi, dv in enumerate(Gs[3].data_vars):
    axs[dvi].plot(Gs[3].time[:], Gs[3][dv][:])
    axs[dvi].set_title(dv, fontsize=10, y=1, pad=-14)
    axs[dvi].yaxis.get_offset_text().set_fontsize(6)

plt.tight_layout()


plt.close(11)
fig, axs = plt.subplots(6,5,num=11,figsize=(26,20), sharex=True, clear=True)
axs = axs.ravel()
tmp = Gs[3]
for dvi, dv in enumerate(Gs[3].data_vars):
    axs[dvi].plot(Gs[3].time[1:], Gs[3][dv][1:])
    axs[dvi].set_title(dv, fontsize=10, y=1, pad=-14)
    axs[dvi].yaxis.get_offset_text().set_fontsize(6)

plt.tight_layout()
