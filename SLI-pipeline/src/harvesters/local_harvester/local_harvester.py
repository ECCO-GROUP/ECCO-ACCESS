import os
import sys
import gzip
import yaml
import shutil
import hashlib
import logging
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import xarray as xr
from pathlib import Path
from xml.etree.ElementTree import parse
from datetime import datetime, timedelta
from urllib.request import urlopen, urlcleanup, urlretrieve

log = logging.getLogger(__name__)


def clean_solr(config, solr_host, grids_to_use, solr_collection_name):
    """
    Remove harvested entries in Solr for dates outside of config date range.
    Also remove related aggregations, and force aggregation rerun for those years.
    """
    dataset_name = config['ds_name']
    config_start = config['start']
    config_end = config['end']

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_query(config, solr_host, fq, solr_collection_name)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

    # Convert config dates to Solr format
    config_start = f'{config_start[:4]}-{config_start[4:6]}-{config_start[6:]}'
    config_end = f'{config_end[:4]}-{config_end[4:6]}-{config_end[6:]}'

    fq = [f'type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_metadata = solr_query(config, solr_host, fq, solr_collection_name)

    if not dataset_metadata:
        return
    else:
        dataset_metadata = dataset_metadata[0]

    print(
        f'Removing Solr documents related to dates outside of configuration start and end dates: \n\t{config_start} to {config_end}.\n')

    # Remove entries earlier than config start date
    fq = f'dataset_s:{dataset_name} AND date_s:[* TO {config_start}}}'
    url = f'{solr_host}{solr_collection_name}/update?commit=true'
    requests.post(url, json={'delete': {'query': fq}})

    # Remove entries later than config end date
    fq = f'dataset_s:{dataset_name} AND date_s:{{{config_end} TO *]'
    url = f'{solr_host}{solr_collection_name}/update?commit=true'
    requests.post(url, json={'delete': {'query': fq}})

    # Forces the bounding years to be re-aggregated to account for potential
    # removed dates
    start_year = config_start[:4]
    end_year = config_end[:4]
    update_body = [{"id": dataset_metadata['id']}]

    for grid in grids:
        solr_grid_years = f'{grid}_years_updated_ss'
        if solr_grid_years in dataset_metadata.keys():
            years = dataset_metadata[solr_grid_years]
        else:
            years = []
        if start_year not in years:
            years.append(start_year)
        if end_year not in years:
            years.append(end_year)

        update_body[0][solr_grid_years] = {"set": years}

    if grids:
        solr_update(config, solr_host, update_body, solr_collection_name)


def md5(fname):
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq):
    """
    Queries Solr database using the filter query passed in.
    Returns list of Solr entries that satisfies the query.
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, update_body, r=False):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def harvester(config_path='', output_path=''):
    """
    Pulls data files for PODAAC id and date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset and harvested granules.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    if not config_path:
        print('No path for configuration file. Can not run harvester.')
        return

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    dataset_name = config['ds_name']
    date_regex = config['date_regex']
    start_time = config['start']

    if config['most_recent']:
        end_time = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")
    else:
        end_time = config['end']

    shortname = config['original_dataset_short_name']

    target_dir = f'{output_path}{dataset_name}/harvested_granules/'

    # If target paths don't exist, make them
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    time_format = "%Y-%m-%dT%H:%M:%SZ"
    entries_for_solr = []
    last_success_item = {}
    start_times = []
    end_times = []
    chk_time = datetime.utcnow().strftime(time_format)
    now = datetime.utcnow()
    updating = False

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    # clean_solr(config, solr_host, grids_to_use, solr_collection_name)
    print(f'Adding {dataset_name} files in {target_dir} to Solr.\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs = {}

    # Query for existing harvested docs
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    harvested_docs = solr_query(config, fq)

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    if len(harvested_docs) > 0:
        for doc in harvested_docs:
            docs[doc['filename_s']] = doc

    # =====================================================
    # Get local files
    # =====================================================

    data_files = []

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if '.DS_Store' in f:
                continue
            data_files.append(f)

    data_files.sort()

    # =====================================================
    # Main loop
    # =====================================================
    for data_file in data_files:
        # "2021-02-17T18:26:20Z"
        date = data_file[7:-3]
        date_start_str = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z'
        date_end_str = f'{date[:4]}-{date[4:6]}-{date[6:8]}T23:59:59Z'
        year = date_start_str[:4]
        local_fp = f'{target_dir}{year}/{data_file}'
        mod_time = datetime.fromtimestamp(os.path.getmtime(local_fp))
        mod_time_string = mod_time.strftime(time_format)

        # Granule metadata used for Solr harvested entries
        item = {}
        item['type_s'] = 'harvested'
        item['date_dt'] = date_start_str
        item['dataset_s'] = dataset_name
        item['filename_s'] = data_file
        item['source_s'] = 'Locally stored file'
        item['modified_time_dt'] = mod_time_string

        if data_file in docs.keys():
            item['id'] = docs[data_file]['id']

        # If granule doesn't exist or previously failed or has been updated since last harvest
        updating = (data_file not in docs.keys()) or \
                   (not docs[data_file]['harvest_success_b']) or \
                   (docs[data_file]['download_time_dt'] <= mod_time_string)

        if updating:
            # print(f' - {data_file} already downloaded.')

            # Create checksum for file
            item['checksum_s'] = md5(local_fp)
            item['granule_file_path_s'] = local_fp
            item['harvest_success_b'] = True
            item['file_size_l'] = os.path.getsize(local_fp)
            item['download_time_dt'] = chk_time

            entries_for_solr.append(item)

            start_times.append(datetime.strptime(date_start_str, date_regex))
            end_times.append(datetime.strptime(date_end_str, date_regex))

            if item['harvest_success_b']:
                last_success_item = item

    # =====================================================
    # Add granule documents to Solr
    # =====================================================

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        r = solr_update(config, entries_for_solr, r=True)
        if r.status_code == 200:
            print('Successfully created or updated Solr harvested documents')
        else:
            print('Failed to create Solr harvested documents')

    # Query for Solr failed harvest documents
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
          f'harvest_success_b:false']
    failed_harvesting = solr_query(config, fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:harvested',
          f'dataset_s:{dataset_name}', f'harvest_success_b:true']
    successful_harvesting = solr_query(config, fq)

    harvest_status = f'All granules successfully harvested'

    if not successful_harvesting:
        harvest_status = f'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'

    overall_start = min(start_times) if start_times else None
    overall_end = max(end_times) if end_times else None

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_query = solr_query(config, fq)

    # If dataset entry exists on Solr
    update = (len(dataset_query) == 1)

    # =====================================================
    # Solr dataset entry
    # =====================================================
    if not update:
        # -----------------------------------------------------
        # Create Solr dataset entry
        # -----------------------------------------------------
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = dataset_name
        ds_meta['short_name_s'] = shortname
        ds_meta['source_s'] = f'Locally stored files'
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = shortname
        ds_meta['original_dataset_url_s'] = config['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']

        # Only include start_date and end_date if there was at least one successful download
        if overall_start != None:
            ds_meta['start_date_dt'] = overall_start.strftime(time_format)
            ds_meta['end_date_dt'] = overall_end.strftime(time_format)

        # Only include last_download_dt if there was at least one successful download
        if last_success_item:
            ds_meta['last_download_dt'] = last_success_item['download_time_dt']

        ds_meta['harvest_status_s'] = harvest_status

        # Update Solr with dataset metadata
        r = solr_update(config, [ds_meta], r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

    # if dataset entry exists, update download time, converage start date, coverage end date
    else:
        # -----------------------------------------------------
        # Update Solr dataset entry
        # -----------------------------------------------------
        dataset_metadata = dataset_query[0]

        # Query for dates of all harvested docs
        getVars = {'q': '*:*',
                   'fq': [f'dataset_s:{dataset_name}', 'type_s:harvested', 'harvest_success_b:true'],
                   'fl': 'date_dt',
                   'rows': 300000}

        url = f'{solr_host}{solr_collection_name}/select?'
        response = requests.get(url, params=getVars)
        dates = [x['date_dt'] for x in response.json()['response']['docs']]

        # Build update document body
        update_doc = {}
        update_doc['id'] = dataset_metadata['id']
        update_doc['last_checked_dt'] = {"set": chk_time}
        if dates:
            update_doc['start_date_dt'] = {"set": min(dates)}
            update_doc['end_date_dt'] = {"set": max(dates)}

        if entries_for_solr:
            update_doc['harvest_status_s'] = {"set": harvest_status}

            if 'download_time_dt' in last_success_item.keys():
                update_doc['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}

        # Update Solr with modified dataset entry
        r = solr_update(config, [update_doc], r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document\n')
        else:
            print('Failed to update Solr dataset document\n')
