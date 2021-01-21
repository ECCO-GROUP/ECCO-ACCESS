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


def solr_query(config, solr_host, fq, solr_collection_name):
    """
    Queries Solr database using the filter query passed in.
    Returns list of Solr entries that satisfies the query.
    """

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, solr_host, update_body, solr_collection_name, r=False):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def harvester(config_path='', output_path='', s3=None, solr_info=''):
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
    host = config['host']
    podaac_id = config['podaac_id']
    shortname = config['original_dataset_short_name']
    name = config['name']
    password = config['password']
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

    if solr_info:
        solr_host = solr_info['solr_url']
        solr_collection_name = solr_info['solr_collection_name']
    else:
        solr_host = config['solr_host_local']
        solr_collection_name = config['solr_collection_name']
    # clean_solr(config, solr_host, grids_to_use, solr_collection_name)
    print(f'Downloading {dataset_name} files to {target_dir}\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs = {}

    # Query for existing harvested docs
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    harvested_docs = solr_query(config, solr_host, fq, solr_collection_name)

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    if len(harvested_docs) > 0:
        for doc in harvested_docs:
            docs[doc['filename_s']] = doc

    # =====================================================
    # Setup PODAAC loop variables
    # =====================================================
    if podaac_id:
        url = f'{host}&datasetId={podaac_id}'
    else:
        url = f'{host}&shortName={shortname}'

    url += f'&endTime={end_time}&startTime={start_time}'

    namespace = {"podaac": "http://podaac.jpl.nasa.gov/opensearch/",
                 "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                 "atom": "http://www.w3.org/2005/Atom",
                 "georss": "http://www.georss.org/georss",
                 "gml": "http://www.opengis.net/gml",
                 "dc": "http://purl.org/dc/terms/",
                 "time": "http://a9.com/-/opensearch/extensions/time/1.0/"}

    next = None
    more = True

    # =====================================================
    # PODAAC loop
    # =====================================================
    while more:
        xml = parse(urlopen(url))
        items = xml.findall('{%(atom)s}entry' % namespace)
        # Loop through all granules in XML returned from URL
        for elem in items:
            updating = False

            # Extract granule information from XML entry and attempt to download data file
            try:
                # Extract download link from XML entry
                try:
                    link = elem.find(
                        "{%(atom)s}link[@title='OPeNDAP URL']" % namespace).attrib['href']
                    link = '.'.join(link.split('.')[:-1])
                    newfile = link.split("/")[-1]
                except:
                    link = elem.find(
                        "{%(atom)s}link[@title='HTTP URL']" % namespace).attrib['href']
                    newfile = link.split("/")[-1]

                # Extract start and end dates from XML entry
                date_start_str = elem.find("{%(time)s}start" % namespace).text
                date_end_str = elem.find("{%(time)s}end" % namespace).text

                # Remove nanoseconds from dates
                if len(date_start_str) > 19:
                    date_start_str = date_start_str[:19] + 'Z'
                if len(date_end_str) > 19:
                    date_end_str = date_end_str[:19] + 'Z'

                # Attempt to get last modified time of file on podaac
                # Not all PODAAC datasets contain last modified time
                try:
                    mod_time = elem.find("{%(atom)s}updated" % namespace).text
                    try:
                        mod_date_time = datetime.strptime(
                            mod_time, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except:
                        mod_date_time = datetime.strptime(
                            mod_time, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    mod_time = str(now)
                    mod_date_time = now

                # Granule metadata used for Solr harvested entries
                item = {}
                item['type_s'] = 'harvested'
                item['date_s'] = date_start_str
                item['dataset_s'] = dataset_name
                item['filename_s'] = newfile
                item['source_s'] = link
                item['modified_time_dt'] = mod_date_time.strftime(time_format)

                # If granule doesn't exist or previously failed or has been updated since last harvest
                updating = (newfile not in docs.keys()) or \
                    (not docs[newfile]['harvest_success_b']) or \
                    (datetime.strptime(
                        docs[newfile]['download_time_dt'], time_format) <= mod_date_time)

                # If updating, download file if necessary
                if updating:
                    year = date_start_str[:4]
                    local_fp = f'{target_dir}{year}/{newfile}'

                    if not os.path.exists(f'{target_dir}{year}'):
                        os.makedirs(f'{target_dir}{year}')

                    # If file doesn't exist locally, download it
                    if not os.path.exists(local_fp):
                        print(f' - Downloading {newfile} to {local_fp}')

                        urlcleanup()
                        if password:
                            r = requests.get(
                                link, auth=HTTPBasicAuth(name, password))
                        else:
                            r = requests.get(link)
                        open(local_fp, 'wb').write(r.content)

                    # If file exists locally, but is out of date, download it
                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                        print(
                            f' - Updating {newfile} and downloading to {local_fp}')
                        urlcleanup()
                        if password:
                            r = requests.get(
                                link, auth=HTTPBasicAuth(name, password))
                        else:
                            r = requests.get(link)
                        open(local_fp, 'wb').write(r.content)

                    else:
                        print(
                            f' - {newfile} already downloaded and up to date')

                    if newfile in docs.keys():
                        item['id'] = docs[newfile]['id']

                    # Create checksum for file
                    item['checksum_s'] = md5(local_fp)
                    item['granule_file_path_s'] = local_fp
                    item['harvest_success_b'] = True
                    item['file_size_l'] = os.path.getsize(local_fp)

                else:
                    print(f' - {newfile} already downloaded and up to date')

            except Exception as e:
                print(f'    - {e}')
                if updating:

                    print(f'    - {newfile} failed to download')

                    item['harvest_success_b'] = False
                    item['granule_file_path_s'] = ''
                    item['file_size_l'] = 0

            if updating:
                item['download_time_dt'] = chk_time

                entries_for_solr.append(item)

                start_times.append(datetime.strptime(
                    date_start_str, date_regex))
                end_times.append(datetime.strptime(
                    date_end_str, date_regex))

                if item['harvest_success_b']:
                    last_success_item = item

        # Check if more granules are available on next page
        # Should only need next if more than 30000 granules exist
        # Hemispherical seaice data should have roughly 30*365*2=21900
        next = xml.find("{%(atom)s}link[@rel='next']" % namespace)
        if next is None:
            more = False
            print(f'\nDownloading {dataset_name} complete\n')
        else:
            url = next.attrib['href']

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        r = solr_update(config, solr_host, entries_for_solr,
                        solr_collection_name, r=True)
        if r.status_code == 200:
            print('Successfully created or updated Solr harvested documents')
        else:
            print('Failed to create Solr harvested documents')

    # Query for Solr failed harvest documents
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
          f'harvest_success_b:false']
    failed_harvesting = solr_query(config, solr_host, fq, solr_collection_name)

    # Query for Solr successful harvest documents
    fq = ['type_s:harvested',
          f'dataset_s:{dataset_name}', f'harvest_success_b:true']
    successful_harvesting = solr_query(
        config, solr_host, fq, solr_collection_name)

    harvest_status = f'All granules successfully harvested'

    if not successful_harvesting:
        harvest_status = f'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'

    overall_start = min(start_times) if start_times else None
    overall_end = max(end_times) if end_times else None

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_query = solr_query(config, solr_host, fq, solr_collection_name)

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
        ds_meta['source_s'] = f'{host}&datasetId={podaac_id}'
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
        r = solr_update(config, solr_host, [ds_meta],
                        solr_collection_name, r=True)

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
                   'fl': 'date_s',
                   'rows': 300000}

        url = f'{solr_host}{solr_collection_name}/select?'
        response = requests.get(url, params=getVars)
        dates = [x['date_s'] for x in response.json()['response']['docs']]

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
        r = solr_update(config, solr_host, [
                        update_doc], solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document\n')
        else:
            print('Failed to update Solr dataset document\n')
