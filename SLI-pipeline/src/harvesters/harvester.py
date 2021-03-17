import os
import hashlib
import logging
from datetime import datetime
from urllib.request import urlopen, urlcleanup
from xml.etree.ElementTree import parse
import requests
import yaml
from requests.auth import HTTPBasicAuth


log = logging.getLogger(__name__)


def md5(fname):
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq, sort=''):
    """
    Queries Solr database using the filter query passed in.
    Returns list of Solr entries that satisfies the query.
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    query_params = {'q': '*:*',
                    'fq': fq,
                    'rows': 300000}

    if sort:
        query_params['sort'] = sort

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def print_resp(resp, msg=''):
    if resp.status_code == 200:
        print(f'Successfully created or updated Solr {msg}')
    else:
        print(f'Failed to create or update Solr {msg}')


def solr_update(config, update_body):
    """
    Posts an update to Solr database with the update body passed in.
    For each item in update_body, a new entry is created in Solr, unless
    that entry contains an id, in which case that entry is updated with new values.
    Optional return of the request status code (ex: 200 for success)
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


def podaac_harvester(config, docs, target_dir):
    """
    Downloads and creates new harvested doc for each data granule within a given date range
    for a given dataset hosted on PODAAC.
    """
    # =====================================================
    # Setup PODAAC loop variables
    # =====================================================

    ds_name = config['ds_name']
    date_regex = config['date_regex']
    host = config['host']
    shortname = config['original_dataset_short_name']

    now = datetime.utcnow()
    now_str = now.strftime(date_regex)

    start_time = config['start']
    end_time = now.strftime("%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []
    last_success = {}

    if config["podaac_id"]:
        url_base = f'{host}&datasetId={config["podaac_id"]}'
    else:
        url_base = f'{host}&shortName={shortname}'

    url = f'{url_base}&endTime={end_time}&startTime={start_time}'

    namespace = {"podaac": "http://podaac.jpl.nasa.gov/opensearch/",
                 "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                 "atom": "http://www.w3.org/2005/Atom",
                 "georss": "http://www.georss.org/georss",
                 "gml": "http://www.opengis.net/gml",
                 "dc": "http://purl.org/dc/terms/",
                 "time": "http://a9.com/-/opensearch/extensions/time/1.0/"}

    # =====================================================
    # PODAAC loop
    # =====================================================
    while True:
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
                        mod_date_time = datetime.strptime(mod_time, date_regex)
                except:
                    mod_time = str(now)
                    mod_date_time = now

                # Granule metadata used for Solr harvested entries
                item = {
                    'type_s': 'harvested',
                    'date_dt': date_start_str,
                    'dataset_s': ds_name,
                    'filename_s': newfile,
                    'source_s': link,
                    'modified_time_dt': mod_date_time.strftime(date_regex)
                }

                # If granule doesn't exist or previously failed or has been updated since last harvest
                updating = (newfile not in docs.keys()) or \
                           (not docs[newfile]['harvest_success_b']) or \
                           (datetime.strptime(
                               docs[newfile]['download_time_dt'], date_regex) <= mod_date_time)

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
                        resp = requests.get(link)
                        open(local_fp, 'wb').write(resp.content)

                    # If file exists locally, but is out of date, download it
                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                        print(f' - Updating {newfile} and downloading to {local_fp}')
                        urlcleanup()
                        resp = requests.get(link)
                        open(local_fp, 'wb').write(resp.content)

                    else:
                        print(f' - {newfile} already downloaded and up to date')

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
                    item['checksum_s'] = ''
                    item['granule_file_path_s'] = ''
                    item['file_size_l'] = 0

            if updating:
                item['download_time_dt'] = now_str
                entries_for_solr.append(item)

                if item['harvest_success_b']:
                    last_success = item

        # Check if more granules are available on next page
        next_page = xml.find("{%(atom)s}link[@rel='next']" % namespace)
        if next_page is None:
            print(f'\nDownloading {ds_name} complete\n')
            break

        url = next_page.attrib['href']

    return entries_for_solr, url_base, last_success


def local_harvester(config, docs, target_dir):
    """
    Creates new harvested doc for each file in a given directory.
    """
    ds_name = config['ds_name']
    date_regex = config['date_regex']
    source = 'Locally stored file'
    now = datetime.utcnow()
    now_str = now.strftime(date_regex)

    start_time = config['start']
    end_time = now.strftime("%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []
    last_success = {}

    # Get local files
    data_files = []
    start = start_time[:8]
    end = end_time[:8]

    for _, _, files in os.walk(target_dir):
        for data_file in files:
            if '.DS_Store' in data_file:
                continue

            f_date = data_file[7:15]
            if f_date < start or f_date > end:
                continue

            data_files.append(data_file)

    for data_file in data_files:
        date = data_file[7:-3]
        date_start_str = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z'
        year = date_start_str[:4]
        local_fp = f'{target_dir}{year}/{data_file}'
        mod_time = datetime.fromtimestamp(os.path.getmtime(local_fp))
        mod_time_string = mod_time.strftime(date_regex)

        # Granule metadata used for Solr harvested entries
        item = {
            'type_s': 'harvested',
            'date_dt': date_start_str,
            'dataset_s': ds_name,
            'filename_s': data_file,
            'source_s': source,
            'modified_time_dt': mod_time_string
        }

        if data_file in docs.keys():
            item['id'] = docs[data_file]['id']

        # If granule doesn't exist or previously failed or has been updated since last harvest
        updating = (data_file not in docs.keys()) or \
                   (not docs[data_file]['harvest_success_b']) or \
                   (docs[data_file]['download_time_dt'] <= mod_time_string)

        if updating:
            print(f' - Adding {data_file}.')

            # Create checksum for file
            item['checksum_s'] = md5(local_fp)
            item['granule_file_path_s'] = local_fp
            item['harvest_success_b'] = True
            item['file_size_l'] = os.path.getsize(local_fp)
            item['download_time_dt'] = now_str

            entries_for_solr.append(item)

            if item['harvest_success_b']:
                last_success = item
        else:
            print(f' - {data_file} already up to date.')

    return entries_for_solr, source, last_success


def harvester(config_path='', output_path=''):
    """
    Pulls data files for PODAAC id and date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset and harvested granules.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    ds_name = config['ds_name']
    shortname = config['original_dataset_short_name']
    date_regex = config['date_regex']

    target_dir = f'{output_path}{ds_name}/harvested_granules/'

    # If target paths don't exist, make them
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f'Downloading {ds_name} files to {target_dir}\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================

    # Query for existing harvested docs
    harvested_docs = solr_query(config, ['type_s:harvested', f'dataset_s:{ds_name}'])

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    docs = {}
    if harvested_docs:
        docs = {doc['filename_s']: doc for doc in harvested_docs}

    now_str = datetime.utcnow().strftime(date_regex)

    # Actual downloading and generation of harvested docs for Solr
    if config['harvester_type'] == 'podaac':
        entries_for_solr, source, last_success = podaac_harvester(config, docs, target_dir)
    elif config['harvester_type'] == 'local':
        entries_for_solr, source, last_success = local_harvester(config, docs, target_dir)

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        resp = solr_update(config, entries_for_solr)
        print_resp(resp, msg='harvested documents')

    # =====================================================
    # Solr dataset entry
    # =====================================================

    # Query for Solr failed harvest documents
    fq = ['type_s:harvested', f'dataset_s:{ds_name}', 'harvest_success_b:false']
    failed_harvesting = solr_query(config, fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:harvested', f'dataset_s:{ds_name}', 'harvest_success_b:true']
    successful_harvesting = solr_query(config, fq, sort='date_dt asc')

    if not successful_harvesting:
        harvest_status = 'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'
    else:
        harvest_status = 'All granules successfully harvested'

    ds_start = successful_harvesting[0]['date_dt'] if successful_harvesting else None
    ds_end = successful_harvesting[-1]['date_dt'] if successful_harvesting else None
    last_dl = last_success['download_time_dt'] if last_success else None

    # Query for Solr Dataset-level Document
    dataset_query = solr_query(config, ['type_s:dataset', f'dataset_s:{ds_name}'])

    if not dataset_query:
        # -----------------------------------------------------
        # Create Solr dataset entry
        # -----------------------------------------------------

        ds_meta = {
            'type_s': 'dataset',
            'dataset_s': ds_name,
            'start_date_dt': ds_start,
            'end_date_dt': ds_end,
            'short_name_s': shortname,
            'source_s': source,
            'last_checked_dt': now_str,
            'last_download_dt': last_dl,
            'harvest_status_s': harvest_status,
            'original_dataset_title_s': config['original_dataset_title'],
            'original_dataset_short_name_s': shortname,
            'original_dataset_url_s': config['original_dataset_url'],
            'original_dataset_reference_s': config['original_dataset_reference'],
            'original_dataset_doi_s': config['original_dataset_doi']
        }

    else:
        # -----------------------------------------------------
        # Update Solr dataset entry
        # -----------------------------------------------------
        ds_meta = dataset_query[0]
        ds_meta['last_checked_dt'] = {"set": now_str}

        if entries_for_solr:
            ds_meta['start_date_dt'] = {"set": ds_start}
            ds_meta['end_date_dt'] = {"set": ds_end}
            ds_meta['harvest_status_s'] = {"set": harvest_status}

            if last_success:
                ds_meta['last_download_dt'] = {"set": last_success['download_time_dt']}

    # Update Solr with modified dataset entry
    resp = solr_update(config, [ds_meta])
    print_resp(resp, msg='dataset document\n')
