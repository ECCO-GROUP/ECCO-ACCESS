import os
import hashlib
import logging
from datetime import datetime

from xml.etree.ElementTree import fromstring
import requests
import yaml


log = logging.getLogger(__name__)


def md5(fname):
    """
    Creates md5 checksum from file

    Params:
        fpath (str): path of the file

    Returns:
        hash_md5.hexdigest (str): double length string containing only hexadecimal digits
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq, sort=''):
    """
    Queries Solr database using the filter query passed in.

    Params:
        config (dict): the dataset specific config file
        fq (List[str]): the list of filter query arguments

    Returns:
        response.json()['response']['docs'] (List[dict]): the Solr docs that satisfy the query
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
    """
    Prints Solr response message

    Params:
        resp (Response): the response object from a solr update
        msg (str): the specific message to print
    """
    if resp.status_code == 200:
        print(f'Successfully created or updated Solr {msg}')
    else:
        print(f'Failed to create or update Solr {msg}')


def solr_update(config, update_body):
    """
    Updates Solr database with list of docs. If a doc contains an existing id field,
    Solr will update or replace that existing doc with the new doc.

    Params:
        config (dict): the dataset specific config file
        update_body (List[dict]): the list of docs to update on Solr

    Returns:
        requests.post(url, json=update_body) (Response): the Response object from the post call
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


def podaac_harvester(config, docs, target_dir):
    """
    Harvests new or updated granules from PODAAC for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    Params:
        config (dict): the dataset specific config file
        docs (dict): the existing granule docs on Solr in dict format
        target_dir (str): the path of the dataset's harvested granules directory

    Returns:
        entries_for_solr (List[dict]): all new or modified granule docs to be posted to Solr
        url_base (str): PODAAC url for the specific dataset
    """

    ds_name = config['ds_name']
    shortname = config['original_dataset_short_name']

    now = datetime.utcnow()
    date_regex = config['date_regex']
    start_time = config['start']
    end_time = now.strftime("%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []

    if config['podaac_id']:
        url_base = f'{config["host"]}&datasetId={config["podaac_id"]}'
    else:
        url_base = f'{config["host"]}&shortName={shortname}'

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
        print('Loading granule entries from PODAAC XML document...')

        xml = fromstring(requests.get(url).text)
        items = xml.findall('{%(atom)s}entry' % namespace)

        # Loop through all granules in XML returned from URL
        for elem in items:
            updating = False

            # Extract granule information from XML entry and attempt to download data file

            # Extract download link from XML entry
            link = elem.find("{%(atom)s}link[@title='OPeNDAP URL']" % namespace).attrib['href'][:-5]
            filename = link.split("/")[-1]
            # Extract start date from XML entry
            date_start_str = elem.find("{%(time)s}start" % namespace).text[:19] + 'Z'

            # Extract modified time of file on podaac
            mod = elem.find("{%(atom)s}updated" % namespace)
            mod_time_str = mod.text

            # Granule metadata used for Solr granule entries
            item = {
                'type_s': 'granule',
                'date_dt': date_start_str,
                'dataset_s': ds_name,
                'filename_s': filename,
                'source_s': link,
                'modified_time_dt': mod_time_str
            }

            if filename in docs.keys():
                item['id'] = docs[filename]['id']

            year = date_start_str[:4]
            local_fp = f'{target_dir}{year}/{filename}'

            if not os.path.exists(f'{target_dir}{year}'):
                os.makedirs(f'{target_dir}{year}')

            # If granule doesn't exist or previously failed or has been updated since last harvest
            # or exists in Solr but doesn't exist where it should
            updating = (filename not in docs.keys()) or \
                (not docs[filename]['harvest_success_b']) or \
                (docs[filename]['download_time_dt'] <= mod_time_str) or \
                (not os.path.exists(local_fp))

            # If updating, download file if necessary
            if updating:
                try:
                    expected_size = requests.head(link).headers.get('content-length', -1)
                    local_mod_time = datetime.fromtimestamp(
                        os.path.getmtime(local_fp)).strftime(date_regex)

                    # Only redownloads if local file is out of town - doesn't waste
                    # time/bandwidth to redownload the same file just because there isn't
                    # a Solr entry. Most useful during development.
                    if not os.path.exists(local_fp) or mod_time_str > local_mod_time:
                        print(f' - Downloading {filename} to {local_fp}')

                        resp = requests.get(link)
                        open(local_fp, 'wb').write(resp.content)
                    else:
                        print(f' - {filename} already downloaded, but not in Solr.')

                    # Create checksum for file
                    item['checksum_s'] = md5(local_fp)
                    item['granule_file_path_s'] = local_fp
                    item['file_size_l'] = os.path.getsize(local_fp)

                    # Make sure file properly downloaded by comparing sizes
                    if expected_size == item['file_size_l']:
                        item['harvest_success_b'] = True
                    else:
                        item['harvest_success_b'] = False

                except Exception as e:
                    print(f'    - {e}')
                    print(f'    - {filename} failed to download')

                    item['harvest_success_b'] = False
                    item['checksum_s'] = ''
                    item['granule_file_path_s'] = ''
                    item['file_size_l'] = 0

                item['download_time_dt'] = now.strftime(date_regex)
                entries_for_solr.append(item)

            else:
                print(f' - {filename} already downloaded, and up to date in Solr.')

        # Check if more granules are available on next page
        next_page = xml.find("{%(atom)s}link[@rel='next']" % namespace)
        if next_page is None:
            print(f'\nDownloading {ds_name} complete\n')
            break

        url = next_page.attrib['href'] + '&itemsPerPage=10000'

    return entries_for_solr, url_base


def local_harvester(config, docs, target_dir):
    """
    Harvests new or updated granules from a local drive for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    Params:
        config (dict): the dataset specific config file
        docs (dict): the existing granule docs on Solr in dict format
        target_dir (str): the path of the dataset's harvested granules directory

    Returns:
        entries_for_solr (List[dict]): all new or modified granule metadata docs to be posted to Solr
        source (str): denotes granule/dataset was harvested from a local directory
    """
    ds_name = config['ds_name']
    date_regex = config['date_regex']
    source = 'Locally stored file'
    now = datetime.utcnow()
    now_str = now.strftime(date_regex)

    start_time = config['start']
    end_time = now.strftime("%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []

    # Get local files
    data_files = []
    start = start_time[:8]
    end = end_time[:8]

    for _, _, files in os.walk(target_dir):
        for filename in files:
            if '.DS_Store' in filename:
                continue

            f_date = filename[7:15]
            if f_date < start or f_date > end:
                continue

            data_files.append(filename)

    for filename in data_files:
        date = filename[7:-3]
        date_start_str = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z'
        year = date_start_str[:4]
        local_fp = f'{target_dir}{year}/{filename}'
        mod_time = datetime.fromtimestamp(os.path.getmtime(local_fp))
        mod_time_string = mod_time.strftime(date_regex)

        # Granule metadata used for Solr granule entries
        item = {
            'type_s': 'granule',
            'date_dt': date_start_str,
            'dataset_s': ds_name,
            'filename_s': filename,
            'source_s': source,
            'modified_time_dt': mod_time_string
        }

        if filename in docs.keys():
            item['id'] = docs[filename]['id']

        # If granule doesn't exist or previously failed or has been updated since last harvest
        updating = (filename not in docs.keys()) or \
                   (not docs[filename]['harvest_success_b']) or \
                   (docs[filename]['download_time_dt'] <= mod_time_string)

        if updating:
            print(f' - Adding {filename} to Solr.')

            # Create checksum for file
            item['checksum_s'] = md5(local_fp)
            item['granule_file_path_s'] = local_fp
            item['harvest_success_b'] = True
            item['file_size_l'] = os.path.getsize(local_fp)
            item['download_time_dt'] = now_str

            entries_for_solr.append(item)

        else:
            print(f' - {filename} already up to date in Solr.')

    return entries_for_solr, source


def harvester(config_path='', output_path=''):
    """
    Harvests new or updated granules from a local drive for a dataset. Posts granule metadata docs
    to Solr and creates or updates dataset metadata doc.
    dataset doc.

    Params:
        config_path (dict): the dataset specific config file
        output_path (dict): the existing granule docs on Solr in dict format
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    ds_name = config['ds_name']
    shortname = config['original_dataset_short_name']

    target_dir = f'{output_path}{ds_name}/harvested_granules/'

    # If target paths don't exist, make them
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f'Harvesting {ds_name} files to {target_dir}\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================

    # Query for existing granule docs
    harvested_docs = solr_query(config, ['type_s:granule', f'dataset_s:{ds_name}'])

    # Dictionary of existing granule docs
    # granule filename : solr entry for that doc
    docs = {}
    if harvested_docs:
        docs = {doc['filename_s']: doc for doc in harvested_docs}

    now_str = datetime.utcnow().strftime(config['date_regex'])

    # Actual downloading and generation of granule docs for Solr
    if config['harvester_type'] == 'podaac':
        entries_for_solr, source = podaac_harvester(config, docs, target_dir)
    elif config['harvester_type'] == 'local':
        entries_for_solr, source = local_harvester(config, docs, target_dir)

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        resp = solr_update(config, entries_for_solr)
        print_resp(resp, msg='harvested documents')

    # =====================================================
    # Solr dataset entry
    # =====================================================

    # Query for Solr failed harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:false']
    failed_harvesting = solr_query(config, fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true']
    successful_harvesting = solr_query(config, fq, sort='date_dt asc')

    if not successful_harvesting:
        harvest_status = 'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'
    else:
        harvest_status = 'All granules successfully harvested'

    # Query for Solr Dataset-level Document
    dataset_query = solr_query(config, ['type_s:dataset', f'dataset_s:{ds_name}'])

    ds_start = successful_harvesting[0]['date_dt'] if successful_harvesting else None
    ds_end = successful_harvesting[-1]['date_dt'] if successful_harvesting else None

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true']
    successful_harvesting = solr_query(config, fq, sort='download_time_dt desc')

    last_dl = successful_harvesting[0]['download_time_dt'] if successful_harvesting else None

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

    if dataset_query:
        ds_meta['id'] = dataset_query[0]['id']

    # Update Solr with modified dataset entry
    resp = solr_update(config, [ds_meta])
    print_resp(resp, msg='dataset document\n')
