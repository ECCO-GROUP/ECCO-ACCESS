import logging
import logging.config
import os
import re
from datetime import datetime
from ftplib import FTP
import time

import requests
from dateutil import parser
from utils import file_utils, solr_utils

logging.config.fileConfig('logs/log.ini', disable_existing_loggers=False)
log = logging.getLogger(__name__)


def valid_date(e, config):
    file_date = file_utils.get_date(config['regex'], e)

    start = config['start'][:8]
    end = str(datetime.now)[
        :8] if config['end'] == 'NOW' else config['end'][:8]

    if file_date >= start and file_date <= end:
        return True
    return False


def granule_update_check(docs, filename, mod_date_time, time_format):
    key = filename.replace('.NRT', '')

    # Granule hasn't been harvested yet
    if key not in docs.keys():
        return True

    entry = docs[key]

    # Granule failed harvesting previously
    if not entry['harvest_success_b']:
        return True

    # Granule has been updated since last harvest
    if datetime.strptime(entry['download_time_dt'], time_format) <= mod_date_time:
        return True

    # Granule is replacing NRT version
    if '.NRT' in entry['filename_s'] and '.NRT' not in filename:
        return True

    # Granule is up to date
    return False


def ifremer_ftp_harvester(config, output_path, grids_to_use=[], s3=None, on_aws=False):
    """
    Pulls data files for ifremer FTP id and date range given in harvester_config.yaml.
    If not on_aws, saves locally, else saves to s3 bucket.
    Creates (or updates) Solr entries for dataset, harvested granule, fields,
    and descendants.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    dataset_name = config['ds_name']
    start_time = config['start']
    end_time = config['end']
    host = config['host']
    ddir = config['ddir']

    if end_time == 'NOW':
        end_time = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")

    target_dir = f'{output_path}/{dataset_name}/harvested_granules/'
    folder = f'/tmp/{dataset_name}/'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    time_format = "%Y-%m-%dT%H:%M:%SZ"
    entries_for_solr = []
    last_success_item = {}
    granule_dates = []
    chk_time = datetime.utcnow().strftime(time_format)
    now = datetime.utcnow()
    updating = False
    aws_upload = False

    # =====================================================
    # Setup AWS Target Bucket
    # =====================================================
    if on_aws:
        target_bucket_name = config['target_bucket_name']
        target_bucket = s3.Bucket(target_bucket_name)
        solr_host = config['solr_host_aws']
        solr_collection_name = config['solr_collection_name']
        solr_utils.clean_solr(config, grids_to_use)
        print(
            f'Downloading {dataset_name} files and uploading to {target_bucket_name}/{dataset_name}\n')
    else:
        solr_host = config['solr_host_local']
        solr_collection_name = config['solr_collection_name']
        solr_utils.clean_solr(config, grids_to_use)
        print(f'Downloading {dataset_name} files to {target_dir}\n')

    # if target path doesn't exist, make them
    if not os.path.exists(folder):
        os.makedirs(folder)

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs = {}
    descendants_docs = {}

    # Query for existing harvested docs
    fq = ['type_s:granule', f'dataset_s:{dataset_name}']
    query_docs = solr_utils.solr_query(fq)

    if len(query_docs) > 0:
        for doc in query_docs:
            docs[doc['filename_s']] = doc

    # Query for existing descendants docs
    fq = ['type_s:descendants', f'dataset_s:{dataset_name}']
    existing_descendants_docs = solr_utils.solr_query(fq)

    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            key = doc['date_s']
            descendants_docs[key] = doc

    # =====================================================
    # ifremer loop
    # =====================================================
    try:
        ftp = FTP(host)
        if 'password' in config.keys():
            ftp.login(config['user'], config['password'])
        else:
            ftp.login(config['user'])
    except Exception as e:
        log.exception(f'Harvesting failed. Unable to connect to FTP. {e}')
        return 'Harvesting failed. Unable to connect to FTP.'

    try:
        files = ftp.nlst(ddir)
        files = [f.split('/')[-1] for f in files if valid_date(f.split('/')[-1], config)]
    except:
        log.exception(f'Error finding files at {ddir}. Check harvester config.')
        return 'Harvesting failed. Unable to find files on FTP.'

    for filename in files:
        if not any(ext in filename for ext in ['.nc', '.bz2', '.gz']):
            continue

        date = file_utils.get_date(config['regex'], filename)
        date_time = datetime.strptime(date, "%Y%m%d")
        new_date_format = f'{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z'

        granule_dates.append(datetime.strptime(
            new_date_format, config['date_regex']))

        url = f'{ddir}{filename}'

        # Granule metadata used for Solr harvested entries
        item = {}
        item['type_s'] = 'granule'
        item['date_s'] = new_date_format
        item['dataset_s'] = dataset_name
        item['filename_s'] = filename
        item['source_s'] = f'ftp://{host}/{url}'

        # Granule metadata used for initializing Solr descendants entries
        descendants_item = {}
        descendants_item['type_s'] = 'descendants'
        descendants_item['date_s'] = item["date_s"]
        descendants_item['dataset_s'] = item['dataset_s']
        descendants_item['filename_s'] = filename
        descendants_item['source_s'] = item['source_s']

        updating = False
        aws_upload = False

        # Attempt to get last modified time of file
        try:
            mod_time = ftp.voidcmd("MDTM "+url)[4:]
            mod_date_time = parser.parse(mod_time)
        except:
            mod_date_time = now

        mod_time = mod_date_time.strftime(time_format)
        item['modified_time_dt'] = mod_time

        updating = granule_update_check(
            docs, filename, mod_date_time, time_format)

        if updating:
            year = date[:4]
            local_fp = f'{folder}{dataset_name}_granule.nc' if on_aws else f'{target_dir}{year}/{filename}'

            if not os.path.exists(f'{target_dir}{year}/'):
                os.makedirs(f'{target_dir}{year}/')
            try:
                # If file doesn't exist locally, download it
                if not os.path.exists(local_fp):
                    print(f' - Downloading {filename} to {local_fp}')
                    with open(local_fp, 'wb') as f:
                        ftp.retrbinary(
                            'RETR '+url, f.write, blocksize=262144)

                # If file exists, but is out of date, download it
                elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                    print(
                        f' - Updating {filename} and downloading to {local_fp}')
                    with open(local_fp, 'wb') as f:
                        ftp.retrbinary(
                            'RETR '+url, f.write, blocksize=262144)
                else:
                    print(
                        f' - {filename} already downloaded and up to date')
                # Create checksum for file
                item['harvest_success_b'] = True
                item['file_size_l'] = os.path.getsize(local_fp)
                item['checksum_s'] = file_utils.md5(local_fp)
                item['pre_transformation_file_path_s'] = local_fp
                item['download_time_dt'] = chk_time

            except Exception as e:
                log.exception(e)
                print(f'Download of {filename} is unsuccessful. {e}')
                item['harvest_success_b'] = False
                item['filename'] = ''
                item['pre_transformation_file_path_s'] = ''
                item['file_size_l'] = 0

            # =====================================================
            # Push data to s3 bucket
            # =====================================================

            if on_aws:
                aws_upload = True
                print("=========uploading file to s3=========")
                target_bucket.upload_file(
                    local_fp, f'{dataset_name}/{filename}')
                item['pre_transformation_file_path_s'] = f's3://{config["target_bucket_name"]}/{dataset_name}/{filename}'
                print("======uploading file to s3 DONE=======")

            # Update descendant item
            key = descendants_item['date_s']

            if key in descendants_docs.keys():
                descendants_item['id'] = descendants_docs[key]['id']

            descendants_item['harvest_success_b'] = item['harvest_success_b']
            descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']

            entries_for_solr.append(item)
            entries_for_solr.append(descendants_item)

            last_success_item = item
        else:
            print(
                f' - {filename} already downloaded and up to date')

    print(f'\nDownloading {dataset_name} complete\n')

    ftp.quit()

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        r = solr_utils.solr_update(entries_for_solr, r=True)

        if r.status_code == 200:
            print('Successfully created or updated Solr harvested documents')
        else:
            print('Failed to create Solr harvested documents')

    # Query for Solr failed harvest documents
    fq = ['type_s:granule', f'dataset_s:{dataset_name}',
          f'harvest_success_b:false']
    failed_harvesting = solr_utils.solr_query(fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{dataset_name}',
          f'harvest_success_b:true']
    successful_harvesting = solr_utils.solr_query(fq)

    harvest_status = f'All granules successfully harvested'

    if not successful_harvesting:
        harvest_status = f'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'

    overall_start = min(granule_dates) if granule_dates else None
    overall_end = max(granule_dates) if granule_dates else None

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_query = solr_utils.solr_query(fq)

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
        ds_meta['short_name_s'] = config['original_dataset_short_name']
        ds_meta['source_s'] = f'ftp://{host}/{ddir}'
        ds_meta['data_time_scale_s'] = config['data_time_scale']
        ds_meta['date_format_s'] = config['date_format']
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
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
        r = solr_utils.solr_update([ds_meta], r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

        # If the dataset entry needs to be created, so do the field entries

        # -----------------------------------------------------
        # Create Solr dataset field entries
        # -----------------------------------------------------

        # Query for Solr field documents
        fq = ['type_s:field', f'dataset_s:{dataset_name}']
        field_query = solr_utils.solr_query(fq)

        body = []
        for field in config['fields']:
            field_obj = {}
            field_obj['type_s'] = {'set': 'field'}
            field_obj['dataset_s'] = {'set': dataset_name}
            field_obj['name_s'] = {'set': field['name']}
            field_obj['long_name_s'] = {'set': field['long_name']}
            field_obj['standard_name_s'] = {'set': field['standard_name']}
            field_obj['units_s'] = {'set': field['units']}

            for solr_field in field_query:
                if field['name'] == solr_field['name_s']:
                    field_obj['id'] = {'set': solr_field['id']}

            body.append(field_obj)

        # Update Solr with dataset fields metadata
        r = solr_utils.solr_update(body, r=True)

        if r.status_code == 200:
            print('Successfully created Solr field documents')
        else:
            print('Failed to create Solr field documents')

    # if dataset entry exists, update download time, converage start date, coverage end date
    else:
        # -----------------------------------------------------
        # Update Solr dataset entry
        # -----------------------------------------------------
        dataset_metadata = dataset_query[0]

        # Query for dates of all harvested docs
        getVars = {'q': '*:*',
                   'fq': [f'dataset_s:{dataset_name}', 'type_s:granule', 'harvest_success_b:true'],
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
        r = solr_utils.solr_update([update_doc], r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document\n')
        else:
            print('Failed to update Solr dataset document\n')
    return harvest_status
