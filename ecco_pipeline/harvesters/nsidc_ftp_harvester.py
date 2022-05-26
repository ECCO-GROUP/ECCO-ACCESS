import hashlib
import logging
import logging.config
import os
import re
from datetime import datetime
from ftplib import FTP
from pathlib import Path

import numpy as np
import requests
from dateutil import parser
from utils import file_utils, solr_utils

logging.config.fileConfig('logs/log.ini', disable_existing_loggers=False)
log = logging.getLogger(__name__)


def getdate(regex, fname):
    """
    Extracts date from file name using regex
    """
    ex = re.compile(regex)
    match = re.search(ex, fname)
    date = match.group()
    return date


def harvester(config, output_path, grids_to_use=[]):
    """
    Pulls data files for NSIDC FTP id and date range given in harvester_config.yaml.
    Creates (or updates) Solr entries for dataset, harvested granule, fields,
    and descendants.
    """

    # =====================================================
    # Read harvester_config.yaml and setup variables
    # =====================================================
    dataset_name = config['ds_name']
    start_time = config['start']
    end_time = config['end']
    data_time_scale = config['data_time_scale']
    host = config['host']
    ddir = config['ddir']

    if end_time == 'NOW':
        end_time = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")

    target_dir = f'{output_path}/{dataset_name}/harvested_granules/'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    time_format = "%Y-%m-%dT%H:%M:%SZ"
    entries_for_solr = []
    last_success_item = {}
    granule_dates = []
    chk_time = datetime.utcnow().strftime(time_format)
    now = datetime.utcnow()
    updating = False

    solr_utils.clean_solr(config, grids_to_use)
    print(f'Downloading {dataset_name} files to {target_dir}\n')

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

    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    query_docs = solr_utils.solr_query(fq)

    # Query for existing descendants docs
    fq = ['type_s:descendants', f'dataset_s:{dataset_name}']
    existing_descendants_docs = solr_utils.solr_query(fq)

    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            if doc['hemisphere_s']:
                key = (doc['date_s'], doc['hemisphere_s'])
            else:
                key = doc['date_s']
            descendants_docs[key] = doc

    # =====================================================
    # Setup NSIDC loop variables
    # =====================================================
    try:
        ftp = FTP(host)
        ftp.login(config['user'])
    except Exception as e:
        log.exception(f'Harvesting failed. Unable to connect to FTP. {e}')
        return 'Harvesting failed. Unable to connect to FTP.'

    start_time_dt = datetime.strptime(start_time, "%Y%m%dT%H:%M:%SZ")
    end_time_dt = datetime.strptime(end_time, "%Y%m%dT%H:%M:%SZ")

    start_year = start_time[:4]
    end_year = end_time[:4]
    years = np.arange(int(start_year), int(end_year) + 1)

    # =====================================================
    # NSIDC loop
    # =====================================================
    # Iterate through hemispheres given in config
    for region in config['regions']:
        hemi = 'nh' if region == 'north' else 'sh'

        for year in years:

            ftp_dir = f'{ddir}{region}/{data_time_scale}/{year}/'

            try:
                files = []
                ftp.dir(ftp_dir, files.append)

                # Ignore first two FTP entries as they aren't related to data
                files = files[2:]
                # Extract file names from FTP directory
                files = [e.split()[-1] for e in files]

                if not files:
                    print(f'No granules found for region {region} in {year}.')
            except:
                log.exception(
                    f'Error finding files at {ftp_dir}. Check harvester config.')
                print(
                    f'Error finding files at {ftp_dir}. Check harvester config.')

            for newfile in files:
                try:
                    if not any(extension in newfile for extension in ['.nc', '.bz2', '.gz']):
                        continue

                    url = f'{ftp_dir}{newfile}'

                    # Extract the date from the filename
                    date = getdate(config['regex'], newfile)
                    date_time = datetime.strptime(date, "%Y%m%d")
                    new_date_format = f'{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z'

                    # Ignore granules with start time less than wanted start time
                    # List of files contains ALL files on FTP dir from that year
                    if (start_time_dt > date_time) or (end_time_dt < date_time):
                        continue

                    granule_dates.append(datetime.strptime(
                        new_date_format, config['date_regex']))

                    # Granule metadata used for Solr harvested entries
                    item = {}
                    item['type_s'] = 'granule'
                    item['date_s'] = new_date_format
                    item['dataset_s'] = dataset_name
                    item['hemisphere_s'] = hemi
                    item['filename_s'] = newfile
                    item['source_s'] = f'ftp://{host}/{url}'

                    # Granule metadata used for initializing Solr descendants entries
                    descendants_item = {}
                    descendants_item['type_s'] = 'descendants'
                    descendants_item['date_s'] = item["date_s"]
                    descendants_item['dataset_s'] = item['dataset_s']
                    descendants_item['hemisphere_s'] = hemi
                    descendants_item['filename_s'] = newfile
                    descendants_item['source_s'] = item['source_s']

                    updating = False

                    # Attempt to get last modified time of file
                    try:
                        mod_time = ftp.voidcmd("MDTM "+url)[4:]
                        mod_date_time = parser.parse(mod_time)
                        mod_time = mod_date_time.strftime(time_format)
                        item['modified_time_dt'] = mod_time
                    except:
                        mod_date_time = now

                    # If granule doesn't exist or previously failed or has been updated since last harvest
                    updating = (not newfile in docs.keys()) or \
                        (not docs[newfile]['harvest_success_b']) or \
                        (datetime.strptime(
                            docs[newfile]['download_time_dt'], time_format) <= mod_date_time)

                    # If updating, download file if necessary
                    if updating:
                        year = date[:4]
                        local_fp = f'{target_dir}{year}/{newfile}'

                        if not os.path.exists(f'{target_dir}{year}/'):
                            os.makedirs(f'{target_dir}{year}/')

                        # If file doesn't exist locally, download it
                        if not os.path.exists(local_fp):
                            print(f' - Downloading {newfile} to {local_fp}')
                            try:
                                with open(local_fp, 'wb') as f:
                                    ftp.retrbinary('RETR '+url, f.write)
                            except:
                                os.unlink(local_fp)

                        # If file exists, but is out of date, download it
                        elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                            print(
                                f' - Updating {newfile} and downloading to {local_fp}')
                            try:
                                with open(local_fp, 'wb') as f:
                                    ftp.retrbinary('RETR '+url, f.write)
                            except:
                                os.unlink(local_fp)

                        else:
                            print(
                                f' - {newfile} already downloaded and up to date')

                        if newfile in docs.keys():
                            item['id'] = docs[newfile]['id']

                        # Create checksum for file
                        item['checksum_s'] = file_utils.md5(local_fp)
                        item['pre_transformation_file_path_s'] = local_fp
                        item['harvest_success_b'] = True
                        item['file_size_l'] = os.path.getsize(local_fp)

                    else:
                        print(
                            f' - {newfile} already downloaded and up to date')

                except Exception as e:
                    log.exception(e)
                    print(f'    - {e}')
                    if updating:
                        print(f'    - {newfile} failed to download')

                        item['harvest_success_b'] = False
                        item['pre_transformation_file_path_s'] = ''
                        item['file_size_l'] = 0

                if updating:
                    item['download_time_dt'] = chk_time

                    # Update Solr entry using id if it exists
                    if hemi:
                        key = (descendants_item['date_s'], hemi)
                    else:
                        key = descendants_item['date_s']

                    if key in descendants_docs.keys():
                        descendants_item['id'] = descendants_docs[key]['id']

                    descendants_item['harvest_success_b'] = item['harvest_success_b']
                    descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']

                    entries_for_solr.append(item)
                    entries_for_solr.append(descendants_item)

                    if item['harvest_success_b']:
                        last_success_item = item

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
    fq = ['type_s:granule',
          f'dataset_s:{dataset_name}', f'harvest_success_b:false']
    failed_harvesting = solr_utils.solr_query(fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule',
          f'dataset_s:{dataset_name}', f'harvest_success_b:true']
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
        ds_meta['data_time_scale_s'] = data_time_scale
        ds_meta['date_format_s'] = config['date_format']
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
        ds_meta['original_dataset_url_s'] = config['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']

        # Only include start_date and end_date if there was at least one successful download
        if overall_start != None:
            ds_meta['start_date_dt'] = overall_start.strftime(
                time_format)
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
        fq = [f'dataset_s:{dataset_name}',
              'type_s:granule', 'harvest_success_b:true']
        dates_query = solr_utils.solr_query(fq, fl='date_s')
        dates = [x['date_s'] for x in dates_query]

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
