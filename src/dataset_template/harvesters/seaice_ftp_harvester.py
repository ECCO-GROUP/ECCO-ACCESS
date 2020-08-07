import os
import re
import sys
import gzip
import json
import yaml
import shutil
import hashlib
import datetime
import requests
import numpy as np
from ftplib import FTP
from dateutil import parser
from xml.etree.ElementTree import parse
from urllib.request import urlopen, urlcleanup, urlretrieve


# Creates checksum from filename
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Extracts date from file name following regex
def getdate(regex, fname):
    ex = re.compile(regex)
    match = re.search(ex, fname)
    date = match.group()
    return date


# Queries Solr based on config information and filter query
# Returns list of Solr entries (docs)
def solr_query(config, solr_host, fq):
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


# Posts update to Solr with provided update body
# Optional return of posting status code
def solr_update(config, solr_host, update_body, r=False):
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


# Pulls data files for given ftp source and date range
# If not on_aws, saves locally, else saves to s3 bucket
# Creates Solr entries for dataset, harvested granule, fields, and lineage
def seaice_ftp_harvester(path_to_file_dir="", s3=None, on_aws=False):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    path_to_yaml = path_to_file_dir + "seaice_harvester_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    # =====================================================
    # Setup AWS Target Bucket
    # =====================================================
    if on_aws:
        target_bucket_name = config['target_bucket_name']
        target_bucket = s3.Bucket(target_bucket_name)
        solr_host = config['solr_host_aws']
    else:
        solr_host = config['solr_host_local']

    # =====================================================
    # Initializing required values
    # =====================================================
    target_dir = config['target_dir'] + '/'
    folder = '/tmp/'+config['ds_name']+'/'
    data_time_scale = config['data_time_scale']
    dataset_name = config['ds_name']

    ftp = FTP(config['host'])
    ftp.login(config['user'])

    if not on_aws:
        print(f'!!downloading files to {target_dir}')
    else:
        print(
            f'!!downloading files to {folder} and uploading to {target_bucket_name}/{dataset_name}')

    # if target path doesn't exist, make them
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    docs = {}
    lineage_docs = {}

    # Query for existing harvested docs
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    query_docs = solr_query(config, solr_host, fq)

    if len(query_docs) > 0:
        for doc in query_docs:
            docs[doc['filename_s']] = doc

    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    query_docs = solr_query(config, solr_host, fq)

    # Query for existing lineage docs
    fq = ['type_s:lineage', f'dataset_s:{dataset_name}']
    existing_lineage_docs = solr_query(config, solr_host, fq)

    if len(existing_lineage_docs) > 0:
        for doc in existing_lineage_docs:
            lineage_docs[doc['date_s']] = doc

    # setup metadata
    meta = []
    item = {}
    last_success_item = {}
    start = []
    end = []
    chk_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.datetime.utcnow()
    updating = False
    aws_upload = False

    start_time = datetime.datetime.strptime(
        config['start'], "%Y%m%dT%H:%M:%SZ")
    end_time = datetime.datetime.strptime(config['end'], "%Y%m%dT%H:%M:%SZ")
    start_year = config['start'][:4]
    end_year = config['end'][:4]
    years = np.arange(int(start_year), int(end_year) + 1)

    # Iterate through years from start and end dates given in config
    for year in years:

        # Iterate through hemispheres given in config
        for region in config['regions']:

            hemi = 'nh' if region == 'north' else 'sh'

            # build source urlbase
            urlbase = f'{config["ddir"]}{region}/{data_time_scale}/{year}/'
            files = []

            # Retrieve list of files from urlbase
            try:
                ftp.dir(urlbase, files.append)
                files = files[2:]
                files = [e.split()[-1] for e in files]
            except:
                print(f'Error finding files at {urlbase}')

            for newfile in files:
                try:

                    url = f'{urlbase}{newfile}'

                    date = getdate(config['regex'], newfile)
                    date_time = datetime.datetime.strptime(date, "%Y%m%d")
                    new_date_format = f'{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z'

                    # Ignore granules with start time less than wanted start time
                    if (start_time > date_time) and (end_time < date_time):
                        continue

                    # granule metadata setup to be populated for each granule
                    item = {}
                    item['type_s'] = 'harvested'
                    item['date_s'] = new_date_format
                    item['dataset_s'] = config['ds_name']
                    item['hemisphere_s'] = hemi
                    item['source_s'] = f'ftp://{config["host"]}/{url}'

                    updating = False
                    aws_upload = False

                    # Attempt to get last modified time of file
                    try:
                        mod_time = ftp.voidcmd("MDTM "+url)[4:]
                        mod_date_time = parser.parse(mod_time)
                        mod_time = mod_date_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        item['modified_time_dt'] = mod_time
                    except:
                        print('Cannot find last modified time. Downloading granule.')
                        mod_date_time = now

                    # If granule doesn't exist or previously failed or has been updated since last harvest
                    updating = (not newfile in docs.keys()) or (not docs[newfile]['harvest_success_b']) \
                        or (datetime.datetime.strptime(docs[newfile]['download_time_dt'], "%Y-%m-%dT%H:%M:%SZ") <= mod_date_time)

                    # If updating, download file
                    if updating:
                        local_fp = f'{folder}{config["ds_name"]}_granule.nc' if on_aws else target_dir + newfile

                        # If file doesn't exist locally, download it
                        if not os.path.exists(local_fp):
                            print(f'Downloading: {local_fp}')

                            # new ftp retrieval
                            with open(local_fp, 'wb') as f:
                                ftp.retrbinary('RETR '+url, f.write)

                        # If file exists, but is out of date, download it
                        elif datetime.datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                            print(f'Updating: {local_fp}')

                            # new ftp retrieval
                            with open(local_fp, 'wb') as f:
                                ftp.retrbinary('RETR '+url, f.write)

                        else:
                            print('File already downloaded and up to date')

                        # Create checksum for file
                        item['checksum_s'] = md5(local_fp)

                        output_filename = f'{dataset_name}/{newfile}' if on_aws else newfile

                        item['pre_transformation_file_path_s'] = f'{target_dir}{newfile}'

                        # =====================================================
                        # Push data to s3 bucket
                        # =====================================================

                        if on_aws:
                            aws_upload = True
                            print("=========uploading file to s3=========")
                            target_bucket.upload_file(
                                local_fp, output_filename)
                            item['pre_transformation_file_path_s'] = f's3://{config["target_bucket_name"]}/{output_filename}'
                            print("======uploading file to s3 DONE=======")

                        item['harvest_success_b'] = True
                        item['filename_s'] = newfile
                        item['file_size_l'] = os.path.getsize(local_fp)

                except Exception as e:
                    print(e)
                    if updating:
                        if aws_upload:
                            print("======aws upload unsuccessful=======")
                            item['message_s'] = 'aws upload unsuccessful'

                        else:
                            print(f'Download {newfile} failed.')
                            print("======file not successful=======")

                        item['harvest_success_b'] = False
                        item['filename'] = ''
                        item['pre_transformation_file_path_s'] = ''
                        item['file_size_l'] = 0

                if updating:
                    item['download_time_dt'] = chk_time

                    # add item to metadata json
                    meta.append(item)
                    # store meta for last successful download
                    last_success_item = item

    ftp.quit()

    # post granule metadata documents for downloaded granules
    r = solr_update(config, solr_host, meta, r=True)

    if meta:
        if r.status_code == 200:
            print('granule metadata post to Solr success')
        else:
            print('granule metadata post to Solr failed')
    else:
        print('no granules found')

    # =====================================================
    # ### writing metadata to file
    # =====================================================
    print("=========creating metadata JSON=========")

    meta_path = f'{dataset_name}.json'
    meta_local_path = f'{target_dir}{meta_path}'
    meta_output_path = f'meta/{meta_path}'

    if len(meta) == 0:
        print('no new downloads')

    # write json file
    with open(meta_local_path, 'w') as meta_file:
        json.dump(meta, meta_file)

    print("======creating meta JSON DONE=======")

    # =====================================================
    # uploading metadata file to s3
    # =====================================================
    if on_aws:
        print("=========uploading meta to s3=========")
        target_bucket.upload_file(meta_local_path, meta_output_path)
        print("======uploading meta to s3 DONE=======")

    overall_start = min(start) if len(start) > 0 else None
    overall_end = max(end) if len(end) > 0 else None

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    docs = solr_query(config, solr_host, fq)

    # If dataset entry exists on Solr
    update = (len(docs) == 1)

    # Update Solr metadata for dataset and fields
    if not update:
        # TODO: THIS SECTION BELONGS WITH DATASET DISCOVERY

        # -----------------------------------------------------
        # Create Solr dataset entry
        # -----------------------------------------------------
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = dataset_name
        ds_meta['short_name_s'] = config['short_name']
        ds_meta['source_s'] = f'ftp://{config["host"]}/{config["ddir"]}'
        ds_meta['data_time_scale_s'] = config['data_time_scale']
        ds_meta['date_format_s'] = config['date_format']
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['original_dataset_title_s'] = config['original_dataset_title']
        ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
        ds_meta['original_dataset_url_s'] = config['original_dataset_url']
        ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
        ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']

        if overall_start != None:
            ds_meta['start_date_dt'] = overall_start.strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            ds_meta['end_date_dt'] = overall_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ds_meta['status_s'] = 'error harvesting - no files found'

        # if no ds entry yet and no qualifying downloads, still create ds entry without download time
        if updating:
            ds_meta['last_download_dt'] = last_success_item['download_time_dt']
            ds_meta['status_s'] = "harvested"
        else:
            ds_meta['status_s'] = "nodata"

        body = []
        body.append(ds_meta)

        # Update Solr with dataset metadata
        r = solr_update(config, solr_host, body, r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

        # -----------------------------------------------------
        # Create Solr dataset field entries
        # -----------------------------------------------------
        body = []
        for field in config['fields']:
            field_obj = {}
            field_obj['type_s'] = 'field'
            field_obj['dataset_s'] = dataset_name
            field_obj['name_s'] = field['name']
            field_obj['long_name_s'] = field['long_name']
            field_obj['standard_name_s'] = field['standard_name']
            field_obj['units_s'] = field['units']
            body.append(field_obj)

        # Update Solr with dataset fields metadata
        r = solr_update(config, solr_host, body, r=True)

        if r.status_code == 200:
            print('Successfully created Solr field documents')
        else:
            print('Failed to create Solr field documents')

    # if dataset entry exists, update download time, converage start date, coverage end date
    else:
        # Check start and end date coverage
        doc = docs[0]
        old_start = datetime.datetime.strptime(
            doc['start_date_dt'], "%Y-%m-%dT%H:%M:%SZ") if 'start_date_dt' in doc.keys() else None
        old_end = datetime.datetime.strptime(
            doc['end_date_dt'], "%Y-%m-%dT%H:%M:%SZ") if 'end_date_dt' in doc.keys() else None
        doc_id = doc['id']

        # build update document body
        update_doc = {}
        update_doc['id'] = doc_id
        update_doc['last_checked_dt'] = {"set": chk_time}

        if meta:
            update_doc['status_s'] = {"set": "harvested"}

            if 'download_time_dt' in last_success_item.keys():
                update_doc['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}

            if old_start == None or overall_start < old_start:
                update_doc['start_date_dt'] = {
                    "set": overall_start.strftime("%Y-%m-%dT%H:%M:%SZ")}

            if old_end == None or overall_end > old_end:
                update_doc['end_date_dt'] = {
                    "set": overall_end.strftime("%Y-%m-%dT%H:%M:%SZ")}

        # Update Solr with modified dataset entry
        r = solr_update(config, solr_host, [update_doc], r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document')
        else:
            print('Failed to update Solr dataset document')
