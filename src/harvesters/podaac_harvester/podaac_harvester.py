import os
import sys
import gzip
import yaml
import shutil
import hashlib
import logging
import requests
import numpy as np
import xarray as xr
from pathlib import Path
from xml.etree.ElementTree import parse
from datetime import datetime, timedelta
from urllib.request import urlopen, urlcleanup, urlretrieve

log = logging.getLogger(__name__)


def clean_solr(config, solr_host, grids_to_use, solr_collection_name):
    """
    Remove harvested, transformed, and descendant entries in Solr for dates
    outside of config date range. Also remove related aggregations, and force
    aggregation rerun for those years.
    """
    dataset_name = config['ds_name']
    config_start = config['start']
    config_end = config['end']

    # Convert config dates to Solr format
    config_start = f'{config_start[:4]}-{config_start[4:6]}-{config_start[6:]}'
    config_end = f'{config_end[:4]}-{config_end[4:6]}-{config_end[6:]}'

    fq = [f'type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_metadata = solr_query(config, solr_host, fq, solr_collection_name)

    if not dataset_metadata:
        return
    else:
        dataset_metadata = dataset_metadata[0]

    # Remove entries earlier than config start date
    fq = [f'dataset_s:{dataset_name}', f'date_s:[* TO {config_start}}}']
    url = f'{solr_host}{solr_collection_name}/update?commit=true'
    requests.post(url, json={'delete': fq})

    # Remove entries later than config end date
    fq = [f'dataset_s:{dataset_name}', f'date_s:{{{config_end} TO *]']
    url = f'{solr_host}{solr_collection_name}/update?commit=true'
    requests.post(url, json={'delete': fq})

    # Add start and end years to 'years_updated' field in dataset entry
    # Forces the bounding years to be re-aggregated to account for potential
    # removed dates
    start_year = config_start[:4]
    end_year = config_end[:4]
    update_body = [{
        "id": dataset_metadata['id']
    }]

    for grid in grids_to_use:
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

    r = solr_update(config, solr_host, update_body,
                    solr_collection_name, r=True)


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
    # solr_collection_name = config['solr_collection_name']

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
    # solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def podaac_harvester(config_path='', output_path='', s3=None, on_aws=False, solr_info=''):
    """
    Pulls data files for PODAAC id and date range given in harvester_config.yaml.
    If not on_aws, saves locally, else saves to s3 bucket.
    Creates (or updates) Solr entries for dataset, harvested granule, fields,
    and descendants.
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
    aggregated = config['aggregated']
    start_time = config['start']
    end_time = config['end']
    host = config['host']
    podaac_id = config['podaac_id']
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

    # =====================================================
    # Setup AWS Target Bucket
    # =====================================================
    if on_aws:
        target_bucket_name = config['target_bucket_name']
        target_bucket = s3.Bucket(target_bucket_name)
        if solr_info:
            solr_host = solr_info['solr_url']
            solr_collection_name = solr_info['solr_collection_name']
        else:
            solr_host = config['solr_host_aws']
            solr_collection_name = config['solr_collection_name']
        print(f'Downloading {dataset_name} files and uploading to \
              {target_bucket_name}/{dataset_name}\n')
    else:
        target_bucket = None
        if solr_info:
            solr_host = solr_info['solr_url']
            solr_collection_name = solr_info['solr_collection_name']
        else:
            solr_host = config['solr_host_local']
            solr_collection_name = config['solr_collection_name']
        print(f'Downloading {dataset_name} files to {target_dir}\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================
    docs = {}
    descendants_docs = {}

    # Query for existing harvested docs
    fq = ['type_s:harvested', f'dataset_s:{dataset_name}']
    harvested_docs = solr_query(config, solr_host, fq, solr_collection_name)

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    if len(harvested_docs) > 0:
        for doc in harvested_docs:
            docs[doc['filename_s']] = doc

    # Query for existing descendants docs
    fq = ['type_s:descendants', f'dataset_s:{dataset_name}']
    existing_descendants_docs = solr_query(config, solr_host, fq, solr_collection_name)

    # Dictionary of existing descendants docs
    # descendant doc date : solr entry for that doc
    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            descendants_docs[doc['date_s']] = doc

    # =====================================================
    # Setup PODAAC loop variables
    # =====================================================
    url = f'{host}&datasetId={podaac_id}'
    if not aggregated:
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
                link = elem.find(
                    "{%(atom)s}link[@title='OPeNDAP URL']" % namespace).attrib['href']
                link = '.'.join(link.split('.')[:-1])
                newfile = link.split("/")[-1]

                # Skip granules of unrecognized file format
                if not any(extension in newfile for extension in ['.nc', '.bz2', '.gz']):
                    continue

                # Extract start and end dates from XML entry
                date_start_str = elem.find("{%(time)s}start" % namespace).text
                date_end_str = elem.find("{%(time)s}end" % namespace).text

                # Ignore granules with start time less than wanted start time
                # PODAAC can grab granule previous to start time if that granule's
                # end time is the same as the config file's start time
                if date_start_str.replace('-', '') < start_time and not aggregated:
                    continue

                # Remove nanoseconds from dates
                if len(date_start_str) > 19:
                    date_start_str = date_start_str[:19] + 'Z'
                if len(date_end_str) > 19:
                    date_end_str = date_end_str[:19] + 'Z'

                # Attempt to get last modified time of file on podaac
                # Not all PODAAC datasets contain last modified time
                try:
                    mod_time = elem.find("{%(atom)s}updated" % namespace).text
                    mod_date_time = datetime.strptime(mod_time, date_regex)

                except:
                    mod_time = str(now)
                    mod_date_time = now

                item = {}
                item['type_s'] = 'harvested'
                item['date_s'] = date_start_str
                item['dataset_s'] = dataset_name
                item['filename_s'] = newfile
                item['source_s'] = link
                item['modified_time_dt'] = mod_date_time.strftime(time_format)

                descendants_item = {}
                descendants_item['type_s'] = 'descendants'
                descendants_item['date_s'] = date_start_str
                descendants_item['dataset_s'] = dataset_name
                descendants_item['filename_s'] = newfile
                descendants_item['source_s'] = link

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
                        if aggregated:
                            print(
                                f'    - {newfile} is aggregated. Downloading may be slow.')

                        urlcleanup()
                        urlretrieve(link, local_fp)

                    # If file exists locally, but is out of date, download it
                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                        print(
                            f' - Updating {newfile} and downloading to {local_fp}')
                        urlcleanup()
                        urlretrieve(link, local_fp)

                    else:
                        print(
                            f' - {newfile} already downloaded and up to date')

                    if newfile in docs.keys():
                        item['id'] = docs[newfile]['id']

                    # Create checksum for file
                    item['checksum_s'] = md5(local_fp)
                    item['pre_transformation_file_path_s'] = local_fp

                    # =====================================================
                    # Push data to s3 bucket
                    # =====================================================

                    if on_aws:
                        aws_upload = True
                        output_filename = f'{dataset_name}/{newfile}' if on_aws else newfile
                        print("=========uploading file to s3=========")
                        try:
                            target_bucket.upload_file(
                                local_fp, output_filename)
                            item['pre_transformation_file_path_s'] = f's3://{config["target_bucket_name"]}/{output_filename}'
                        except:
                            print("======aws upload unsuccessful=======")
                            item['message_s'] = 'aws upload unsuccessful'
                        print("======uploading file to s3 DONE=======")

                    item['harvest_success_b'] = True
                    item['file_size_l'] = os.path.getsize(local_fp)

                    # =====================================================
                    # Handling data in aggregated form
                    # =====================================================
                    if aggregated:
                        # Aggregated file has already been downloaded
                        # Must extract individual granule slices
                        print(
                            f' - Extracting individual data granules from aggregated data file')

                        # Remove old outdated aggregated file from disk
                        for f in os.listdir(f'{target_dir}{year}/'):
                            if str(f) != str(newfile):
                                os.remove(f'{target_dir}{year}/{f}')

                        ds = xr.open_dataset(local_fp)

                        # List comprehension extracting times within desired date range
                        ds_times = [
                            time for time
                            in np.datetime_as_string(ds.time.values)
                            if start_time[:9] <= time.replace('-', '')[:9] <= end_time[:9]
                        ]

                        for time in ds_times:
                            year = time[:4]
                            new_ds = ds.sel(time=time)
                            file_name = f'{dataset_name}_{time.replace("-","")[:8]}.nc'
                            local_fp = f'{target_dir}{year}/{file_name}'
                            time_s = f'{time[:-10]}Z'

                            # granule metadata setup to be populated for each granule
                            item = {}
                            item['type_s'] = 'harvested'
                            item['date_s'] = time_s
                            item['dataset_s'] = dataset_name
                            item['filename_s'] = file_name
                            item['source_s'] = link
                            item['modified_time_dt'] = mod_date_time.strftime(
                                time_format)
                            item['download_time_dt'] = chk_time

                            # descendants metadta setup to be populated for each granule
                            descendants_item = {}
                            descendants_item['type_s'] = 'descendants'
                            descendants_item['dataset_s'] = item['dataset_s']
                            descendants_item['date_s'] = item["date_s"]
                            descendants_item['source_s'] = item['source_s']

                            if not os.path.exists(f'{target_dir}{year}'):
                                os.makedirs(f'{target_dir}{year}')

                            try:
                                # Save slice as NetCDF
                                new_ds.to_netcdf(path=local_fp)

                                # Create checksum for file
                                item['checksum_s'] = md5(local_fp)
                                item['pre_transformation_file_path_s'] = local_fp
                                item['harvest_success_b'] = True
                                item['file_size_l'] = os.path.getsize(local_fp)
                            except:
                                print(f'    - {file_name} failed to save')
                                item['harvest_success_b'] = False
                                item['pre_transformation_file_path_s'] = ''
                                item['file_size_l'] = 0
                                item['checksum_s'] = ''

                            if on_aws:
                                aws_upload = True
                                output_filename = f'{dataset_name}/{newfile}' if on_aws else newfile
                                print("=========uploading file to s3=========")
                                try:
                                    target_bucket.upload_file(
                                        local_fp, output_filename)
                                    item['pre_transformation_file_path_s'] = f's3://{config["target_bucket_name"]}/{output_filename}'
                                except:
                                    print("======aws upload unsuccessful=======")
                                    item['message_s'] = 'aws upload unsuccessful'
                                print("======uploading file to s3 DONE=======")

                            # Query for existing granule in Solr in order to update it
                            fq = ['type_s:harvested', f'dataset_s:{dataset_name}',
                                  f'date_s:{time_s[:10]}*']
                            granule = solr_query(config, solr_host, fq, solr_collection_name)

                            if granule:
                                item['id'] = granule[0]['id']

                            if time_s in descendants_docs.keys():
                                descendants_item['id'] = descendants_docs[time_s]['id']

                            entries_for_solr.append(item)
                            entries_for_solr.append(descendants_item)

                            start_times.append(datetime.strptime(
                                time[:-3], '%Y-%m-%dT%H:%M:%S.%f'))
                            end_times.append(datetime.strptime(
                                time[:-3], '%Y-%m-%dT%H:%M:%S.%f'))

                            if item['harvest_success_b']:
                                last_success_item = item

                else:
                    print(f' - {newfile} already downloaded and up to date')

            except Exception as e:
                print(f'    - {e}')
                if updating:

                    print(f'    - {newfile} failed to download')

                    item['harvest_success_b'] = False
                    item['pre_transformation_file_path_s'] = ''
                    item['file_size_l'] = 0

            if updating:
                item['download_time_dt'] = chk_time

                if date_start_str in descendants_docs.keys():
                    descendants_item['id'] = descendants_docs[date_start_str]['id']

                descendants_item['harvest_success_b'] = item['harvest_success_b']
                descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']

                if not aggregated:
                    entries_for_solr.append(item)
                    entries_for_solr.append(descendants_item)

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
        r = solr_update(config, solr_host, entries_for_solr, solr_collection_name, r=True)
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
    successful_harvesting = solr_query(config, solr_host, fq, solr_collection_name)

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
        ds_meta['short_name_s'] = config['original_dataset_short_name']
        ds_meta['source_s'] = f'{host}&datasetId={podaac_id}'
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
        r = solr_update(config, solr_host, [ds_meta], solr_collection_name, r=True)

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
        field_query = solr_query(config, solr_host, fq, solr_collection_name)

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
        r = solr_update(config, solr_host, body, solr_collection_name, r=True)

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

        # Check start and end date coverage
        old_start = datetime.strptime(dataset_metadata['start_date_dt'], time_format) \
            if 'start_date_dt' in dataset_metadata.keys() else None
        old_end = datetime.strptime(dataset_metadata['end_date_dt'], time_format) \
            if 'end_date_dt' in dataset_metadata.keys() else None

        # Build update document body
        update_doc = {}
        update_doc['id'] = dataset_metadata['id']
        update_doc['last_checked_dt'] = {"set": chk_time}

        if entries_for_solr:
            update_doc['harvest_status_s'] = {"set": harvest_status}

            if 'download_time_dt' in last_success_item.keys():
                update_doc['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}

            if old_start == None or overall_start < old_start:
                update_doc['start_date_dt'] = {
                    "set": overall_start.strftime(time_format)}

            if old_end == None or overall_end > old_end:
                update_doc['end_date_dt'] = {
                    "set": overall_end.strftime(time_format)}

        # Update Solr with modified dataset entry
        r = solr_update(config, solr_host, [update_doc], solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document\n')
        else:
            print('Failed to update Solr dataset document\n')
