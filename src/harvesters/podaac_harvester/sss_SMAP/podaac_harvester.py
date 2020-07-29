from xml.etree.ElementTree import parse
from datetime import datetime, timedelta
import sys
from os import path
import os
import re
from urllib.request import urlopen, urlcleanup, urlretrieve
import gzip
import shutil
import hashlib
import requests
import json
import yaml


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def getdate(regex, fname):
    ex = re.compile(regex)
    match = re.search(ex, fname)
    date = match.group()
    return date


def solr_query(config, fq):
    solr_host = config['solr_host']
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, update_body, r=False):
    solr_host = config['solr_host']
    solr_collection_name = config['solr_collection_name']

    url = solr_host + solr_collection_name + '/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def podaac_harvester(path_to_file_dir="", s3=None, on_aws=False):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    # TODO decide on command line argument for different dataset configs
    path_to_yaml = path_to_file_dir + "podaac_harvester_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    # =====================================================
    # Setup AWS Target Bucket
    # =====================================================
    if on_aws:
        target_bucket_name = config['target_bucket_name']
        target_bucket = s3.Bucket(target_bucket_name)

    # =====================================================
    # Download raw data files
    # =====================================================
    target_dir = config['target_dir'] + '/'
    folder = '/tmp/'+config['ds_name']+'/'

    if not on_aws:
        print("!!downloading files to "+target_dir)
    else:
        print("!!downloading files to "+folder+" and uploading to " +
              target_bucket_name+"/"+config['ds_name'])

    print("======downloading files========")
    if config['aggregated']:
        url = f'{config["host"]}&datasetId={config["podaac_id"]}'
    else:
        url = f'{config["host"]}&datasetId={config["podaac_id"]}&endTime={config["end"]}&startTime={config["start"]}'
    print(url)

    namespace = {"podaac": "http://podaac.jpl.nasa.gov/opensearch/",
                 "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                 "atom": "http://www.w3.org/2005/Atom",
                 "georss": "http://www.georss.org/georss",
                 "gml": "http://www.opengis.net/gml",
                 "dc": "http://purl.org/dc/terms/",
                 "time": "http://a9.com/-/opensearch/extensions/time/1.0/"}

    next = None
    firstIter = True
    more = True

    # ------------------------------------------------------------

    # if target path doesn't exist, make it
    # if tmp folder for downloaded files doesn't exist, create it in temp lambda storage
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # get old data if exists
    docs = {}

    fq = ['type_s:harvested', f'dataset_s:{config["ds_name"]}']
    query_docs = solr_query(config, fq)

    if len(query_docs) > 0:
        for doc in query_docs:
            docs[doc['filename_s']] = doc

    # setup metadata
    meta = []
    item = {}
    last_success_item = {}
    start = []
    end = []
    years_updated = set()
    chk_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    dl_count = 0    # track how many new files to transform and wait for
    now = datetime.utcnow()
    updating = False
    aws_upload = False

    if not config['aggregated']:
        config_start_datetime = datetime.strptime(
            config['start'], "%Y%m%dT%H:%M:%SZ")
        config_end_datetime = datetime.strptime(
            config['end'], "%Y%m%dT%H:%M:%SZ")

    while more:
        xml = parse(urlopen(url))
        if firstIter:
            totalResults = int(
                xml.find('{%(opensearch)s}totalResults' % namespace).text)
            firstIter = False
        items = xml.findall('{%(atom)s}entry' % namespace)

        for elem in items:
            updating = False
            aws_upload = False

            try:
                # download link
                link = elem.find(
                    "{%(atom)s}link[@title='OPeNDAP URL']" % namespace).attrib['href']
                link = '.'.join(link.split('.')[:-1])
                newfile = link.split("/")[-1]

                # dates
                start_str = elem.find("{%(time)s}start" % namespace).text
                # end_str = elem.find("{%(time)s}end" % namespace).text
                # date_range = '{start}-{end}'.format(
                #     start=start_str, end=end_str)

                # start_datetime = datetime.strptime(
                #     start_str, config['date_regex'])
                # end_datetime = datetime.strptime(end_str, config['date_regex'])

                # start.append(start_datetime)
                # end.append(end_datetime)

                # metadata setup
                item = {}               # to be populated for each file
                item['type_s'] = 'harvested'
                item['date_s'] = start_str
                item['dataset_s'] = config['ds_name']
                item['source_s'] = link

                try:
                    # get last modified time of file on podaac
                    mod_time = elem.find("{%(atom)s}updated" % namespace).text
                    mod_date_time = datetime.strptime(
                        mod_time, config['date_regex'])
                    item['modified_time_dt'] = mod_time

                except:
                    print('Cannot find last modified time.  Downloading granule.')
                    mod_date_time = now

                # compare modified timestamp or if granule previously downloaded
                updating = (not newfile in docs.keys()) or (not docs[newfile]['harvest_success_b']) \
                    or (datetime.strptime(docs[newfile]['download_time_dt'], "%Y-%m-%dT%H:%M:%SZ") <= mod_date_time)

                # if no granule metadata or download time less than modified time, download new file
                if updating:
                    local_fp = f'{folder}{config["ds_name"]}_granule.nc' if on_aws else target_dir + newfile

                    if not os.path.exists(local_fp):
                        print('Downloading: ' + local_fp)

                        urlcleanup()
                        urlretrieve(link, local_fp)

                        # unzip .gz files
                        if newfile[-3:] == '.gz':
                            with gzip.open(local_fp, "rb") as f_in, open(local_fp[:-3], "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                os.remove(local_fp)
                            newfile_ext = os.path.splitext(
                                os.listdir(folder)[0])[1]
                            local_fp = local_fp[:-3]+newfile_ext

                    elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= mod_date_time:
                        print('Updating: ' + local_fp)

                        urlcleanup()
                        urlretrieve(link, local_fp)

                        # unzip .gz files
                        if newfile[-3:] == '.gz':
                            with gzip.open(local_fp, "rb") as f_in, open(local_fp[:-3], "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                os.remove(local_fp)
                            newfile_ext = os.path.splitext(
                                os.listdir(folder)[0])[1]
                            local_fp = local_fp[:-3]+newfile_ext

                    else:
                        print('File already downloaded and up to date')
                        
                    item['checksum_s'] = md5(local_fp)

                    # =====================================================
                    # ### Push data to s3 bucket
                    # =====================================================
                    output_filename = config['ds_name'] + \
                        '/' + newfile if on_aws else newfile
                    item['pre_transformation_file_path_s'] = target_dir + newfile

                    if on_aws:
                        aws_upload = True
                        print("=========uploading file=========")
                        # print('uploading '+output_filename)
                        target_bucket.upload_file(local_fp, output_filename)
                        item['pre_transformation_file_path_s'] = 's3://' + \
                            config['target_bucket_name']+'/'+output_filename
                        print("======uploading file DONE=======")

                    dl_count += 1
                    item['harvest_success_b'] = True
                    item['filename_s'] = newfile
                    item['file_size_l'] = os.path.getsize(local_fp)

                    years_updated.add(start_str[:4])

            except:
                if updating:
                    if aws_upload:
                        print("======aws upload unsuccessful=======")
                        item['message_s'] = 'aws upload unsuccessful'

                    else:
                        print("Download "+newfile+" failed.")
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

        next = xml.find("{%(atom)s}link[@rel='next']" % namespace)
        if next is None:
            more = False
            print(config['ds_name']+' done')
        else:
            url = next.attrib['href']

    # =====================================================
    # ### writing metadata to file
    # =====================================================
    print("=========creating meta=========")

    meta_path = config['ds_name']+'.json'
    meta_local_path = target_dir+meta_path
    meta_output_path = 'meta/'+meta_path

    if len(meta) == 0:
        print('no new downloads')

    # write json file
    with open(meta_local_path, 'w') as meta_file:
        json.dump(meta, meta_file)

    print("======creating meta DONE=======")

    if on_aws:
        # =====================================================
        # ### uploading metadata file to s3
        # =====================================================
        print("=========uploading meta=========")

        target_bucket.upload_file(meta_local_path, meta_output_path)

        print("======uploading meta DONE=======")

    # =====================================================
    # ### posting metadata logs to Solr
    # =====================================================
    print("=========posting meta=========")

    headers = {'Content-Type': 'application/json'}
    overall_start = min(start) if len(start) > 0 else None
    overall_end = max(end) if len(end) > 0 else None
    # =====================================================
    # Query for Solr Dataset-level Document
    # =====================================================

    fq = ['type_s:dataset', 'dataset_s:'+config['ds_name']]
    docs = solr_query(config, fq)

    update = (len(docs) == 1)

    # if no dataset-level entry in Solr, create one
    if not update:
        # TODO: THIS SECTION BELONGS WITH DATASET DISCOVERY
        # -----------------------------------------------------
        # Create Solr Dataset-level Document if doesn't exist
        # -----------------------------------------------------
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = config['ds_name']
        ds_meta['short_name_s'] = config['short_name']
        ds_meta['source_s'] = f'{config["host"]}&datasetId={config["podaac_id"]}'
        ds_meta['data_time_scale_s'] = config['data_time_scale']
        ds_meta['date_format_s'] = config['date_format']
        ds_meta['last_checked_dt'] = chk_time
        ds_meta['years_updated_ss'] = list(years_updated)
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

        # Post document
        r = solr_update(config, body, r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

        # TODO: update for changes in yaml (incinerate and rewrite)
        # modify updating variable to account for updates in config and then incinerate and rewrite
        body = []
        for field in config['fields']:
            field_obj = {}
            field_obj['type_s'] = 'field'
            field_obj['dataset_s'] = config['ds_name']
            field_obj['name_s'] = field['name']
            field_obj['long_name_s'] = field['long_name']
            field_obj['standard_name_s'] = field['standard_name']
            field_obj['units_s'] = field['units']
            body.append(field_obj)

        # post document
        r = solr_update(config, body, r=True)

        if r.status_code == 200:
            print('Successfully created Solr field documents')
        else:
            print('Failed to create Solr field documents')

    # if record exists, update download time, converage start date, coverage end date
    else:
        # =====================================================
        # Check start and end date coverage
        # =====================================================
        doc = docs[0]
        old_start = datetime.strptime(
            doc['start_date_dt'], "%Y-%m-%dT%H:%M:%SZ") if 'start_date_dt' in doc.keys() else None
        old_end = datetime.strptime(
            doc['end_date_dt'], "%Y-%m-%dT%H:%M:%SZ") if 'end_date_dt' in doc.keys() else None
        doc_id = doc['id']

        # build update document body
        update_doc = {}
        update_doc['id'] = doc_id
        update_doc['last_checked_dt'] = {"set": chk_time}
        update_doc['status_s'] = {"set": "harvested"}
        if years_updated:
            update_doc['years_updated_ss'] = {"set": list(years_updated)}

        if updating:
            # only update to "harvested" if there is further preprocessing to do
            update_doc['status_s'] = "harvested"

            if len(meta) > 0 and 'download_time_dt' in last_success_item.keys():
                update_doc['last_download_dt'] = {
                    "set": last_success_item['download_time_dt']}
            if old_start == None or overall_start < old_start:
                update_doc['start_date_dt'] = {
                    "set": overall_start.strftime("%Y-%m-%dT%H:%M:%SZ")}
            if old_end == None or overall_end > old_end:
                update_doc['end_date_dt'] = {
                    "set": overall_end.strftime("%Y-%m-%dT%H:%M:%SZ")}

        body = [update_doc]
        r = solr_update(config, body, r=True)

        if r.status_code == 200:
            print('Successfully updated Solr dataset document')
        else:
            print('Failed to update Solr dataset document')

    # post granule metadata documents for downloaded granules
    r = solr_update(config, meta, r=True)

    if r.status_code == 200:
        print('granule metadata post to Solr success')
    else:
        print('granule metadata post to Solr failed')
    print("=========posted meta==========")
