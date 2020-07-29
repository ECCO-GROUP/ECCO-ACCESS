from __future__ import print_function
from xml.etree.ElementTree import parse
import datetime
# from datetime import datetime, timedelta
import sys
from os import path
import os
import re
# from urllib.request import urlopen, urlcleanup, urlretrieve
import gzip
import shutil
import hashlib
import requests
import json
import yaml
from ftplib import FTP
from dateutil import parser
import numpy as np

import base64
import itertools
import netrc
import ssl
try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor


CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    errprefix = ''
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        errprefix = 'netrc error: '
    except Exception as e:
        if (not ('No such file' in str(e))):
            print('netrc error: {0}'.format(str(e)))
        username = None
        password = None

    while not credentials:
        if not username:
            username = 'ecco_access' #hardcoded username
            password = 'ECCOAccess1' #hardcoded password
        credentials = '{0}:{1}'.format(username, password)
        credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')

        if url:
            try:
                req = Request(url)
                req.add_header('Authorization', 'Basic {0}'.format(credentials))
                opener = build_opener(HTTPCookieProcessor())
                opener.open(req)
            except HTTPError:
                print(errprefix + 'Incorrect username or password')
                errprefix = ''
                credentials = None
                username = None
                password = None

    return credentials

def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ''

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += '&version={0}'.format(padded_version)
        desired_pad_length -= 1
    return query_params


def build_cmr_query_url(short_name, version, time_start, time_end,
                        bounding_box=None, polygon=None,
                        filename_filter=None):
    params = '&short_name={0}'.format(short_name)
    params += build_version_query_params(version)
    params += '&temporal[]={0},{1}'.format(time_start, time_end)
    if polygon:
        params += '&polygon={0}'.format(polygon)
    elif bounding_box:
        params += '&bounding_box={0}'.format(bounding_box)
    if filename_filter:
        option = '&options[producer_granule_id][pattern]=true'
        params += '&producer_granule_id[]={0}{1}'.format(filename_filter, option)
    return CMR_FILE_URL + params


def cmr_download(urls):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    print('Downloading {0} files...'.format(url_count))
    credentials = None

    for index, url in enumerate(urls, start=1):
        if not credentials and urlparse(url).scheme == 'https':
            credentials = get_credentials(url)

        filename = url.split('/')[-1]
        print('{0}/{1}: {2}'.format(str(index).zfill(len(str(url_count))),
                                    url_count,
                                    filename))

        try:
            # In Python 3 we could eliminate the opener and just do 2 lines:
            # resp = requests.get(url, auth=(username, password))
            # open(filename, 'wb').write(resp.content)
            req = Request(url)
            if credentials:
                req.add_header('Authorization', 'Basic {0}'.format(credentials))
            opener = build_opener(HTTPCookieProcessor())
            data = opener.open(req).read()
            open(filename, 'wb').write(data)
        except HTTPError as e:
            print('HTTP error {0}, {1}'.format(e.code, e.reason))
        except URLError as e:
            print('URL error: {0}'.format(e.reason))
        except IOError:
            raise
        except KeyboardInterrupt:
            quit()


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = [e['links']
               for e in search_results['feed']['entry']
               if 'links' in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if 'href' not in link:
            # Exclude links with nothing to download
            continue
        if 'inherited' in link and link['inherited'] is True:
            # Why are we excluding these links?
            continue
        if 'rel' in link and 'data#' not in link['rel']:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if 'title' in link and 'opendap' in link['title'].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link['href'].split('/')[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link['href'])

    return urls


def cmr_search(short_name, version, time_start, time_end,
               bounding_box='', polygon='', filename_filter=''):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(short_name=short_name, version=version,
                                        time_start=time_start, time_end=time_end,
                                        bounding_box=bounding_box,
                                        polygon=polygon, filename_filter=filename_filter)
    print('Querying for data:\n\t{0}\n'.format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header('cmr-scroll-id', cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(response.info()).items()}
                cmr_scroll_id = headers['cmr-scroll-id']
                hits = int(headers['cmr-hits'])
                if hits > 0:
                    print('Found {0} matches.'.format(hits))
                else:
                    print('Found no matches.')
            search_page = response.read()
            search_page = json.loads(search_page.decode('utf-8'))
            url_scroll_results = cmr_filter_urls(search_page)
            if not url_scroll_results:
                break
            if hits > CMR_PAGE_SIZE:
                print('.', end='')
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > CMR_PAGE_SIZE:
            print()
        return urls
    except KeyboardInterrupt:
        quit()

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


def seaice_harvester(path_to_file_dir="", s3=None, on_aws=False):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    # TODO decide on command line argument for different dataset configs
    path_to_yaml = path_to_file_dir + "seaice_harvester_config.yaml"
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
    data_time_scale = config['data_time_scale']


    short_name = config['ds_name'][7:]
    version = '1'



    if not on_aws:
        print("!!downloading files to "+target_dir)
    else:
        print("!!downloading files to "+folder+" and uploading to " +
              target_bucket_name+"/"+config['ds_name'])

    print("======downloading files========")


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


    fq = ['type_s:dataset', f'dataset_s:{config["ds_name"]}']
    query_docs = solr_query(config, fq)


    # setup metadata
    meta = []
    item = {}
    last_success_item = {}
    start = []
    end = []
    years_updated = set()
    chk_time = datetime.datetime.utcnow().strftime(config['date_regex'])
    now = datetime.datetime.utcnow()
    updating = False
    aws_upload = False

    start_year = config['start'][:4]
    end_year = config['end'][:4]
    years = np.arange(int(start_year), int(end_year) + 1)
    start_time = datetime.datetime.strptime(config['start'],config['date_regex'])
    end_time = datetime.datetime.strptime(config['end'],config['date_regex'])
   

    url_list = cmr_search(short_name, version, config['start'], config['end'])

    for year in years:

        iso_dates_at_end_of_month = []
        
        # pull one record per month
        for month in range(1,13):
            # to find the last day of the month, we go up one month, 
            # and back one day
            #   if Jan-Nov, then we'll go forward one month to Feb-Dec

            if month < 12:
                cur_mon_year = np.datetime64(str(year) + '-' + str(month+1).zfill(2))
            # for december we go up one year, and set month to january
            else:
                cur_mon_year = np.datetime64(str(year+1) + '-' + str('01'))
            
            # then back one day
            last_day_of_month = cur_mon_year - np.timedelta64(1,'D')
            
            iso_dates_at_end_of_month.append((str(last_day_of_month)).replace('-', ''))


        url_dict = {}

        for file_date in iso_dates_at_end_of_month:
            end_of_month_url = [url for url in url_list if file_date in url]

            if end_of_month_url:
                url_dict[file_date] = end_of_month_url[0]
        
        for file_date, url in url_dict.items():

            # Date in filename is end date of 30 day period
            filename = url.split('/')[-1]
            local_fp = f'{folder}{config["ds_name"]}_granule.nc' if on_aws else target_dir + filename

            date = getdate(config['regex'],filename)
            date_time = datetime.datetime.strptime(date,"%Y%m%d")
            new_date_format = f'{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z'

            # check if file in download date range
            if (start_time <= date_time) and (end_time >= date_time):
                item = {}
                item['type_s'] = 'harvested'
                item['date_s'] = new_date_format
                item['dataset_s'] = config['ds_name']
                item['hemisphere_s'] = 'nh'

                updating = False
                aws_upload = False

                try:
                    item['source_s'] = url

                    # TODO: find a way to get last modified (see line 436 as well)
                    # get last modified date
                    # timestamp = ftp.voidcmd("MDTM "+url)[4:]    # string
                    # time = parser.parse(timestamp)              # datetime object
                    # timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ") # string
                    # item['modified_time_dt'] = timestamp

                    # compare modified timestamp or if granule previously downloaded
                    # updating = (not newfile in docs.keys()) or (not docs[newfile]['harvest_success_b']) or (datetime.datetime.strptime(docs[newfile]['download_time_dt'], "%Y-%m-%dT%H:%M:%SZ") <= time)
                    updating = (not filename in docs.keys()) or (not docs[filename]['harvest_success_b'])

                    if updating:
                        if not os.path.exists(local_fp):

                            print('Downloading: ' + local_fp)

                            credentials = get_credentials(url)
                            req = Request(url)
                            req.add_header('Authorization', 'Basic {0}'.format(credentials))
                            opener = build_opener(HTTPCookieProcessor())
                            data = opener.open(req).read()
                            open(local_fp, 'wb').write(data)


                        # elif datetime.datetime.fromtimestamp(os.path.getmtime(local_fp)) <= time:
                        elif datetime.datetime.fromtimestamp(os.path.getmtime(local_fp)) <= parser.parse(file_date):

                            print('Updating: ' + local_fp)

                            credentials = get_credentials(url)
                            req = Request(url)
                            req.add_header('Authorization', 'Basic {0}'.format(credentials))
                            opener = build_opener(HTTPCookieProcessor())
                            data = opener.open(req).read()
                            open(local_fp, 'wb').write(data)

                        else:
                            print('File already downloaded and up to date')

                        # calculate checksum and expected file size
                        item['checksum_s'] = md5(local_fp)



                        # =====================================================
                        # ### Push data to s3 bucket
                        # =====================================================
                        output_filename = config['ds_name'] + \
                            '/' + filename if on_aws else filename
                        item['pre_transformation_file_path_s'] = target_dir + filename

                        if on_aws:
                            aws_upload = True
                            print("=========uploading file=========")
                            # print('uploading '+output_filename)
                            target_bucket.upload_file(local_fp, output_filename)
                            item['pre_transformation_file_path_s'] = 's3://' + \
                                config['target_bucket_name']+'/'+output_filename
                            print("======uploading file DONE=======")

                        item['harvest_success_b'] = True
                        item['filename_s'] = filename
                        item['file_size_l'] = os.path.getsize(local_fp)

                        years_updated.add(date[:4])

                except Exception as e:
                    print('error', e)
                    if updating:
                        if aws_upload:
                            print("======aws upload unsuccessful=======")
                            item['message_s'] = 'aws upload unsuccessful'

                        else:
                            print("Download "+filename+" failed.")
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
        ds_meta['source_s'] = url_list[0][:-30]
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
        old_start = datetime.datetime.strptime(
            doc['start_date_dt'], "%Y-%m-%dT%H:%M:%SZ") if 'start_date_dt' in doc.keys() else None
        old_end = datetime.datetime.strptime(
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
