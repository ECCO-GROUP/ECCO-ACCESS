import os
import re
import ssl
import sys
import json
import yaml
import base64
import netrc
import hashlib
import requests
import itertools
import numpy as np
import xarray as xr
from pathlib import Path
from dateutil import parser
from datetime import datetime
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from xml.etree.ElementTree import fromstring
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor


CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def clean_solr(config, solr_host, grids_to_use, solr_collection_name):
    """
    Remove harvested, transformed, and descendant entries in Solr for dates
    outside of config date range. Also remove related aggregations, and force
    aggregation rerun for those years.
    """
    dataset_name = config['ds_name']

    # Dates already in Solr format
    config_start = config['start']
    config_end = config['end']

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_query(config, solr_host, fq, solr_collection_name)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

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

    # Add start and end years to 'years_updated' field in dataset entry
    # Forces the bounding years to be re-aggregated to account for potential
    # removed dates
    start_year = config_start[:4]
    end_year = config_end[:4]
    update_body = [{
        "id": dataset_metadata['id']
    }]

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


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    errprefix = ''
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(
            urlparse(URS_URL).hostname)
        errprefix = 'netrc error: '
    except Exception as e:
        if (not ('No such file' in str(e))):
            print('netrc error: {0}'.format(str(e)))
        username = None
        password = None

    while not credentials:
        if not username:
            username = 'ecco_access'  # hardcoded username
            password = 'ECCOAccess1'  # hardcoded password
        credentials = '{0}:{1}'.format(username, password)
        credentials = base64.b64encode(
            credentials.encode('ascii')).decode('ascii')

        if url:
            try:
                req = Request(url)
                req.add_header('Authorization',
                               'Basic {0}'.format(credentials))
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
        params += '&producer_granule_id[]={0}{1}'.format(
            filename_filter, option)
    return CMR_FILE_URL + params


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
                headers = {k.lower(): v for k, v in dict(
                    response.info()).items()}
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


def solr_query(config, solr_host, fq, solr_collection_name):
    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


def solr_update(config, solr_host, update_body, solr_collection_name, r=False):
    url = solr_host + solr_collection_name + '/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def seaice_harvester(config_path='', output_path='', s3=None, on_aws=False, solr_info='', grids_to_use=[]):
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
    end_time = config['end']
    regex = config['regex']
    date_format = config['date_format']
    data_time_scale = config['data_time_scale']
    target_bucket_name = config['target_bucket_name']
    target_dir = f'{output_path}{dataset_name}/harvested_granules/'
    folder = f'/tmp/{dataset_name}/'

    entries_for_solr = []
    last_success_item = {}
    start_times = []
    end_times = []
    chk_time = datetime.utcnow().strftime(date_regex)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # =====================================================
    # Code to import ecco utils locally...
    # =====================================================
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[5]}/ecco-cloud-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    # =====================================================
    # Setup AWS Target Bucket
    # =====================================================
    if on_aws:
        target_bucket = s3.Bucket(target_bucket_name)
        if solr_info:
            solr_host = solr_info['solr_url']
            solr_collection_name = solr_info['solr_collection_name']
        else:
            solr_host = config['solr_host_local']
            solr_collection_name = config['solr_collection_name']
        clean_solr(config, solr_host, grids_to_use, solr_collection_name)
        print(
            f'Downloading {dataset_name} files and uploading to {target_bucket_name}/{dataset_name}\n')
    else:
        target_bucket = None
        if solr_info:
            solr_host = solr_info['solr_url']
            solr_collection_name = solr_info['solr_collection_name']
        else:
            solr_host = config['solr_host_local']
            solr_collection_name = config['solr_collection_name']
        clean_solr(config, solr_host, grids_to_use, solr_collection_name)
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
    existing_descendants_docs = solr_query(
        config, solr_host, fq, solr_collection_name)

    # Dictionary of existing descendants docs
    # descendant doc date : solr entry for that doc
    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            if 'hemisphere_s' in doc.keys() and doc['hemisphere_s']:
                key = (doc['date_s'], doc['hemisphere_s'])
            else:
                key = doc['date_s']
            descendants_docs[key] = doc

    # =====================================================
    # Setup NSIDC RDEFT4 loop variables
    # =====================================================
    short_name = 'RDEFT4'
    version = '1'

    item = {}

    start_year = start_time[:4]
    end_year = end_time[:4]
    years = np.arange(int(start_year), int(end_year) + 1)
    start_time_dt = datetime.strptime(start_time, date_regex)
    end_time_dt = datetime.strptime(end_time, date_regex)

    url_list = cmr_search(short_name, version, start_time, end_time)

    for year in years:

        iso_dates_at_end_of_month = []

        # pull one record per month
        for month in range(1, 13):
            # to find the last day of the month, we go up one month,
            # and back one day
            #   if Jan-Nov, then we'll go forward one month to Feb-Dec
            if month < 12:
                cur_mon_year = np.datetime64(
                    str(year) + '-' + str(month+1).zfill(2))
            # for december we go up one year, and set month to january
            else:
                cur_mon_year = np.datetime64(str(year+1) + '-' + str('01'))

            # then back one day
            last_day_of_month = cur_mon_year - np.timedelta64(1, 'D')

            iso_dates_at_end_of_month.append(
                (str(last_day_of_month)).replace('-', ''))

        url_dict = {}

        for file_date in iso_dates_at_end_of_month:
            end_of_month_url = [url for url in url_list if file_date in url]

            if end_of_month_url:
                url_dict[file_date] = end_of_month_url[0]

        for file_date, url in url_dict.items():

            # Date in filename is end date of 30 day period
            filename = url.split('/')[-1]

            date = getdate(regex, filename)
            date_time = datetime.strptime(date, "%Y%m%d")
            new_date_format = f'{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z'

            tb, _ = ea.make_time_bounds_from_ds64(np.datetime64(
                new_date_format) + np.timedelta64(1, 'D'), 'AVG_MON')
            new_date_format = f'{str(tb[0])[:10]}T00:00:00Z'
            year = new_date_format[:4]

            local_fp = f'{folder}{dataset_name}_granule.nc' if on_aws else f'{target_dir}{year}/{filename}'

            if not os.path.exists(f'{target_dir}{year}/'):
                os.makedirs(f'{target_dir}{year}/')

            # Extract modified time from xml file associated with .nc file
            xml_url = f'{url}.xml'
            credentials = get_credentials(xml_url)
            req = Request(xml_url)
            req.add_header('Authorization',
                           'Basic {0}'.format(credentials))
            opener = build_opener(HTTPCookieProcessor())
            data = opener.open(req).read()
            root = fromstring(data)
            modified_time = root.find(
                'GranuleURMetaData').find('LastUpdate').text.replace(' ', 'T')
            modified_time = f'{modified_time[:-1]}Z'
            original_start_time = root.find('GranuleURMetaData').find(
                'RangeDateTime').find('RangeBeginningTime').text
            original_start_date = root.find('GranuleURMetaData').find(
                'RangeDateTime').find('RangeBeginningDate').text
            original_end_time = root.find('GranuleURMetaData').find(
                'RangeDateTime').find('RangeEndingTime').text
            original_end_date = root.find('GranuleURMetaData').find(
                'RangeDateTime').find('RangeEndingDate').text

            time_coverage_start = f'{original_start_date}T{original_start_time}'
            time_coverage_end = f'{original_end_date}T{original_end_time}'

            # check if file in download date range
            if (start_time_dt <= date_time) and (end_time_dt >= date_time):
                item = {}
                item['type_s'] = 'harvested'
                item['date_s'] = new_date_format
                item['dataset_s'] = dataset_name
                item['filename_s'] = filename
                item['source_s'] = url
                item['modified_time_dt'] = modified_time

                descendants_item = {}
                descendants_item['type_s'] = 'descendants'
                descendants_item['date_s'] = item["date_s"]
                descendants_item['dataset_s'] = item['dataset_s']
                descendants_item['filename_s'] = filename
                descendants_item['source_s'] = item['source_s']

                updating = False
                aws_upload = False

                try:
                    updating = (not filename in docs.keys()) or \
                               (not docs[filename]['harvest_success_b']) or \
                               (docs[filename]['download_time_dt']
                                < modified_time)

                    # If updating, download file if necessary
                    if updating:
                        # If file doesn't exist locally, download it
                        if not os.path.exists(local_fp):
                            print(f' - Downloading {filename} to {local_fp}')

                            credentials = get_credentials(url)
                            req = Request(url)
                            req.add_header('Authorization',
                                           'Basic {0}'.format(credentials))
                            opener = build_opener(HTTPCookieProcessor())
                            data = opener.open(req).read()
                            open(local_fp, 'wb').write(data)

                            ds = xr.open_dataset(local_fp)
                            ds.attrs['time_coverage_start'] = time_coverage_start
                            ds.attrs['time_coverage_end'] = time_coverage_end
                            ds.to_netcdf(local_fp)

                        # If file exists locally, but is out of date, download it
                        elif datetime.fromtimestamp(os.path.getmtime(local_fp)) <= modified_time:
                            print(
                                f' - Updating {filename} and downloading to {local_fp}')

                            credentials = get_credentials(url)
                            req = Request(url)
                            req.add_header('Authorization',
                                           'Basic {0}'.format(credentials))
                            opener = build_opener(HTTPCookieProcessor())
                            data = opener.open(req).read()
                            open(local_fp, 'wb').write(data)

                            ds = xr.open_dataset(local_fp)
                            ds.attrs['time_coverage_start'] = time_coverage_start
                            ds.attrs['time_coverage_end'] = time_coverage_end
                            ds.to_netcdf(local_fp)

                        else:
                            print(
                                f' - {filename} already downloaded and up to date')

                        if filename in docs.keys():
                            item['id'] = docs[filename]['id']

                        # calculate checksum and expected file size
                        item['checksum_s'] = md5(local_fp)
                        item['pre_transformation_file_path_s'] = local_fp

                        # =====================================================
                        # ### Push data to s3 bucket
                        # =====================================================
                        if on_aws:
                            aws_upload = True
                            output_filename = f'{dataset_name}/{filename}' if on_aws else filename
                            print("=========uploading file=========")
                            # print('uploading '+output_filename)
                            target_bucket.upload_file(
                                local_fp, output_filename)
                            item['pre_transformation_file_path_s'] = f's3://{target_bucket_name}/{output_filename}'
                            print("======uploading file DONE=======")

                        item['harvest_success_b'] = True
                        item['file_size_l'] = os.path.getsize(local_fp)

                    else:
                        print(
                            f' - {filename} already downloaded and up to date')

                except Exception as e:
                    print('error', e)
                    if updating:
                        if aws_upload:
                            print("======aws upload unsuccessful=======")
                            item['message_s'] = 'aws upload unsuccessful'

                        else:
                            print(f'    - {filename} failed to download')

                        item['harvest_success_b'] = False
                        item['filename'] = ''
                        item['pre_transformation_file_path_s'] = ''
                        item['file_size_l'] = 0

                if updating:
                    item['download_time_dt'] = chk_time

                    # Update Solr entry using id if it exists

                    key = descendants_item['date_s']

                    if key in descendants_docs.keys():
                        descendants_item['id'] = descendants_docs[key]['id']

                    descendants_item['harvest_success_b'] = item['harvest_success_b']
                    descendants_item['pre_transformation_file_path_s'] = item['pre_transformation_file_path_s']
                    entries_for_solr.append(descendants_item)

                    start_times.append(datetime.strptime(
                        new_date_format, date_regex))
                    end_times.append(datetime.strptime(
                        new_date_format, date_regex))

            # add item to metadata json
            entries_for_solr.append(item)
            # store meta for last successful download
            last_success_item = item

    print(f'\nDownloading {dataset_name} complete\n')

    if entries_for_solr:
        # post granule metadata documents for downloaded granules
        r = solr_update(config, solr_host, entries_for_solr,
                        solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully created or updated Solr harvested documents')
        else:
            print('Failed to create Solr harvested documents')

    # Query for Solr failed harvest documents
    fq = ['type_s:harvested',
          f'dataset_s:{dataset_name}', f'harvest_success_b:false']
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

    overall_start = min(start_times) if len(start_times) > 0 else None
    overall_end = max(end_times) if len(end_times) > 0 else None

    fq = ['type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_query = solr_query(config, solr_host, fq, solr_collection_name)

    update = (len(dataset_query) == 1)

    # =====================================================
    # Solr dataset entry
    # =====================================================
    if not update:
        # -----------------------------------------------------
        # Create Solr Dataset-level Document if doesn't exist
        # -----------------------------------------------------
        ds_meta = {}
        ds_meta['type_s'] = 'dataset'
        ds_meta['dataset_s'] = dataset_name
        ds_meta['short_name_s'] = config['original_dataset_short_name']
        ds_meta['source_s'] = url_list[0][:-30]
        ds_meta['data_time_scale_s'] = data_time_scale
        ds_meta['date_format_s'] = date_format
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

        # if no ds entry yet and no qualifying downloads, still create ds entry without download time
        if updating:
            ds_meta['last_download_dt'] = last_success_item['download_time_dt']

        ds_meta['harvest_status_s'] = harvest_status

        # Post document
        r = solr_update(config, solr_host, [ds_meta],
                        solr_collection_name, r=True)

        if r.status_code == 200:
            print('Successfully created Solr dataset document')
        else:
            print('Failed to create Solr dataset document')

        # TODO: update for changes in yaml (incinerate and rewrite)
        # modify updating variable to account for updates in config and then incinerate and rewrite
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

    # if record exists, update download time, converage start date, coverage end date
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
