import os
import time
from datetime import datetime
from pathlib import Path

import requests

from conf.global_settings import SOLR_COLLECTION, SOLR_HOST

def solr_query(fq):
    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = f'{SOLR_HOST}{SOLR_COLLECTION}/select?'
    try:
        response = requests.get(url, params=getVars, headers={'Connection': 'close'})
    except:
        time.sleep(5)
        response = requests.get(url, params=getVars, headers={'Connection': 'close'})

    return response.json()['response']['docs']


def solr_update(update_body, r=False):
    url = f'{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true'
    response = requests.post(url, json=update_body)
    if r:
        return response


def ping_solr():
    url = f'{SOLR_HOST}{SOLR_COLLECTION}/admin/ping'
    requests.get(url)
    return


def core_check():
    url = f'{SOLR_HOST}admin/cores?action=STATUS&core={SOLR_COLLECTION}'
    response = requests.get(url).json()
    if response['status'][SOLR_COLLECTION].keys():
        return True
    return False


def check_grids():
    if not solr_query(['type_s=grid']):
        return True
    return False


def validate_granules():
    granules = solr_query(['type_s=granule'])
    docs_to_remove = []

    for granule in granules:
        file_path = granule['pre_transformation_file_path_s']
        if os.path.exists(file_path):
            continue
        else:
            docs_to_remove.append(granule['id'])

    if docs_to_remove:
        solr_update({'delete': docs_to_remove})

        print(f'Succesfully removed {len(docs_to_remove)} granules from Solr')


def clean_solr(config, grids_to_use):
    """
    Remove harvested, transformed, and descendant entries in Solr for dates
    outside of config date range. Also remove related aggregations, and force
    aggregation rerun for those years.
    """
    dataset_name = config['ds_name']
    config_start = config['start']
    config_end = config['end']

    if config_end == 'NOW':
        config_end = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_query(fq)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

    # Convert config dates to Solr format
    config_start = f'{config_start[:4]}-{config_start[4:6]}-{config_start[6:]}'
    config_end = f'{config_end[:4]}-{config_end[4:6]}-{config_end[6:]}'

    fq = [f'type_s:dataset', f'dataset_s:{dataset_name}']
    dataset_metadata = solr_query(fq)

    if not dataset_metadata:
        return
    else:
        dataset_metadata = dataset_metadata[0]

    print(
        f'Removing Solr documents related to dates outside of configuration start and end dates: \n\t{config_start} to {config_end}.\n')

    # Remove entries earlier than config start date
    fq = f'dataset_s:{dataset_name} AND date_s:[* TO {config_start}}}'
    url = f'{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true'
    requests.post(url, json={'delete': {'query': fq}})

    # Remove entries later than config end date
    fq = f'dataset_s:{dataset_name} AND date_s:{{{config_end} TO *]'
    url = f'{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true'
    requests.post(url, json={'delete': {'query': fq}})

    # Add start and end years to '{grid}_years_updated' field in dataset entry
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
        solr_update(update_body)
