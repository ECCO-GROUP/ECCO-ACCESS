import os
import sys
import json
import yaml
import hashlib
import requests
import xarray as xr
from datetime import datetime


# Creates checksum from filename
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Queries Solr based on config information and filter query
# Returns list of Solr entries (docs)
def solr_query(config, solr_host, fq):
    solr_collection_name = config['solr_collection_name']

    getVars = {'q': '*:*',
               'fq': fq,
               'rows': 300000}

    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    return response.json()['response']['docs']


# Posts update to Solr with provided update body
# Optional return of posting status code
def solr_update(config, solr_host, update_body, r=False):
    solr_collection_name = config['solr_collection_name']

    url = solr_host + solr_collection_name + '/update?commit=true'

    if r:
        return requests.post(url, json=update_body)
    else:
        requests.post(url, json=update_body)


def main(path=''):
    # =====================================================
    # Read configurations from YAML file
    # =====================================================
    if path:
        path_to_yaml = f'{path}/grids_config.yaml'
        path_to_file_dir = f'{path}/grids/'
    else:
        path_to_yaml = f'{os.path.dirname(sys.argv[0])}/grids_config.yaml'
        path_to_file_dir = f'{os.path.dirname(sys.argv[0])}/grids/'
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    solr_host = config['solr_host']

    # =====================================================
    # Scan directory for grid types
    # =====================================================
    grid_files = [f for f in os.listdir(path_to_file_dir) if os.path.isfile(
        os.path.join(path_to_file_dir, f))]

    # =====================================================
    # Extract grid names from netCDF
    # =====================================================
    grids = []

    # Assumes grids conform to metadata standard (see documentation)
    for grid_file in grid_files:
        if config['grids_to_use']:
            if grid_file in config['grids_to_use']:
                ds = xr.open_dataset(path_to_file_dir + grid_file)

                grid_name = ds.attrs['name']
                grid_type = ds.attrs['type']
                grids.append((grid_name, grid_type, grid_file))
        else:
            ds = xr.open_dataset(path_to_file_dir + grid_file)

            grid_name = ds.attrs['name']
            grid_type = ds.attrs['type']
            grids.append((grid_name, grid_type, grid_file))

    # =====================================================
    # Query for Solr Grid-type Documents
    # =====================================================
    fq = ['type_s:grid']
    docs = solr_query(config, solr_host, fq)

    grids_in_solr = []

    if len(docs) > 0:
        for doc in docs:
            grids_in_solr.append(doc['grid_name_s'])

    # =====================================================
    # Create Solr grid-type document for each missing grid type
    # =====================================================
    for grid_name, grid_type, grid_file in grids:

        grid_path = path_to_file_dir + grid_file
        update_body = []

        if grid_name not in grids_in_solr:
            grid_meta = {}
            grid_meta['type_s'] = 'grid'
            grid_meta['grid_type_s'] = grid_type
            grid_meta['grid_name_s'] = grid_name

            if '\\' in grid_path:
                grid_path = grid_path.replace('\\', '/')
            grid_meta['grid_path_s'] = grid_path

            grid_meta['date_added_dt'] = datetime.utcnow().strftime(
                "%Y-%m-%dT%H:%M:%SZ")

            grid_meta['grid_checksum_s'] = md5(grid_path)
            update_body.append(grid_meta)
        else:
            current_checksum = md5(grid_path)

            for doc in docs:
                if doc['grid_name_s'] == grid_name:
                    solr_checksum = doc['grid_checksum_s']

            if current_checksum != solr_checksum:
                # Delete previous grid's transformations from Solr
                update_body = {
                    "delete": {
                        "query": f'type_s:transformation AND grid_name_s:{grid_name}'
                    }
                }

                r = solr_update(config, solr_host, update_body, r=True)

                if r.status_code == 200:
                    print(
                        f'Successfully deleted Solr transformation documents for {grid_name}')
                else:
                    print(
                        f'Failed to delete Solr transformation documents for {grid_name}')

                # Delete previous grid's aggregations from Solr
                update_body = {
                    "delete": {
                        "query": f'type_s:aggregation AND grid_name_s:{grid_name}'
                    }
                }

                r = solr_update(config, solr_host, update_body, r=True)

                if r.status_code == 200:
                    print(
                        f'Successfully deleted Solr aggregation documents for {grid_name}')
                else:
                    print(
                        f'Failed to delete Solr aggregation documents for {grid_name}')

                # Update grid on Solr
                fq = [f'grid_name_s:{grid_name}', 'type_s:grid']
                grid_metadata = solr_query(config, solr_host, fq)[0]

                update_body = [
                    {
                        "id": grid_metadata['id'],
                        "grid_type_s": {"set": grid_type},
                        "grid_name_s": {"set": grid_name},
                        "grid_checksum_s": {"set": current_checksum},
                        "date_added_dt": {"set": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
                    }
                ]

        r = solr_update(config, solr_host, update_body, r=True)

        if r.status_code == 200:
            print(f'Successfully updated {grid_name} Solr grid document')
        else:
            print(f'Failed to update Solr {grid_name} grid document')


if __name__ == '__main__':
    main()
