import numpy as np
import xarray as xr
import glob
import sys
import yaml
import requests
from collections import defaultdict
import itertools
import os
from grid_transformation import run_locally_wrapper, solr_query, solr_update


def get_remaining_transformations(config, source_file_path):
    dataset_name = config['dataset_name']

    # Query for grids
    fq = ['type_s:grid']
    docs = solr_query(config, fq)
    grids = [doc['grid_name_s'] for doc in docs]

    # query for fields
    fq = ['type_s:field', f'dataset_s:{dataset_name}']
    docs = solr_query(config, fq)
    fields = [field_entry for field_entry in docs]

    grid_field_combinations = list(itertools.product(grids, fields))

    # query for transformations
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
          f'pre_transformation_file_path_s:"{source_file_path}"']
    docs = solr_query(config, fq)

    if len(docs) > 0:
        existing_transformations = {
            (doc['grid_name_s'], doc['field_s']): doc['origin_checksum_s'] for doc in docs}

        drop_list = []

        for (grid, field) in grid_field_combinations:
            field_name = field['name_s']

            # If transformation exists, must compare checksums and versions for updates
            if (grid, field_name) in existing_transformations:
                # Compare origin checksum for transformed file
                # If it is the same as the harvested checksum,
                # harvested file has not been updated and no new transformation is needed
                fq = [f'dataset_s:{dataset_name}', 'type_s:harvested',
                      f'pre_transformation_file_path_s:"{source_file_path}"']
                harvested_checksum = solr_query(config, fq)[0]['checksum_s']
                origin_checksum = existing_transformations[(grid, field_name)]

                # Compare if transformation version number matches config version
                fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
                      f'pre_transformation_file_path_s:"{source_file_path}"']
                existing_transformation = solr_query(config, fq)[0]

                if existing_transformation['transformation_version_f'] == config['version'] and origin_checksum == harvested_checksum:
                    drop_list.append((grid, field))

        grid_field_combinations = [
            combo for combo in grid_field_combinations if combo not in drop_list]

    # Build dictionary of remaining transformations
    # grid:list of fields
    grid_field_dict = defaultdict(list)

    for grid, field in grid_field_combinations:
        grid_field_dict[grid].append(field)

    return dict(grid_field_dict)


##################################################
if __name__ == "__main__":
    system_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Pull config information
    path_to_yaml = system_path + "/grid_transformation_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get all harvested granule for this dataset
    fq = [f'dataset_s:{config["dataset_name"]}', 'type_s:harvested']
    harvested_granules = solr_query(config, fq)

    for item in harvested_granules:
        f = item.get('pre_transformation_file_path_s', '')

        # Skips items that weren't harvested properly
        if f == '':
            print("ERROR - pre transformation path doesn't exist")
            continue

        # Get transformations to be completed for this file
        remaining_transformations = get_remaining_transformations(config, f)
        if remaining_transformations:
            run_locally_wrapper(
                system_path, f, remaining_transformations, output_dir)
        else:
            print(f'No new transformations for {item["date_s"]}')

    # Update Solr dataset entry status to transformed
    fq = [f'dataset_s:{config["dataset_name"]}', 'type_s:dataset']
    dataset_metadata = solr_query(config, fq)[0]

    update_body = [{
        "id": dataset_metadata['id'],
        "status_s": {"set": 'transformed'}
    }]
    solr_update(config, update_body)
