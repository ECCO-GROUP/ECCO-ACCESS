import itertools
import logging
import logging.config
import os
from collections import defaultdict
from multiprocessing import Pool

import requests

from utils import solr_utils
from grid_transformation.grid_transformation import run_locally_wrapper

logs_path = 'ecco_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def get_remaining_transformations(config, granule_file_path, grids):
    """
    Given a single granule, the function uses Solr to find all combinations of
    grids and fields that have yet to be transformed. It returns a dictionary
    where the keys are grids and the values are lists of fields.
    """
    dataset_name = config['ds_name']

    # Query for fields
    fq = ['type_s:field', f'dataset_s:{dataset_name}']
    docs = solr_utils.solr_query(fq)
    fields = [field_entry for field_entry in docs]

    # Cartesian product of grid/field combinations
    grid_field_combinations = list(itertools.product(grids, fields))

    # Query for existing transformations
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
          f'pre_transformation_file_path_s:"{granule_file_path}"']
    docs = solr_utils.solr_query(fq)

    # if a transformation entry exists for this granule, check to see if the
    # checksum of the harvested granule matches the checksum recorded in the
    # transformation entry for this granule, if not then we have to retransform
    # also check to see if the version of the transformation code recorded in
    # the entry matches the current version of the transformation code, if not
    # redo the transformation.

    # these checks are made for each grid/field pair associated with the
    # harvested granule)

    if len(docs) > 0:

        # Dictionary where key is grid, field tuple and value is harvested granule checksum
        # For existing transformations pulled from Solr
        existing_transformations = {
            (doc['grid_name_s'], doc['field_s']): doc['origin_checksum_s'] for doc in docs}

        drop_list = []

        for (grid, field) in grid_field_combinations:
            field_name = field['name_s']

            # If transformation exists, must compare checksums and versions for updates
            if (grid, field_name) in existing_transformations:

                # Query for harvested granule checksum
                fq = [f'dataset_s:{dataset_name}', 'type_s:granule',
                      f'pre_transformation_file_path_s:"{granule_file_path}"']
                harvested_checksum = solr_utils.solr_query(fq)[0]['checksum_s']

                origin_checksum = existing_transformations[(grid, field_name)]

                # Query for existing transformation
                fq = [f'dataset_s:{dataset_name}', 'type_s:transformation',
                      f'pre_transformation_file_path_s:"{granule_file_path}"']
                transformation = solr_utils.solr_query(fq)[0]

                # Triple if:
                # 1. do we have a version entry,
                # 2. compare transformation version number and current transformation version number
                # 3. compare checksum of harvested file (currently in solr) and checksum
                #    of the harvested file that was previously transformed (recorded in transformation entry)
                if 'transformation_version_f' in transformation.keys() and \
                        transformation['transformation_version_f'] == config['version'] and \
                        origin_checksum == harvested_checksum:

                    # all tests passed, we do not need to redo the transformation
                    # for this grid/field pair

                    # Add grid/field combination to drop_list
                    drop_list.append((grid, field))

        # Remove drop_list grid/field combinations from list of remaining transformations
        grid_field_combinations = [
            combo for combo in grid_field_combinations if combo not in drop_list]

    # Build dictionary of remaining transformations
    # -- grid_field_dict has grid key, entries is list of fields
    grid_field_dict = defaultdict(list)

    for grid, field in grid_field_combinations:
        grid_field_dict[grid].append(field)

    return dict(grid_field_dict)


def delete_mismatch_transformations(config):
    """
    Function called when using the wipe_transformations pipeline argument. Queries
    Solr for all transformation entries for the current dataset and compares the
    transformation version in Solr and in the config YAML. If they differ, the
    function deletes the transformed file from disk and the entry from Solr.
    """
    dataset_name = config['ds_name']

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']
    config_version = config['version']

    # Query for existing transformations
    fq = [f'dataset_s:{dataset_name}', 'type_s:transformation']
    transformations = solr_utils.solr_query(fq)

    for transformation in transformations:
        if transformation['transformation_version_f'] != config_version:
            # Remove file from disk
            if os.path.exists(transformation['transformation_file_path_s']):
                os.remove(transformation['transformation_file_path_s'])

            # Remove transformation entry from Solr
            url = f'{solr_host}{solr_collection_name}/update?commit=true'
            requests.post(url, json={'delete': [transformation['id']]})


def multiprocess_transformation(granule, config, output_path, grids):
    """
    Callable function that performs the actual transformation on a granule.
    """

    # f is file path to granule from solr
    f = granule.get('pre_transformation_file_path_s', '')

    # Skips granules that weren't harvested properly
    if f == '':
        print("ERROR - pre transformation path doesn't exist")
        return ('', '')

    # Get transformations to be completed for this file
    remaining_transformations = get_remaining_transformations(config, f, grids)

    # Perform remaining transformations
    if remaining_transformations:
        grids_updated, year = run_locally_wrapper(
            f, remaining_transformations, output_path, config, verbose=False)

        return (grids_updated, year)
    else:
        print(
            f' - CPU id {os.getpid()} no new transformations for {granule["filename_s"]}')
        return ('', '')


def main(config, output_path, multiprocessing=False, user_cpus=1, wipe=False, grids_to_use=[]):
    """
    This function performs all remaining grid/field transformations for all harvested
    granules for a dataset. It also makes use of multiprocessing to perform multiple
    transformations at the same time. After all transformations have been attempted,
    the Solr dataset entry is updated with additional metadata.
    """

    # import grid_transformation

    dataset_name = config['ds_name']

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    transformation_version = config['version']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if wipe:
        print(
            'Removing transformations with out of sync version numbers from Solr and disk')
        delete_mismatch_transformations(config)

    # Get all harvested granules for this dataset
    fq = [f'dataset_s:{dataset_name}',
          'type_s:granule', 'harvest_success_b:true']
    harvested_granules = solr_utils.solr_query(fq)

    years_updated = defaultdict(list)

    # Query for grids
    if not grids_to_use:
        fq = ['type_s:grid']
        docs = solr_utils.solr_query(fq)
        grids = [doc['grid_name_s'] for doc in docs]
    else:
        grids = grids_to_use

    if multiprocessing:
        # PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING
        # Query for dataset metadata
        fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
        dataset_metadata = solr_utils.solr_query(fq)[0]

        # Precompute grid factors using one dataset data file
        # (or one from each hemisphere, if data is hemispherical) before running main loop
        data_for_factors = []
        for grid in grids:
            data_for_factors = []
            nh_added = False
            sh_added = False

            # Find appropriate granule(s) to use for factor calculation
            for granule in harvested_granules:
                if 'hemisphere_s' in granule.keys():
                    hemi = f'_{granule["hemisphere_s"]}'
                else:
                    hemi = ''

                grid_factors = f'{grid}{hemi}_factors_path_s'
                grid_factors_version = f'{grid}{hemi}_factors_version_f'

                if grid_factors in dataset_metadata.keys() and transformation_version == dataset_metadata[grid_factors_version]:
                    continue

                file_path = granule.get('pre_transformation_file_path_s', '')
                if file_path:
                    if hemi:
                        # Get one of each
                        if hemi == '_nh' and not nh_added:
                            data_for_factors.append(granule)
                            nh_added = True
                        elif hemi == '_sh' and not sh_added:
                            data_for_factors.append(granule)
                            sh_added = True
                        if nh_added and sh_added:
                            break
                    else:
                        data_for_factors.append(granule)
                        break

        # Actually perform transformation on chosen granule(s)
        # This will generate factors and avoid redundant calculations when using multiprocessing
        for granule in data_for_factors:
            file_path = granule['pre_transformation_file_path_s']

            # Get transformations to be completed for this file
            remaining_transformations = get_remaining_transformations(
                config, file_path, grids)

            grids_updated, year = run_locally_wrapper(
                file_path, remaining_transformations, output_path, config, verbose=True)

            for grid in grids_updated:
                if year not in years_updated[grid]:
                    years_updated[grid].append(year)
        # END PRE GENERATE FACTORS TO ACCOMODATE MULTIPROCESSING

        # BEGIN MULTIPROCESSING
        # Create list of tuples of function arguments (necessary for using pool.starmap)
        multiprocess_tuples = [(granule, config, output_path, grids)
                               for granule in harvested_granules]

        grid_years_list = []

        print('\nUSING MULTIPROCESSING. LOW VERBOSITY FOR TRANSFORMATIONS.\n')

        # for grid in grids:
        print(f'Running transformations for {grids} grids\n')

        with Pool(processes=user_cpus) as pool:
            grid_years_list = pool.starmap(
                multiprocess_transformation, multiprocess_tuples)
            pool.close()
            pool.join()

        for (grids, year) in grid_years_list:
            if grids and year:
                for grid in grids:
                    if year not in years_updated[grid]:
                        years_updated[grid].append(year)

    else:
        for granule in harvested_granules:
            # f is file path to granule from solr
            f = granule.get('pre_transformation_file_path_s', '')

            # Skips granules that weren't harvested properly
            if f == '':
                print("ERROR - pre transformation path doesn't exist")
                continue

            # Get transformations to be completed for this file
            remaining_transformations = get_remaining_transformations(
                config, f, grids)

            # Perform remaining transformations
            if remaining_transformations:
                grids_updated, year = run_locally_wrapper(
                    f, remaining_transformations, output_path, config, verbose=True)

                for grid in grids_updated:
                    if grid in years_updated.keys():
                        if year not in years_updated[grid]:
                            years_updated[grid].append(year)
                    else:
                        years_updated[grid] = [year]
            else:
                print(
                    f' - CPU id {os.getpid()} no new transformations for {granule["filename_s"]}')

    # Query Solr for dataset metadata
    fq = [f'dataset_s:{dataset_name}', 'type_s:dataset']
    dataset_metadata = solr_utils.solr_query(fq)[0]

    # Query Solr for successful transformation documents
    fq = [f'dataset_s:{dataset_name}',
          'type_s:transformation', 'success_b:true']
    successful_transformations = solr_utils.solr_query(fq)

    # Query Solr for failed transformation documents
    fq = [f'dataset_s:{dataset_name}',
          'type_s:transformation', 'success_b:false']
    failed_transformations = solr_utils.solr_query(fq)

    transformation_status = f'All transformations successful'

    if not successful_transformations and not failed_transformations:
        transformation_status = f'No transformations performed'
    elif not successful_transformations:
        transformation_status = f'No successful transformations'
    elif failed_transformations:
        transformation_status = f'{len(failed_transformations)} transformations failed'

    # Update Solr dataset entry years_updated list and status to transformed
    update_body = [{
        "id": dataset_metadata['id'],
        "transformation_status_s": {"set": transformation_status},
    }]

    # Combine Solr dataset entry years_updated list with transformation years_updated
    for grid in years_updated.keys():
        solr_grid_years = f'{grid}_years_updated_ss'
        if solr_grid_years in dataset_metadata.keys():
            existing_years = dataset_metadata[solr_grid_years]

            for year in years_updated[grid]:
                if year not in existing_years:
                    existing_years.append(year)

        else:
            existing_years = years_updated[grid]

        update_body[0][solr_grid_years] = {"set": existing_years}

    r = solr_utils.solr_update(update_body, r=True)

    if r.status_code == 200:
        print(
            f'\nSuccessfully updated Solr with transformation information for {dataset_name}\n')
    else:
        print(
            f'\nFailed to update Solr with transformation information for {dataset_name}\n')

    return transformation_status


##################################################
if __name__ == "__main__":
    main()
