import os
import sys
import numpy as np
from pathlib import Path
from shutil import copyfile

# list all template folders
transformation_template_path = Path(
    f'{Path(__file__).parents[1]}/preprocessing/grid_transformation')
aggregation_template_path = Path(
    f'{Path(__file__).parents[1]}/preprocessing/aggregation_by_year')
podaac_template_path = Path(
    f'{Path(__file__).parents[1]}/harvesters/podaac_harvester')
osisaf_template_path = Path(
    f'{Path(__file__).parents[1]}/harvesters/osisaf_ftp_harvester')
nsidc_template_path = Path(
    f'{Path(__file__).parents[1]}/harvesters/nsidc_ftp_harvester')

# update to template -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# zip list of harvester names and their template paths
harvesters_list = zip(['podaac_harvester', 'osisaf_ftp_harvester', 'nsidc_ftp_harvester'], [
                      podaac_template_path, osisaf_template_path, nsidc_template_path])

# path to harvester and preprocessing folders
path_to_harvesters = Path(f'{Path(__file__).parents[2]}/harvesters')
path_to_preprocessing = Path(f'{Path(__file__).parents[2]}/preprocessing')

# harvesters --------------------------------------------------------------------------------------------------------------------------
update_harvesters = True
if update_harvesters:
    for name, path in harvesters_list:
        # get all dataset directories in harvester folder
        harvester_dirs = [dataset for dataset in os.listdir(
            f'{path_to_harvesters}/{name}') if '.' not in dataset and 'harvested_granule' not in dataset and 'RDEFT4' not in dataset]

        # for each dataset directory, copy each template file to the file already present (excluding the configuration file)
        for dataset in harvester_dirs:
            dataset_harvester_path = Path(
                f'{path_to_harvesters}/{name}/{dataset}')

            for template_file in os.listdir(path):
                if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
                    source_file = Path(f'{path}/{template_file}')
                    destination_file = Path(
                        f'{dataset_harvester_path}/{template_file}')
                    copyfile(source_file, destination_file)

# preprocessing -----------------------------------------------------------------------------------------------------------------------
dirs = [dataset for dataset in os.listdir(
    path_to_preprocessing) if '.' not in dataset]
# for each dataset, copy each template file to the file already present (excluding the configuration file)
# same process for transformation and aggregation but with different directory names

update_preprocessing = False

if update_preprocessing:
    for dataset in dirs:
        ds_transformation_path = Path(
            f'{path_to_preprocessing}/{dataset}/grid_transformation')
        ds_aggregation_path = Path(
            f'{path_to_preprocessing}/{dataset}/aggregation_by_year')

        # transformation
        for template_file in os.listdir(transformation_template_path):
            if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
                source_file = Path(
                    f'{transformation_template_path}/{template_file}')
                destination_file = Path(
                    f'{ds_transformation_path}/{template_file}')
                copyfile(source_file, destination_file)

        # aggregation
        for template_file in os.listdir(aggregation_template_path):
            if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
                source_file = Path(
                    f'{aggregation_template_path}/{template_file}')
                destination_file = Path(
                    f'{ds_aggregation_path}/{template_file}')
                copyfile(source_file, destination_file)
