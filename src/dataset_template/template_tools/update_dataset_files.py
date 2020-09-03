import numpy as np
from pathlib import Path
from shutil import copyfile
import os

transformation_template_path = Path(str(Path(__file__).parents[1]) + '/preprocessing/grid_transformation')
aggregation_template_path = Path(str(Path(__file__).parents[1]) + '/preprocessing/aggregation_by_year')

path_to_preprocessing = Path(str(Path(__file__).parents[2]) + '/preprocessing')

dirs = [dataset for dataset in os.listdir(path_to_preprocessing) if '.' not in dataset]

for dataset in dirs:
    ds_transformation_path = Path(f'{path_to_preprocessing}/{dataset}/grid_transformation')
    ds_aggregation_path = Path(f'{path_to_preprocessing}/{dataset}/aggregation_by_year')

    for template_file in os.listdir(transformation_template_path):
        if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
            source_file = Path(f'{transformation_template_path}/{template_file}')
            destination_file = Path(f'{ds_transformation_path}/{template_file}')
            copyfile(source_file, destination_file)

    for template_file in os.listdir(aggregation_template_path):
        if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
            source_file = Path(f'{aggregation_template_path}/{template_file}')
            destination_file = Path(f'{ds_aggregation_path}/{template_file}')
            copyfile(source_file, destination_file)

update_harvesters = False

if update_harvesters:
    podaac_template_path = Path(str(Path(__file__).parents[1]) + '/harvesters/podaac_harvester')
    osisaf_template_path = Path(str(Path(__file__).parents[1]) + '/harvesters/osisaf_ftp_harvester')
    nsidc_template_path = Path(str(Path(__file__).parents[1]) + '/harvesters/nsidc_ftp_harvester')
    harvesters_list = zip(['podaac_harvester', 'osisaf_ftp_harvester', 'nsidc_ftp_harvester'], [podaac_template_path, osisaf_template_path, nsidc_template_path])

    path_to_harvesters = Path(str(Path(__file__).parents[2]) + '/harvesters')

    for name, path in harvesters_list:
        harvester_dirs = [dataset for dataset in os.listdir(f'{path_to_harvesters}/{name}') if '.' not in dataset and 'harvested_granule' not in dataset and 'RDEFT4' not in dataset]
        
        for dataset in harvester_dirs:
            dataset_harvester_path = Path(f'{path_to_harvesters}/{name}/{dataset}')

            for template_file in os.listdir(path):
                if 'config' not in template_file and '.DS' not in template_file and '__pycache__' != template_file:
                    source_file = Path(f'{path}/{template_file}')
                    destination_file = Path(f'{dataset_harvester_path}/{template_file}')
                    copyfile(source_file, destination_file)