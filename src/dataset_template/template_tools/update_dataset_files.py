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