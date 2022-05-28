"""
Created May 18, 2022

Author: Duncan Bark

"""

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path

sys.path.append(f'{Path(__file__).parent.resolve()}')
from eccov4r4_gen_for_podaac_cloud import generate_netcdfs


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--upload_model_to_S3', default=False, action='store_true',
                        help='Upload model output data to provided directory (or S3 bucket) in config file')

    parser.add_argument('--process_data', default=False, action='store_true',
                        help='Starts processing model data using config file values')

    parser.add_argument('--time_steps_to_process', nargs="+",
                        help='which time steps to process')

    parser.add_argument('--grouping_to_process', type=int,
                        help='which dataset grouping to process, there are 20 in v4r4')

    parser.add_argument('--product_type', type=str, choices=['latlon', 'native'],
                        help='one of either "latlon" or "native" ')

    parser.add_argument('--output_freq_code', type=str, choices=['AVG_MON','AVG_DAY','SNAPSHOT'],
                        help='one of AVG_MON, AVG_DAY, or SNAPSHOT')

    parser.add_argument('--output_dir', type=str, default=f'{Path(__file__).parent.parent.resolve() / "temp_output"}',
                        help='output directory')

    parser.add_argument('--debug', default=False, action='store_true',
                        help='Sets debug flag (additional print outs and skips processing')
    return parser


if __name__ == "__main__":
    # output_freq_codes  one of either 'AVG_DAY' or 'AVG_MON'
    
    # time_steps_to_process : one of
    # 'all'
    # a list [1,3,4,...]

    # grouping to process:
    #   a single number

    # diags_root:
    #   ECCO FIELD INPUT DIRECTORY
    #   model diagnostic output
    #   subdirectories must be:
    #       'diags_all/diags_mon'
    #       'diags_all/diags_mon'
    #       'diags_all/diags_inst'

    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    dict_key_args = {key: value for key, value in args._get_kwargs()} 

    # Testing/setup paths and config -------------------------------------
    path_to_yaml = Path(__file__).parent.resolve() / 'configs' / 'gen_netcdf_config.yaml'
    with open(path_to_yaml, "r") as f:
        config = yaml.load(f, yaml.Loader)


    # Load directories (local vs AWS)
    # Default directories
    parent_dir = Path(__file__).parent.resolve()
    mapping_factors_dir = parent_dir / 'mapping_factors'
    diags_root = parent_dir / 'diags_all'
    metadata_json_dir = parent_dir / 'metadata' / 'ECCov4r4_metadata_json'
    podaac_dir = parent_dir / 'metadata' / 'ECCov4r4_metadata_json' / 'PODAAC_datasets-revised_20210226.5.csv'
    ecco_grid_dir = parent_dir / 'ecco_grids'
    ecco_grid_dir_mds = parent_dir / 'ecco_grids'
    output_dir_base = parent_dir / 'temp_output'

    local = True
    if local:
        print('\nGetting local directories from config file')
        if config['mapping_factors_dir'] != '':
            mapping_factors_dir = Path(config['mapping_factors_dir'])
        if config['model_data_dir'] != '':
            diags_root = Path(config['model_data_dir'])

        # METADATA
        if config['metadata_json_dir'] != '':
            metadata_json_dir = Path(config['metadata_json_dir'])
        if config['podaac_dir'] != '':
            podaac_dir = Path(config['podaac_dir'])

        if config['ecco_grid_dir'] != '':
            ecco_grid_dir = Path(config['ecco_grid_dir'])
        if config['ecco_grid_dir_mds'] != '':
            ecco_grid_dir_mds = Path(config['ecco_grid_dir_mds'])

        if config['output_dir'] != '':
            output_dir_base = config['output_dir']
    else:
        print('\nGetting AWS Cloud directories from config file')
        # mapping_factors_dir = config['mapping_factors_dir_cloud']
    
    # PODAAC fields
    ecco_grid_filename = config['ecco_grid_filename']

    # Define precision of output files, float32 is standard
    array_precision = np.float32

    debug_mode = False

    # Get all configurations
    all_jobs = []
    with open(f'{Path(__file__).parent.resolve() / "configs" / "jobs.txt"}', 'r') as j:
        for line in j:
            if '#' in line:
                continue
            line_vals = line.strip().split(',')
            all_jobs.append([int(line_vals[0]), line_vals[1], line_vals[2], line_vals[3]])


    # **********
    # TODO: Run mapping_factors for all jobs and include path to correct factors in running of the job below
    # **********


    # loop through all jobs
    # this is where each lambda job would be created
    for (grouping_to_process, product_type, output_freq_code, time_steps_to_process) in all_jobs:

        # **********
        # TODO: CREATE LAMBDA REQUEST FOR EACH "JOB"
        # **********


        print(f'time_steps_to_process: {time_steps_to_process} ({type(time_steps_to_process)})')
        print(f'grouping_to_process: {grouping_to_process} ({type(grouping_to_process)})')
        print(f'product_type: {product_type} ({type(product_type)})')
        print(f'output_freq_code: {output_freq_code} ({type(output_freq_code)})')

        generate_netcdfs(output_freq_code,
                            product_type,
                            mapping_factors_dir,
                            output_dir_base,
                            diags_root,
                            metadata_json_dir,
                            podaac_dir,
                            ecco_grid_dir,
                            ecco_grid_dir_mds,
                            ecco_grid_filename,
                            grouping_to_process,
                            time_steps_to_process,
                            array_precision,
                            debug_mode)

    # **********
    # TODO: Check output S3 bucket for data
    # **********