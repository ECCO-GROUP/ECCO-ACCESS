"""
Created May 18, 2022

Author: Duncan Bark

"""

import sys
import json
import boto3
import argparse
import platform
import subprocess
from pathlib import Path

sys.path.append(f'{Path(__file__).parent.resolve()}')
from eccov4r4_gen_for_podaac_cloud import generate_netcdfs
from upload_S3 import upload_S3
from gen_netcdf_utils import create_all_factors
import ecco_cloud_utils as ea


# Get credentials for AWS from "~/.aws/credentials" file
def get_credentials():
    cred_path = Path.home() / '.aws/credentials'
    credentials = {}
    if cred_path.exists():
        with open(cred_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                elif line[0] == '#':
                    credentials['expiration_date'] = line.split(' = ')[-1]
                elif line[0] == '[':
                    credentials['profile_name'] = line[1:-1]
                else:
                    name, value = line.split(' = ')
                    credentials[name] = value
    return credentials


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--upload_model_to_S3', default=False, action='store_true',
                        help='Upload model output data to provided directory (or S3 bucket) in config file')

    parser.add_argument('--process_data', default=False, action='store_true',
                        help='Starts processing model data using config file values')

    parser.add_argument('--use_cloud', default=False, action='store_true',
                        help='Process data using AWS cloud services')

    parser.add_argument('--debug', default=False, action='store_true',
                        help='Sets debug flag (additional print outs and skips processing')
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    dict_key_args = {key: value for key, value in args._get_kwargs()} 

    upload_to_S3 = dict_key_args['upload_model_to_S3']
    process_data = dict_key_args['process_data']
    debug_mode = dict_key_args['debug']
    local = not dict_key_args['use_cloud']

    # Testing/setup paths and config -------------------------------------
    # path_to_yaml = Path(__file__).parent.resolve() / 'configs' / 'gen_netcdf_config.yaml'
    # with open(path_to_yaml, "r") as f:
    #     config = yaml.load(f, yaml.Loader)
    # local = config['local']

    # Load 'prodcut_generation_config.json'
    config_json = json.load(open(Path(__file__).parent.resolve() / 'configs' / 'product_generation_config.json'))
    config_metadata = {}
    for entry in config_json:
        config_metadata[entry['name']] = entry['value']


    # Load directories (local vs AWS)
    # Default directories
    parent_dir = Path(__file__).parent.resolve()
    mapping_factors_dir_default = parent_dir / 'mapping_factors'
    diags_root_default = parent_dir / 'diags_all'
    metadata_default = parent_dir / 'metadata' / 'ECCov4r4_metadata_json'
    podaac_metadata_filename_default = 'PODAAC_datasets-revised_20210226.5.csv'
    ecco_grid_dir_default = parent_dir / 'ecco_grids'
    ecco_grid_dir_mds_default = parent_dir / 'ecco_grids'
    ecco_grid_filename_default = 'GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'
    output_dir_base_default = parent_dir / 'temp_output'

    # AWS defaults
    aws_profile_name_default = 'saml-pub'
    aws_region_default = 'us-west-2'
    model_granule_bucket_default = 'ecco-model-granules'
    processed_data_bucket_default = 'ecco-processed-data'


    if config_metadata['mapping_factors_dir'] == '':
        config_metadata['mapping_factors_dir'] = Path(mapping_factors_dir_default)
    if config_metadata['model_data_dir'] == '':
        config_metadata['model_data_dir'] = Path(diags_root_default)
    if config_metadata['metadata_dir'] == '':
        config_metadata['metadata_dir'] = Path(metadata_default)
    if config_metadata['podaac_metadata_filename'] == '':
        config_metadata['podaac_metadata_filename'] = podaac_metadata_filename_default
    if config_metadata['ecco_grid_dir'] == '':
        config_metadata['ecco_grid_dir'] = Path(ecco_grid_dir_default)
    if config_metadata['ecco_grid_dir_mds'] == '':
        config_metadata['ecco_grid_dir_mds'] = Path(ecco_grid_dir_mds_default)
    if config_metadata['ecco_grid_filename'] == '':
        config_metadata['ecco_grid_filename'] = ecco_grid_filename_default
    if config_metadata['output_dir_base'] == '':
        config_metadata['output_dir_base'] = Path(output_dir_base_default)

    if config_metadata['aws_profile_name'] == '':
        config_metadata['aws_profile_name'] = aws_profile_name_default
    if config_metadata['aws_region'] == '':
        config_metadata['aws_region'] = aws_region_default
    if config_metadata['source_bucket'] == '':
        config_metadata['source_bucket'] = model_granule_bucket_default
    if config_metadata['output_bucket'] == '':
        config_metadata['output_bucket'] = processed_data_bucket_default

    # Get all configurations
    all_jobs = []
    with open(f'{Path(__file__).parent.resolve() / "configs" / "jobs.txt"}', 'r') as j:
        for line in j:
            if '#' in line:
                continue
            line_vals = line.strip().split(',')
            all_jobs.append([int(line_vals[0]), line_vals[1], line_vals[2], line_vals[3]])


    # Verify AWS access keys
    credentials = {}
    if not local:

        if 'linux' in platform.platform().lower():
            aws_login_file = './aws-login.linux.amd64'
        else:
            aws_login_file = './aws-login.darwin.amd64'

        # Get current credentials
        credentials = get_credentials()

        # Verify credentials
        try:
            if credentials != {}:
                boto3.setup_default_session(profile_name=credentials['profile_name'])
                try:
                    boto3.client('s3').list_buckets()
                except:
                    subprocess.run([aws_login_file, '-r', f'{config_metadata["aws_region"]}'])
                    credentials = get_credentials()
            else:
                subprocess.run([aws_login_file, '-r', f'{config_metadata["aws_region"]}'])
                credentials = get_credentials()
        except Exception as e:
            print(f'Unable to login to AWS. Exiting')
            print(e)
            sys.exit()

        # Upload data to S3 bucket
        if upload_to_S3:
            status = upload_S3(config_metadata['model_data_dir'], config_metadata['source_bucket'], credentials)
            if not status:
                print(f'Uploading to S3 failed. Exiting')
                sys.exit()

    # Creates mapping_factors (2D and 3D), landmask, and latlon_grid files
    create_all_factors(ea, config_metadata, ['2D', '3D'], debug_mode=debug_mode)

    # loop through all jobs
    # this is where each lambda job would be created
    if process_data:
        for (grouping_to_process, product_type, output_freq_code, time_steps_to_process) in all_jobs:

            # **********
            # TODO: CREATE LAMBDA REQUEST FOR EACH "JOB" (END GAME TODO)
            # **********

            # **********
            # TODO: Verify data in S3 bucket for run
            # **********

            # **********
            # TODO: Have code access S3 bucket for data instead of local directories
            # **********

            print(f'time_steps_to_process: {time_steps_to_process} ({type(time_steps_to_process)})')
            print(f'grouping_to_process: {grouping_to_process} ({type(grouping_to_process)})')
            print(f'product_type: {product_type} ({type(product_type)})')
            print(f'output_freq_code: {output_freq_code} ({type(output_freq_code)})')

            generate_netcdfs(output_freq_code,
                                product_type,
                                grouping_to_process,
                                time_steps_to_process,
                                config_metadata,
                                debug_mode,
                                local,
                                credentials)


        # **********
        # TODO: Check output S3 bucket for data
        # **********