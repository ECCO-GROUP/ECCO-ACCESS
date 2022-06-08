"""
Created May 18, 2022

Author: Duncan Bark

"""

import sys
import json
import boto3
import argparse
import platform
from pathlib import Path

from matplotlib.pyplot import get

sys.path.append(f'{Path(__file__).parent.resolve()}')
# from eccov4r4_gen_for_podaac_cloud import generate_netcdfs
from aws_helpers import get_credentials_helper, upload_S3, create_lambda_function, get_aws_credentials
from gen_netcdf_utils import create_all_factors
import ecco_cloud_utils as ea


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

    parser.add_argument('--force_reconfigure', default=False, action='store_true',
                        help='Force code to re-run code to get AWS credentials')
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
    force_reconfigure = dict_key_args['force_reconfigure']

    # Testing/setup paths and config -------------------------------------
    # path_to_yaml = Path(__file__).parent.resolve() / 'configs' / 'gen_netcdf_config.yaml'
    # with open(path_to_yaml, "r") as f:
    #     config = yaml.load(f, yaml.Loader)
    # local = config['local']

    # Load 'product_generation_config.json'
    config_json = json.load(open(Path(__file__).parent.resolve() / 'configs' / 'product_generation_config.json'))
    config_metadata = {}
    for entry in config_json:
        config_metadata[entry['name']] = entry['value']


    # Load directories (local vs AWS)
    # Default directories
    parent_dir = Path(__file__).parent.resolve()
    mapping_factors_dir_default = str(parent_dir / 'mapping_factors')
    diags_root_default = str(parent_dir / 'diags_all')
    metadata_default = str(parent_dir / 'metadata' / 'ECCov4r4_metadata_json')
    podaac_metadata_filename_default = 'PODAAC_datasets-revised_20210226.5.csv'
    ecco_grid_dir_default = str(parent_dir / 'ecco_grids')
    ecco_grid_dir_mds_default = str(parent_dir / 'ecco_grids')
    ecco_grid_filename_default = 'GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'
    output_dir_base_default = str(parent_dir / 'temp_output')

    if config_metadata['mapping_factors_dir'] == '':
        config_metadata['mapping_factors_dir'] = str(Path(mapping_factors_dir_default))
    if config_metadata['model_data_dir'] == '':
        config_metadata['model_data_dir'] = str(Path(diags_root_default))
    if config_metadata['metadata_dir'] == '':
        config_metadata['metadata_dir'] = str(Path(metadata_default))
    if config_metadata['podaac_metadata_filename'] == '':
        config_metadata['podaac_metadata_filename'] = podaac_metadata_filename_default
    if config_metadata['ecco_grid_dir'] == '':
        config_metadata['ecco_grid_dir'] = str(Path(ecco_grid_dir_default))
    if config_metadata['ecco_grid_dir_mds'] == '':
        config_metadata['ecco_grid_dir_mds'] = str(Path(ecco_grid_dir_mds_default))
    if config_metadata['ecco_grid_filename'] == '':
        config_metadata['ecco_grid_filename'] = ecco_grid_filename_default
    if config_metadata['output_dir_base'] == '':
        config_metadata['output_dir_base'] = str(Path(output_dir_base_default))

    # Creates mapping_factors (2D and 3D), landmask, and latlon_grid files
    create_all_factors(ea, config_metadata, ['2D', '3D'], debug_mode=debug_mode)

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
        # Load 'aws_config.json'
        aws_config_json = json.load(open(Path(__file__).parent.resolve() / 'configs' / 'aws_config.json'))
        aws_config_metadata = {}
        for entry in aws_config_json:
            aws_config_metadata[entry['name']] = entry['value']

        # AWS defaults
        aws_profile_name_default = 'saml-pub'
        aws_region_default = 'us-west-2'
        model_granule_bucket_default = 'ecco-model-granules'
        processed_data_bucket_default = 'ecco-processed-data'

        if aws_config_metadata['profile_name'] == '':
            aws_config_metadata['profile_name'] = aws_profile_name_default
        if aws_config_metadata['region'] == '':
            aws_config_metadata['region'] = aws_region_default
        if aws_config_metadata['source_bucket'] == '':
            aws_config_metadata['source_bucket'] = model_granule_bucket_default
        if aws_config_metadata['output_bucket'] == '':
            aws_config_metadata['output_bucket'] = processed_data_bucket_default
        
        function_name = aws_config_metadata['function_name']
        image_uri = aws_config_metadata['image_uri']
        role = aws_config_metadata['role']
        account_id = aws_config_metadata['account_id']
        region = aws_config_metadata['region']
        memory_size = aws_config_metadata['memory_size']

        if 'linux' in platform.platform().lower():
            aws_login_file = './aws-login.linux.amd64'
        else:
            aws_login_file = 'aws-login.darwin.amd64'

        # Verify credentials
        credentials = get_credentials_helper()
        try:
            if force_reconfigure:
                # Getting new credentials
                credentials = get_aws_credentials(aws_login_file, region)
            elif credentials != {}:
                boto3.setup_default_session(profile_name=credentials['profile_name'])
                try:
                    boto3.client('s3').list_buckets()
                except:
                    # Present credentials are invalid, try to get new ones
                    credentials = get_aws_credentials(aws_login_file, region)
            else:
                # No credentials present, try to get new ones
                credentials = get_aws_credentials(aws_login_file, region)
        except Exception as e:
            print(f'Unable to login to AWS. Exiting')
            print(e)
            sys.exit()

        # Create AWS session, and lambda client
        boto3.setup_default_session(profile_name=credentials['profile_name'])
        lambda_client = boto3.client('lambda')

        # Create arn
        prefix = 'aws'
        arn = f'arn:{prefix}:iam::{account_id}:role/{role}'

        # Upload data to S3 bucket
        if upload_to_S3:
            status = upload_S3(config_metadata['model_data_dir'], aws_config_metadata['source_bucket'], credentials)
            if not status:
                print(f'Uploading to S3 failed. Exiting')
                sys.exit()

        # create lambda function
        create_lambda_function(lambda_client, function_name, arn, memory_size, image_uri)

    # loop through all jobs
    # this is where each lambda job is created and executed
    if process_data:
        for (grouping_to_process, product_type, output_freq_code, time_steps_to_process) in all_jobs:

            # **********
            # TODO: Verify data in S3 bucket for current lambda job
            # **********

            # **********
            # CREATE LAMBDA REQUEST FOR EACH "JOB"
            # **********
            # create payload for current lambda job
            payload = {
                'grouping_to_process': grouping_to_process,
                'product_type': product_type,
                'output_freq_code': output_freq_code,
                'time_steps_to_process': time_steps_to_process,
                'config_metadata': config_metadata,
                'aws_metadata': aws_config_metadata,
                'debug_mode': debug_mode,
                'local': local,
                'credentials': credentials
            }

            # invoke lambda job
            try:
                invoke_response = lambda_client.invoke(
                    FunctionName=function_name,
                    InvocationType='Event',
                    Payload=json.dumps(payload)
                )
            except Exception as e:
                print(f'Lambda invoke error: {e}')

            import pdb; pdb.set_trace()

            # print(f'time_steps_to_process: {time_steps_to_process} ({type(time_steps_to_process)})')
            # print(f'grouping_to_process: {grouping_to_process} ({type(grouping_to_process)})')
            # print(f'product_type: {product_type} ({type(product_type)})')
            # print(f'output_freq_code: {output_freq_code} ({type(output_freq_code)})')

            # generate_netcdfs(output_freq_code,
            #                     product_type,
            #                     grouping_to_process,
            #                     time_steps_to_process,
            #                     config_metadata,
            #                     aws_metadata,
            #                     debug_mode,
            #                     local,
            #                     credentials)


        # Delete lambda function
        lambda_client.delete_function(FunctionName=function_name)

        # **********
        # TODO: Check output S3 bucket for data
        # **********