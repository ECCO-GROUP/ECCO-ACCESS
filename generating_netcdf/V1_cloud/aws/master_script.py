"""
Created May 18, 2022

Author: Duncan Bark

"""

from email.policy import default
import os
import sys
import copy
import json
import time
import boto3
import argparse
import platform
from pathlib import Path
from collections import defaultdict

from numpy import product

sys.path.append(f'{Path(__file__).parent.resolve()}')
from aws_helpers import get_credentials_helper, upload_S3, create_lambda_function, get_aws_credentials, save_logs, get_logs
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

    parser.add_argument('--use_lambda', default=False, action='store_true',
                        help='Completes processing via AWS lambda')

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
    use_lambda = dict_key_args['use_lambda']
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
    # Not needed unless changes have been made to the factors code and you need
    # to update the factors/mask in the lambda docker image
    # create_all_factors(ea, config_metadata, ['2D', '3D'], debug_mode=debug_mode)

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

        source_bucket = aws_config_metadata['source_bucket']
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

        # Create arn
        prefix = 'aws'
        arn = f'arn:{prefix}:iam::{account_id}:role/{role}'

        # Setup AWS session and S3 client
        boto3.setup_default_session(profile_name=credentials['profile_name'])
        s3 = boto3.client('s3')

        # Upload data to S3 bucket
        if upload_to_S3:
            status = upload_S3(s3, config_metadata['model_data_dir'], aws_config_metadata['source_bucket'])
            if not status:
                print(f'Uploading to S3 failed. Exiting')
                sys.exit()

        # setup AWS Lambda
        if use_lambda:
            # Create lambda client
            lambda_client = boto3.client('lambda')

            # create lambda function
            create_lambda_function(lambda_client, function_name, arn, memory_size, image_uri)
            lambda_start_time = time.strftime('%Y%m%d:%H%M%S', time.localtime())

            # values for cost estimation
            ms_to_sec = 0.001
            MB_to_GB = 0.0009765625
            USD_per_GBsec = 0.0000166667

            # job information
            num_jobs = 0
            job_logs = {}
            job_logs['Total Time (s)'] = 0
            job_logs['Cost Information'] = defaultdict(float)
    
            start_time = int(time.time()/ms_to_sec)

    # Get grouping information
    metadata_fields = ['ECCOv4r4_groupings_for_1D_datasets',
        'ECCOv4r4_groupings_for_latlon_datasets',
        'ECCOv4r4_groupings_for_native_datasets']

    # load METADATA
    metadata = {}

    for mf in metadata_fields:
        mf_e = mf + '.json'
        with open(str(Path(config_metadata['metadata_dir']) / mf_e), 'r') as fp:
            metadata[mf] = json.load(fp)

    groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
    groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
    groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

    # loop through all jobs and either process them locally
    # or invoke the created lambda function
    if process_data:
        for (grouping_to_process, product_type, output_freq_code, time_steps_to_process) in all_jobs:

            # Reorganize model output in S3 to be structured by grouping, grid, and frequency
            # i.e. ecco_group_0_avg_mon_latlon, ecco_group_11_snap, etc.
            # buckets_list = s3.list_buckets()

            if product_type == 'latlon':
                curr_grouping = groupings_for_latlon_datasets[grouping_to_process]
            elif product_type == 'native':
                curr_grouping = groupings_for_native_datasets[grouping_to_process]

            fields = curr_grouping['fields'].split(', ')
            dimension = curr_grouping['dimension']
            filename = curr_grouping['filename']

            if output_freq_code == 'AVG_DAY':
                s3_dir_prefix = 'diags_all/diags_daily'
                period_suffix = 'day_mean'

            elif output_freq_code == 'AVG_MON':
                s3_dir_prefix = 'diags_all/diags_monthly'
                period_suffix = 'mon_mean'

            elif output_freq_code == 'SNAPSHOT':
                s3_dir_prefix = 'diags_all/diags_inst'
                period_suffix = 'day_inst'
            else:
                print('valid options are AVG_DAY, AVG_MON, SNAPSHOT')
                print('you provided ', output_freq_code)
                sys.exit()        

            s3_field_paths = []
            for field in fields:
                s3_field_paths.append(f'{s3_dir_prefix}/{field}_{period_suffix}')

            field_files = defaultdict(list)
            field_time_steps = defaultdict(list)
            for s3_field_path in s3_field_paths:
                start_after = ''
                field = s3_field_path.split('/')[-1]
                while True:
                    source_objects = s3.list_objects_v2(Bucket=source_bucket, Prefix=s3_field_path, StartAfter=start_after)

                    if 'Contents' not in source_objects:
                        break

                    data_keys = [key['Key'] for key in source_objects['Contents'] if '.data' in key['Key']]
                    field_files[field].extend(data_keys)
                    start_after = data_keys[-1]

                    time_steps = [key.split('.')[-2] for key in data_keys]
                    field_time_steps[field].extend(time_steps)

                    if product_type == 'native':
                        meta_keys = [key['Key'] for key in source_objects['Contents'] if '.meta' in key['Key']]
                        field_files[field].extend(meta_keys)
                        start_after = meta_keys[-1]

            # group number of time steps and files to process based on time to execute
            
            import pdb; pdb.set_trace()


            # **********
            # CREATE LAMBDA REQUEST FOR EACH "JOB"
            # **********
            if use_lambda:
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

                data_to_process= {
                    'grouping_to_process': grouping_to_process,
                    'product_type': product_type,
                    'output_freq_code': output_freq_code,
                    'time_steps_to_process': time_steps_to_process
                }

                # invoke lambda job
                try:
                    if use_lambda:
                        invoke_response = lambda_client.invoke(
                            FunctionName=function_name,
                            InvocationType='Event',
                            Payload=json.dumps(payload),   
                        )

                        job_logs[invoke_response['ResponseMetadata']['RequestId'].strip()] = {
                            'date':invoke_response['ResponseMetadata']['HTTPHeaders']['date'], 
                            'status': invoke_response['StatusCode'], 
                            'data': data_to_process, 
                            'report': [], 
                            'error': [],
                            'end': False
                        }
                
                        num_jobs += 1
            
                except Exception as e:
                    print(f'Lambda invoke error: {e}')

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
        
        if use_lambda:
            log_client = boto3.client('logs')
            log_group_name = '/aws/lambda/ecco_processing'
            ended_log_stream_names = []
            num_jobs_ended = 0
            log_save_time = time.time()
            estimated_jobs = []
            last_job_logs = copy.deepcopy(job_logs)
            end_jobs_list = []
            ctr = -1
            try:
                while True:
                    # intital log
                    if ctr == -1:
                        ctr += 1
                        total_time = (int(time.time()/ms_to_sec)-start_time) * ms_to_sec
                        job_logs['Total Time (s)'] = total_time
                        job_logs, estimated_jobs = save_logs(job_logs, MB_to_GB, estimated_jobs, lambda_start_time, ctr, fn_extra='INITIAL')

                    print(f'Processing job logs -- {num_jobs_ended}/{num_jobs}')
                    time.sleep(2)
                    end_time = int(time.time()/ms_to_sec)
                    log_stream_names = []
                    
                    # TODO: loop through ended_log_stream_names and delete those that did not have an error

                    log_streams = get_logs(log_client, log_group_name, log_stream_names, type='logStream')
                    for ls in log_streams:
                        if ls['logStreamName'] not in ended_log_stream_names:
                            log_stream_names.append(ls['logStreamName'])

                    if log_stream_names != []:
                        # get logs for SUCCESS, DURATION, FILES, ERROR, REPORT, START, and END
                        success_jobs = []
                        extra_logs = {}
                        error_logs = defaultdict(list)
                        report_logs = defaultdict(list)
                        job_id_report_name = {}
                        start_logs = defaultdict(int)
                        end_logs = defaultdict(int)
                        key_logs = get_logs(log_client, log_group_name, log_stream_names, start_time=start_time, end_time=end_time, filter_pattern='?SUCCESS ?DURATION ?FILES ?ERROR ?REPORT ?START ?END', type='event')

                        # find start logs, count number for each ID
                        # Job is only considered ended if it has the same number of start logs and end logs to a maximum of 3
                        for log in key_logs:
                            if 'START' in log['message']:
                                job_id = log['message'].split(' ')[2].strip()
                                start_logs[job_id] += 1

                        for log in key_logs:
                            logStreamName = log['logStreamName']

                            # get SUCCESS logs
                            if 'SUCCESS' in log['message']:
                                if logStreamName not in success_jobs:
                                    success_jobs.append(logStreamName)

                            # get TIME logs
                            if 'DURATION' in log['message']:
                                if logStreamName not in extra_logs.keys():
                                    extra_logs[logStreamName] = {}
                                if 'Duration (s)' not in extra_logs[logStreamName].keys():
                                    extra_logs[logStreamName]['Duration (s)'] = defaultdict(float)
                                _, duration_type, duration, _ = log['message'].split('\t')
                                extra_logs[logStreamName]['Duration (s)']['TOTAL'] += float(duration)
                                extra_logs[logStreamName]['Duration (s)'][duration_type] += float(duration)
                            
                            # get COUNT logs
                            if 'FILES' in log['message']:
                                if logStreamName not in extra_logs.keys():
                                    extra_logs[logStreamName] = {}
                                if 'Files (#)' not in extra_logs[logStreamName].keys():
                                    extra_logs[logStreamName]['Files (#)'] = defaultdict(int)
                                _, file_type, file_count = log['message'].split('\t')
                                extra_logs[logStreamName]['Files (#)'][file_type] += int(file_count)

                            # get ERROR logs
                            if 'ERROR' in log['message']:
                                error_logs[logStreamName].append(log['message'])

                            # get REPORT logs
                            if 'REPORT' in log['message']:
                                report_job_id = ''
                                report = {'logStreamName':logStreamName}
                                report_message = log['message'].split('\t')[:-1]
                                for rm in report_message:
                                    if 'REPORT' in rm:
                                        rm = rm[7:]
                                    rm = rm.split(': ')
                                    if ' ms' in rm[-1]:
                                        rm[-1] = float(rm[-1].replace(' ms', '').strip()) * ms_to_sec
                                        rm[0] = f'{rm[0].strip()} (s)'
                                    elif ' MB' in rm[-1]:
                                        rm[-1] = int(rm[-1].replace(' MB', '').strip())
                                        rm[0] = f'{rm[0].strip()} (MB)'
                                    elif 'RequestId' in rm[0]:
                                        report_job_id = rm[-1].strip()
                                        continue
                                    report[rm[0]] = rm[-1]

                                # estimate cost
                                request_time = report['Billed Duration (s)']
                                request_memory = report['Memory Size (MB)'] * MB_to_GB
                                cost_estimate = request_memory * request_time * USD_per_GBsec
                                report['Cost Estimate (USD)'] = cost_estimate

                                report_logs[report_job_id].append(report)
                                job_id_report_name[report_job_id] = report

                            # get END logs
                            if 'END' in log['message']:
                                end_job_id = log['message'].split(': ')[-1].strip()
                                end_logs[end_job_id] += 1
                                if (end_job_id.strip() not in end_jobs_list) and (start_logs[end_job_id] == end_logs[end_job_id]):
                                    end_jobs_list.append(end_job_id.strip())

                    for job_id in end_jobs_list:
                        if (job_id in job_logs.keys()) and (not job_logs[job_id]['end']):
                            logStreamName = job_id_report_name[job_id]['logStreamName']
                            num_jobs_ended += 1
                            job_logs[job_id]['report'] = report_logs[job_id]
                            job_logs[job_id]['extra'] = extra_logs[logStreamName]
                            job_logs[job_id]['end'] = True
                            job_logs[job_id]['success'] = logStreamName in success_jobs
                            if job_id in job_id_report_name.keys() and job_id_report_name[job_id]['logStreamName'] in error_logs.keys():
                                job_logs[job_id]['error'] = error_logs[job_id_report_name[job_id]['logStreamName']]

                    # print('pre-ended_log')
                    ended_log_stream_names.extend([job_id_report_name[jid]['logStreamName'] for jid in end_jobs_list if jid in job_id_report_name.keys()])
                    ended_log_stream_names = list(set(ended_log_stream_names))

                    if (num_jobs_ended == num_jobs):
                        ctr += 1
                        print(f'Processing job logs -- {num_jobs_ended}/{num_jobs}')
                        total_time = (int(time.time()/ms_to_sec)-start_time) * ms_to_sec
                        job_logs['Total Time (s)'] = total_time
                        # write final job_log to file
                        job_logs, estimated_jobs = save_logs(job_logs, MB_to_GB, estimated_jobs, lambda_start_time, ctr, fn_extra='FINAL')
                        break

                    # write job_log to file every >~10 seconds
                    if (time.time() - log_save_time >= 10) and (job_logs != last_job_logs):
                        ctr += 1
                        log_save_time = time.time()
                        total_time = (int(time.time()/ms_to_sec)-start_time) * ms_to_sec
                        job_logs['Total Time (s)'] = total_time
                        last_job_logs, estimated_jobs = save_logs(job_logs, MB_to_GB, estimated_jobs, lambda_start_time, ctr)
                        job_logs = copy.deepcopy(last_job_logs)
            except Exception as e:
                print(f'Error processing logs for lambda jobs')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


            # Delete lambda function
            lambda_client.delete_function(FunctionName=function_name)

        # **********
        # TODO: Check output S3 bucket for data
        # **********