import sys
import glob
import time
import json
import boto3
import subprocess
from pathlib import Path


def get_logs(log_client, log_group_name, log_stream_names, start_time=0, end_time=0, filter_pattern='', type=''):
    if type == 'event':
        events_current = log_client.filter_log_events(logGroupName=log_group_name, logStreamNames=log_stream_names, filterPattern=filter_pattern, startTime=start_time, endTime=end_time)
        ret_logs = events_current['events']
        while True:
            if 'nextToken' in events_current.keys():
                events_current = log_client.filter_log_events(logGroupName=log_group_name, logStreamNames=log_stream_names, filterPattern=filter_pattern, nextToken=events_current['nextToken'])
                if events_current['events'] != []:
                    ret_logs.extend(events_current['events'])
            else:
                break
    elif type == 'logStream':
        log_streams_current = log_client.describe_log_streams(logGroupName=log_group_name, orderBy='LastEventTime')
        ret_logs = log_streams_current['logStreams']
        while True:
            if 'nextToken' in log_streams_current.keys():
                log_streams_current = log_client.describe_log_streams(logGroupName=log_group_name, orderBy='LastEventTime', nextToken=log_streams_current['nextToken'])
                if log_streams_current['logStreams'] != []:
                    ret_logs.extend(log_streams_current['logStreams'])
            else:
                break
    return ret_logs


def save_logs(job_logs, MB_to_GB, estimated_jobs, start_time, ctr, fn_extra=''):
    for job in job_logs.keys():
        if job != 'Cost Information' and job != 'Total Time (s)':
            if job not in estimated_jobs:
                if (fn_extra != 'INITIAL') and (job_logs[job]['end']):
                    estimated_jobs.append(job)
                if job_logs[job]['report'] != []:
                    job_reports = job_logs[job]['report']
                    for job_report in job_reports:
                        request_duration_time = job_report["Duration (s)"]
                        request_time = job_report["Billed Duration (s)"]
                        request_memory = job_report["Memory Size (MB)"]
                        cost_estimate = job_report["Cost Estimate (USD)"]
                        job_logs['Cost Information'][f'{job_report["Memory Size (MB)"]} MB Total Time (s)'] += request_duration_time
                        job_logs['Cost Information'][f'{job_report["Memory Size (MB)"]} MB Total Billed Time (s)'] += request_time
                        job_logs['Cost Information'][f'{job_report["Memory Size (MB)"]} MB Total GB*s'] += (request_memory * MB_to_GB * request_time)
                        job_logs['Cost Information'][f'{job_report["Memory Size (MB)"]} MB Total Cost (USD)'] += cost_estimate
                        job_logs['Cost Information']['Total Cost'] += cost_estimate

    if fn_extra != '' and fn_extra[0] != '_':
        fn_extra = f'{fn_extra}'
    time_str = time.strftime('%Y%m%d:%H%M%S', time.localtime())
    with open(f'./logs/job_logs_{start_time}_{ctr}_{time_str}_{fn_extra}.json', 'w') as f:
        json.dump(job_logs, f, indent=4)
    
    return job_logs, estimated_jobs


def upload_S3(s3, source_path, bucket, check_list=True):
    # Upload provided file to the provided bucket via the provided credentials.

    # Collect list of files within source_path
    data_files = glob.glob(f'{source_path}/**/*.data', recursive=True)
    num_files = len(data_files)

    # Collect files currently on the S3 bucket
    # If, when uploading, the name exists in this list, skip it.
    files_on_s3 = []
    if check_list:
        response = s3.list_objects(Bucket=bucket)
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            print(f'Unable to collect objects in bucket {bucket}')
            return -1
        else:
            files_on_s3 = [k['Key'] for k in response['Contents']]

    # Upload photos from source_path to S3 bucket
    upload = input(f'About to upload {len(data_files)} files, from {source_path}, to bucket {bucket}. Continue? y/n \n')
    if upload.strip().lower() == 'y':
        print('\nUploading files')
        for i, data_file in enumerate(data_files):
            print(f'\t{i+1:7} / {num_files}', end='\r')
            name = f'diags_all/{data_file.split("/diags_all/")[-1]}'
            if name in files_on_s3:
                continue
            try:
                response = s3.upload_file(data_file, bucket, name)
                print(f'Uploaded {data_file} to bucket {bucket}')
            except:
                print(f'Unable to upload file {data_file} to bucket {bucket}')
        print()

    return 1


def create_lambda_function(client, function_name, role, memory_size, image_uri):
    # Create lambda function using the provided values

    # TODO: Check if the function has already been made
    try:
        client.create_function(
            FunctionName=function_name,
            Role=role,
            PackageType='Image',
            Code={'ImageUri':image_uri},
            Publish=True,
            Timeout=900,
            MemorySize=memory_size
        )
    except:
        print('\nFunction already made')

    print(f'\nVerifying lambda function creation ({function_name})...')
    while True:
        status = client.get_function_configuration(FunctionName=function_name)['State']
        if status == "Failed":
            print(f'\tFailed to create function ({function_name}). Try again')
            sys.exit()
        elif status == 'Active':
            print(f'\tFunction created successfully')
            break
        time.sleep(5)
    
    return


# Get credentials for AWS from "~/.aws/credentials" file
def get_credentials_helper():
    cred_path = Path.home() / '.aws/credentials'
    credentials = {}
    if cred_path.exists():
        with open(cred_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    break
                elif line[0] == '#':
                    credentials['expiration_date'] = line.split(' = ')[-1]
                elif line[0] == '[':
                    credentials['profile_name'] = line[1:-1]
                else:
                    name, value = line.split(' = ')
                    credentials[name] = value
    return credentials


def get_aws_credentials(aws_login_file, aws_region):
    try:
        subprocess.run([aws_login_file, '-r', f'{aws_region}'], check=True)
        credentials = get_credentials_helper()
    except:
        print(f'Unable to run script to get credentials ("{aws_login_file}"). Exiting')
        sys.exit()

    return credentials


















