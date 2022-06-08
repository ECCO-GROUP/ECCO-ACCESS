import sys
import glob
import time
import boto3
import subprocess
from pathlib import Path


def upload_S3(source_path, bucket, credentials, check_list=True):
    # Upload provided file to the provided bucket via the provided credentials.

    # Setup S3 bucket client via boto3
    boto3.setup_default_session(profile_name=credentials['profile_name'])
    s3 = boto3.client('s3')

    # Collect list of files within source_path
    data_files = glob.glob(f'{source_path}/**/*.data', recursive=True)
    num_files = len(data_files)

    # Collect files currently on the S3 bucket
    # If, when uploading, the name exists in this list, skip it.
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


















