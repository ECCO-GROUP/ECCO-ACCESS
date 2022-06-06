import glob
import boto3

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