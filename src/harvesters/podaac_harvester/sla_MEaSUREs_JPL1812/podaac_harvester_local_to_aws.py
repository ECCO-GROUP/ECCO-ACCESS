import os
import sys
import gzip
import yaml
import boto3
import shutil
import urllib
from xml.etree.ElementTree import parse
from datetime import datetime, timedelta

from podaac_harvester import podaac_harvester


def lambda_handler(path_to_file_dir):
    print("=====starting AWS session======")
    ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
    SECRET_KEY = os.environ['AWS_SECRET_KEY']

    session = boto3.Session(aws_access_key_id=ACCESS_KEY,
                            aws_secret_access_key=SECRET_KEY)

    # construct s3 resource instead of client
    s3 = session.resource('s3')

    print("===starting AWS session DONE===")

    podaac_harvester(path=path_to_file_dir, s3=s3, on_aws=True)


if __name__ == '__main__':
    path_to_file_dir = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'
    lambda_handler(path_to_file_dir)
