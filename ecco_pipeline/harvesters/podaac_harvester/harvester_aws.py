import os
import sys
import gzip
import yaml
import boto3
import shutil
import urllib
from os import path
from xml.etree.ElementTree import parse
from datetime import datetime, timedelta

from podaac_harvester import podaac_harvester


def lambda_handler(event, context):
    # "Process upload event"
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    key = event['Records'][0]["s3"]["object"]["key"]
    copy_source = {'Bucket': bucket, 'Key': key}

    print("=====starting AWS session======")

    s3 = boto3.resource('s3')

    print("===starting AWS session DONE===")

    podaac_harvester(s3=s3, on_aws=True)
