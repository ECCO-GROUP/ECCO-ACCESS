# for local run
from boto3.session import Session
import sys
import os

from grid_transformation import run_using_aws_wrapper


def lambda_handler(filename):
    # --------------
    # DEV/AWS TODO: replace this section with s3 trigger when put into lambda
    # --------------
    print("=====starting AWS session======")
    ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
    SECRET_KEY = os.environ['AWS_SECRET_KEY']

    session = Session(aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    # construct s3 resource
    s3 = session.resource('s3')

    print("===starting AWS session DONE===")

    run_using_aws_wrapper(s3, filename)


if __name__ == '__main__':
    filename = sys.argv[1]
    print(filename)
    lambda_handler(filename)
