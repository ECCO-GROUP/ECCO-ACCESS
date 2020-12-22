import os
import sys
from boto3.session import Session
from aggregation import run_aggregation

# ### For Reference: Python binary file i/o specifications
# ```
# endianness:
#     big     : '>'
#     little  : '<'
#
# precision
#     float32':'f4',
#     float64':'f8',
# ```

# Arguments are <path to folder> <year>


def lambda_handler(system_path):

    print("=====starting AWS session======")
    ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
    SECRET_KEY = os.environ['AWS_SECRET_KEY']

    session = Session(aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    s3 = session.resource('s3')
    print("===starting AWS session DONE===")

    run_aggregation(system_path, s3=s3)


if __name__ == '__main__':
    system_path = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'

    lambda_handler(system_path)
