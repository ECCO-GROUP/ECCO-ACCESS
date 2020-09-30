import boto3
# conda env also need: scipy, netCDF4
from aggregation import run_aggregation


def lambda_handler(event, context):
    # "Process upload event"
    # bucket = event['Records'][0]["s3"]["bucket"]["name"]
    # filename = event['Records'][0]["s3"]["object"]["key"]
    #
    # print("Received event. Bucket: [%s], Key: [%s]" % (bucket, filename))

    # construct s3 resource instead of client
    s3 = boto3.resource('s3')

    print("===starting AWS session DONE===")

    run_aggregation('', s3=s3)
