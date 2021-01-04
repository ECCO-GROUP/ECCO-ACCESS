import boto3
# conda env also need: scipy, netCDF4
from grid_transformation import run_using_aws_wrapper


def lambda_handler(event, context):
    # "Process upload event"
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    filename = event['Records'][0]["s3"]["object"]["key"]

    print("Received event. Bucket: [%s], Key: [%s]" % (bucket, filename))

    # Query for grids in solr
    # for grid in grids:
    #   trigger lambda with key as filename$grid
    #          split filename$grid by $ to get filename and grid (or do the split within run_using_aws)
    #          that lambda will call run_using_aws(s3, filename, grid)

    # construct s3 resource instead of client
    s3 = boto3.resource('s3')

    print("===starting AWS session DONE===")

    run_using_aws_wrapper(s3, filename)
