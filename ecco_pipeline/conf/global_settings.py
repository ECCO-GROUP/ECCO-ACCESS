import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = Path('/Users/marlis/Developer/ECCO ACCESS/ecco_output')

SOLR_HOST = 'http://localhost:8983/solr/'
SOLR_COLLECTION = 'ecco_datasets'

SOLR_HOST_AWS = 'http://ec2-3-16-187-19.us-east-2.compute.amazonaws.com:8983/solr/'
TARGET_BUCKET_NAME = 'ecco-preprocess'

os.chdir(ROOT_DIR)
