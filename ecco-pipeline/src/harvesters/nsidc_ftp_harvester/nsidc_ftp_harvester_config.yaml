# =====================================================
# Seaice FTP
# =====================================================
start: "" # yyyymmddThh:mm:ssZ
end: "" # yyyymmddThh:mm:ssZ
user: anonymous # does not change
host: sidads.colorado.edu # does not change
regex: '\d{8}'
date_regex: "%Y-%m-%dT%H:%M:%SZ" # does not change

# =====================================================
# Dataset
# =====================================================
ds_name: "" # Name for dataset
ddir: "" #ex: pub/DATASETS/NOAA/G10016/
aggregated: false # if data is available aggregated
short_name: ""
data_time_scale: "" # daily or monthly
date_format: "" # format of date in file name ex: yyyymmdd
regions: [] # regions data is split into ex: ['north', 'south']
fields: [{ name: "", long_name: "", standard_name: "", units: "" }]

# new_data_attrs:
original_dataset_title: ""
original_dataset_short_name: ""
original_dataset_url: ""
original_dataset_reference: ""
original_dataset_doi: ""

# =====================================================
# Solr
# =====================================================
solr_host_local: http://localhost:8983/solr/ # doesn't change if following standard Solr setup
solr_host_aws: http://ec2-3-16-187-19.us-east-2.compute.amazonaws.com:8983/solr/
solr_collection_name: ecco_datasets # doesn't change

# =====================================================
# AWS
# =====================================================
target_bucket_name: ecco-preprocess
