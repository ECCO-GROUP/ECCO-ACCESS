# ecco-preprocessing

## Local development
#### Download and install Solr:
1. http://yonik.com/solr-tutorial/
2. Start Solr and create a collection named ecco_datasets

#### Update the following config YAML files with local file paths:
1. /src/grids/grids_config.yaml
2. /src/harvesters/podaac_harvester/podaac_harvester_config.yaml
3. /src/preprocessing/sst/grid_transformation/grid_transformation_config.yaml
4. /src/preprocessing/sst/aggregation_by_year/aggregation_config.yaml

#### Initial order of files to run:
1. /src/grids/grids_to_solr.py
2. /src/harvesters/podaac_harvester/podaac_harvester_local.py
3. /src/preprocessing/sst/grid_transformation/grid_transformation_local.py
4. /src/preprocessing/sst/aggregation_by_year/aggregation_local.py

___
Harvest data files and generated output files and/or directories are not tracked on this repo.
