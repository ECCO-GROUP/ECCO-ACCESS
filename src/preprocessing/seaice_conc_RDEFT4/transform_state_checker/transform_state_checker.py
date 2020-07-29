import requests
import yaml
import os
import sys
import time


def get_solr_results():
    system_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    path_to_yaml = system_path + "/state_checker_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    dataset = config['dataset']
    solr_host = config['solr_host']
    solr_collection_name = config['solr_collection_name']

    # get number of harvested files for a dataset
    getVars = {'q': '*:*',
               'fq': ['dataset_s:'+dataset, 'type_s:harvested'],
               'rows': 300000}
    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    harvested_count = response.json()['response']['numFound']

    # get number of fields for dataset
    getVars = {'q': '*:*',
               'fq': ['dataset_s:'+dataset, 'type_s:dataset'],
               'fl': ['fields_s'],
               'rows': 300000}
    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    fields = response.json()['response']['docs'][0]['fields_s'].split(", ")
    fields_count = len(fields)

    # get number of grids
    getVars = {'q': '*:*',
               'fq': ['type_s:grid'],
               'rows': 300000}
    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    grid_count = response.json()['response']['numFound']

    expected_transformation_count = harvested_count * fields_count * grid_count

    # get number of transformations no longer in progress
    getVars = {'q': '*:*',
               'fq': ['dataset_s:'+dataset, 'type_s:transformation', 'transformation_in_progress_b:false'],
               'rows': 300000}
    url = solr_host + solr_collection_name + '/select?'
    response = requests.get(url, params=getVars)
    transformation_count = response.json()['response']['numFound']

    return expected_transformation_count - transformation_count


def local_handler():
    remaining_transformations = get_solr_results()

    while remaining_transformations != 0:
        print(str(remaining_transformations) + " remaining transformations")
        print("not time yet!!")
        time.sleep(3)
        remaining_transformations = get_solr_results()

    print("Time to run aggregation!")


local_handler()
