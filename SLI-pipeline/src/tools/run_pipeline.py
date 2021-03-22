import os
import sys
import logging
import argparse
import importlib
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from collections import defaultdict
import requests
import yaml

# Hardcoded output directory path for pipeline files
# Leave blank to be prompted for an output directory
OUTPUT_DIR = '/Users/kevinmarlis/Developer/JPL/sealevel_output/'


def create_parser():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default=False, action='store_true',
                        help='Runs prompt to select pipeline output directory.')

    parser.add_argument('--options_menu', default=False, action='store_true',
                        help='Display option menu to select which steps in the pipeline to run.')

    parser.add_argument('--force_processing', default=False, action='store_true',
                        help='Force reprocessing of any existing aggregated cycles.')

    parser.add_argument('--harvested_entry_validation', default=False,
                        help='verifies each Solr harvester entry points to a valid file.')

    return parser


def verify_solr_running():
    solr_host = 'http://localhost:8983/solr/'
    solr_collection_name = 'sealevel_datasets'

    try:
        requests.get(f'{solr_host}{solr_collection_name}/admin/ping')
        return
    except requests.ConnectionError:
        print('\nSolr not currently running! Please double check and run pipeline again.\n')
        sys.exit()


def harvested_entry_validation():
    """
    """
    solr_host = 'http://localhost:8983/solr/'
    solr_collection_name = 'sealevel_datasets'

    response = requests.get(
        f'{solr_host}{solr_collection_name}/select?fq=type_s%3Aharvested&q=*%3A*')

    if response.status_code == 200:
        docs_to_remove = []
        harvested_docs = response.json()['response']['docs']

        for doc in harvested_docs:
            file_path = doc['pre_transformation_file_path_s']
            if os.path.exists(file_path):
                continue
            docs_to_remove.append(doc['id'])

        url = f'{solr_host}{solr_collection_name}/update?commit=true'
        requests.post(url, json={'delete': docs_to_remove})

        print('Succesfully removed entries from Solr')

    else:
        print('Solr not online or collection does not exist')
        sys.exit()


def show_menu():
    while True:
        print('\n------------- OPTIONS -------------')
        print('1) Harvest and process all datasets')
        print('2) Harvest all datasets')
        print('3) Process all datasets')
        print('4) Dataset input')
        selection = input('Enter option number: ')

        if selection in ['1', '2', '3', '4']:
            return selection
        print(
            f'Unknown option entered, "{selection}", please enter a valid option\n')


def print_log(log_path):
    """
    """

    print('\n=========================================================')
    print('===================== \033[36mPrinting log\033[0m ======================')
    print('=========================================================')

    dataset_statuses = defaultdict(lambda: defaultdict(list))

    # Parse logger for messages
    with open(log_path) as log:
        logs = log.read().splitlines()

    for line in logs:
        log_line = yaml.load(line, yaml.Loader)
        ds = log_line['name'].replace('pipeline.', '').replace(
            '.harvester', '').replace('.processing', '')
        preprocessing_step = log_line['name'].replace('pipeline.', '').replace(f'{ds}.', '')

        if log_line['level'] == 'INFO':
            dataset_statuses[ds][preprocessing_step].append(('INFO', log_line["message"]))

        if log_line['level'] == 'ERROR':
            if ('ERROR', log_line["message"]) not in dataset_statuses[ds][preprocessing_step]:
                dataset_statuses[ds][preprocessing_step].append(('ERROR', log_line["message"]))

    # Print dataset status summaries
    for ds, steps in dataset_statuses.items():
        print(f'\033[93mPipeline status for {ds}\033[0m:')
        for _, messages in steps.items():
            for (_, message) in messages:
                if 'successful' in message:
                    print(f'\t\033[92m{message}\033[0m')
                else:
                    print(f'\t\033[91m{message}\033[0m')


def run_harvester(datasets, path_to_harvesters, output_dir):
    print('\n=========================================================')
    print(
        '================== \033[36mRunning harvesters\033[0m ===================')
    print('=========================================================\n')
    for ds in datasets:
        harv_logger = logging.getLogger(f'pipeline.{ds}.harvester')
        try:
            print(f'\033[93mRunning harvester for {ds}\033[0m')
            print('=========================================================')

            config_path = Path(
                f'{Path(__file__).resolve().parents[2]}/datasets/{ds}/harvester_config.yaml')

            path_to_code = Path(f'{path_to_harvesters}/')

            sys.path.insert(1, str(path_to_code))

            ret_import = importlib.import_module('harvester')
            ret_import = importlib.reload(ret_import)

            ret_import.harvester(config_path=config_path, output_path=output_dir)

            sys.path.remove(str(path_to_code))

            harv_logger.info('Harvesting successful')
            print('\033[92mHarvest successful\033[0m')
        except Exception as e:
            sys.path.remove(str(path_to_code))
            harv_logger.info('Harvesting failed: %s', e)

            print('\033[91mHarvesting failed\033[0m')
        print('=========================================================')


def run_processing(datasets, path_to_processors, output_dir, reprocess):
    print('\n=========================================================')
    print(
        '================= \033[36mRunning processing\033[0m ==================')
    print('=========================================================\n')
    for ds in datasets:
        proc_logger = logging.getLogger(f'pipeline.{ds}.processing')
        try:
            print(f'\033[93mRunning processing for {ds}\033[0m')
            print('=========================================================')
            config_path = Path(
                f'{Path(__file__).resolve().parents[2]}/datasets/{ds}/processing_config.yaml'
            )

            path_to_code = Path(f'{path_to_processors}')

            sys.path.insert(1, str(path_to_code))

            ret_import = importlib.import_module('processing')
            ret_import = importlib.reload(ret_import)

            ret_import.processing(config_path=config_path,
                                  output_path=output_dir,
                                  reprocess=reprocess)

            sys.path.remove(str(path_to_code))

            proc_logger.info('Processing successful')
            print('\033[92mProcessing successful\033[0m')
        except Exception as e:
            print(e)
            sys.path.remove(str(path_to_code))
            proc_logger.info('Processing failed: %s', e)
            print('\033[91mProcessing failed\033[0m')
        print('=========================================================')


if __name__ == '__main__':
    verify_solr_running()
    print('\n=========================================================')
    print('============= SEA LEVEL INDICATORS PIPELINE =============')
    print('=========================================================')

    # path to harvester and preprocessing folders
    pipeline_path = Path(__file__).resolve()

    PATH_TO_HARVESTERS = Path(f'{pipeline_path.parents[1]}/harvesters')
    PATH_TO_PROCESSORS = Path(f'{pipeline_path.parents[1]}/processors')
    PATH_TO_DATASETS = Path(f'{pipeline_path.parents[2]}/datasets')

    PARSER = create_parser()
    args = PARSER.parse_args()

    # -------------- Harvested Entry Validation --------------
    if args.harvested_entry_validation:
        harvested_entry_validation()

    # ------------------- Output directory -------------------
    if args.output_dir or not OUTPUT_DIR:
        print('\nPlease choose your output directory')

        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        OUTPUT_DIR = f'{filedialog.askdirectory()}/'

        if OUTPUT_DIR == '/':
            print('No output directory given. Exiting.')
            sys.exit()
    else:
        if not os.path.exists(OUTPUT_DIR):
            print(f'{OUTPUT_DIR} is an invalid output directory. Exiting.')
            sys.exit()
    print(f'\nUsing output directory: {OUTPUT_DIR}')

    # ------------------ Force Reprocessing ------------------
    REPROCESS = bool(args.force_processing)

    # --------------------- Run pipeline ---------------------

    # Initialize logger
    logger_path = f'{OUTPUT_DIR}/pipeline.log'
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)

    # Setup file handler to output log file
    fh = logging.FileHandler(logger_path, 'w+')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "{'name': '%(name)s', 'level': '%(levelname)s', 'message': '%(message)s'}")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Setup console handler to print to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch_formatter = logging.Formatter("'%(filename)s':'%(lineno)d,  %(message)s'")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    DATASETS = [ds for ds in os.listdir(PATH_TO_DATASETS) if ds != '.DS_Store']

    CHOSEN_OPTION = show_menu() if args.options_menu else '1'

    # Run all
    if CHOSEN_OPTION == '1':
        for dataset in DATASETS:
            run_harvester([dataset], PATH_TO_HARVESTERS, OUTPUT_DIR)
            run_processing([dataset], PATH_TO_PROCESSORS, OUTPUT_DIR, REPROCESS)
        # Run indexing here:
        # run_indexing()

    # Run harvester
    elif CHOSEN_OPTION == '2':
        for dataset in DATASETS:
            run_harvester([dataset], PATH_TO_HARVESTERS, OUTPUT_DIR)

    # Run processing
    elif CHOSEN_OPTION == '3':
        for dataset in DATASETS:
            run_processing([dataset], PATH_TO_PROCESSORS, OUTPUT_DIR, REPROCESS)
        # Run indexing here:
        # run_indexing()

    # Manually enter dataset and pipeline step(s)
    elif CHOSEN_OPTION == '4':
        ds_dict = dict(enumerate(DATASETS, start=1))
        while True:
            print('\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')
            ds_index = input('\nEnter dataset number: ')

            if not ds_index.isdigit() or int(ds_index) not in range(1, len(DATASETS)+1):
                print(f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        CHOSEN_DS = ds_dict[int(ds_index)]
        print(f'\nUsing {CHOSEN_DS} dataset')

        STEPS = ['harvest', 'process', 'all']
        steps_dict = dict(enumerate(STEPS, start=1))

        while True:
            print('\nAvailable steps:\n')
            for i, step in steps_dict.items():
                print(f'{i}) {step}')
            steps_index = input('\nEnter pipeline step(s) number: ')

            if not steps_index.isdigit() or int(steps_index) not in range(1, len(STEPS)+1):
                print(
                    f'Invalid step(s), "{steps_index}", please enter a valid selection')
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if 'harvest' in wanted_steps:
            run_harvester([CHOSEN_DS], PATH_TO_HARVESTERS, OUTPUT_DIR)
        if 'process' in wanted_steps:
            run_processing([CHOSEN_DS], PATH_TO_PROCESSORS, OUTPUT_DIR, REPROCESS)
            # Run indexing here:
            # run_indexing()
        if wanted_steps == 'all':
            run_harvester([CHOSEN_DS], PATH_TO_HARVESTERS, OUTPUT_DIR)
            run_processing([CHOSEN_DS], PATH_TO_PROCESSORS, OUTPUT_DIR, REPROCESS)
            # Run indexing here:
            # run_indexing()

    print_log(logger_path)
