import argparse
import importlib
import logging
import os
import sys
import tkinter as tk
from collections import defaultdict
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from tkinter import filedialog

import numpy as np
import requests
import yaml

# Hardcoded output directory path for pipeline files
# Leave blank to be prompted for an output directory
output_dir = Path('/net/b230-cdot2-svm3/ecco_nfs_1/marlis/pipeline_output')
# output_dir = Path('/Users/marlis/Developer/ECCO ACCESS/ecco_output')

# Verify output_dir
if not output_dir:
    print('Missing output directory. Please fill in.')
    exit()

# Verify solr is running
try:
    requests.get('http://localhost:8983/solr/ecco_datasets/admin/ping')
except requests.ConnectionError:
    print('\nSolr not currently running! Please double check and run pipeline again.\n')
    sys.exit()

CONFIG_PATH = Path(f'{Path(__file__).resolve().parents[2]}/dataset_configs/')
LOG_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

logs_path = Path(output_dir / f'logs/{LOG_TIME}/')
logs_path.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grids_to_solr', default=False, action='store_true',
                        help='updates Solr with grids in grids_config')

    parser.add_argument('--output_dir', default=False, action='store_true',
                        help='runs prompt to select pipeline output directory')

    parser.add_argument('--add_dataset', default=False, action='store_true',
                        help='adds new dataset')

    parser.add_argument('--single_processing', default=False, action='store_true',
                        help='turns off the use of multiprocessing during transformation')

    parser.add_argument('--multiprocesses', type=int, choices=range(1, cpu_count()+1),
                        default=int(cpu_count()/2), metavar=f'[1, {cpu_count()}]',
                        help=f'sets the number of multiprocesses used during transformation with a \
                            system max of {cpu_count()} with default set to half of system max')

    parser.add_argument('--harvested_entry_validation', default=False, nargs='*',
                        help='verifies each Solr harvester entry points to a valid file. if no args given, defaults to \
                            hard coded Solr address. Otherwise takes two args: Solr host url and collection name')

    parser.add_argument('--wipe_transformations', default=False, action='store_true',
                        help='deletes transformations with version number different than what is \
                            currently in transformation_config')

    parser.add_argument('--developer_solr', default=False, nargs='*',
                        help='Uses provided Solr host url and collection name for all Solr entries. if no args given, defaults to \
                            hard coded Solr host url and collection name in configuration files. Otherwise takes two args: Solr host url and collection name')

    parser.add_argument('--grids_to_use', default=False, nargs='*',
                        help='Names of grids to use during the pipeline')

    return parser


def check_solr_grids(solr_host, solr_collection_name):
    response = requests.get(
        f'{solr_host}{solr_collection_name}/select?fq=type_s%3Agrid&q=*%3A*')
    if not response.json()['response']['docs']:
        return True
    else:
        return False


def harvested_entry_validation(args=[]):
    solr_host = 'http://localhost:8983/solr/'
    solr_collection_name = 'ecco_datasets'

    if args:
        solr_host = args[0]
        solr_collection_name = args[1]

    try:
        response = requests.get(
            f'{solr_host}{solr_collection_name}/select?fq=type_s%3Aharvested&q=*%3A*')

        if response.status_code == 200:
            docs_to_remove = []
            harvested_docs = response.json()['response']['docs']

            for doc in harvested_docs:
                file_path = doc['pre_transformation_file_path_s']
                if os.path.exists(file_path):
                    continue
                else:
                    docs_to_remove.append(doc['id'])

            url = f'{solr_host}{solr_collection_name}/update?commit=true'
            requests.post(url, json={'delete': docs_to_remove})

            print('Succesfully removed entries from Solr')

        else:
            print('Solr not online or collection does not exist')
            sys.exit()

    except Exception as e:
        print(e)
        print('Bad Solr URL')
        sys.exit()


def print_log(output_dir):
    print('\n=========================================================')
    print(
        '===================== \033[36mPrinting log\033[0m ======================')
    print('=========================================================')

    log_path = output_dir / f'logs/{LOG_TIME}/pipeline.log'
    dataset_statuses = defaultdict(lambda: defaultdict(list))

    # Parse logger for messages
    with open(log_path) as log:
        logs = log.read().splitlines()

    if logs:
        for line in logs:
            log_line = yaml.load(line, yaml.Loader)

            if 'harvesting' in log_line['message']:
                ds = log_line['message'].split()[0]
                step = 'harvesting'
                msg = log_line['message'].replace(f'{ds} ', '', 1)
                msg = msg[0].upper() + msg[1:]

            elif 'transformation' in log_line['message']:
                ds = log_line['message'].split()[0]
                step = 'cycle creation'
                msg = log_line['message'].replace(f'{ds} ', '', 1)
                msg = msg[0].upper() + msg[1:]

            elif 'aggregation' in log_line['message']:
                ds = 'non dataset specific steps'
                step = 'regridding'
                msg = log_line['message']

            if log_line['level'] == 'INFO':
                dataset_statuses[ds][step] = [('INFO', msg)]

            if log_line['level'] == 'ERROR':
                dataset_statuses[ds][step] = [('ERROR', msg)]

        # Print dataset status summaries
        for ds, steps in dataset_statuses.items():
            print(f'\033[93mPipeline status for {ds}\033[0m:')
            for _, messages in steps.items():
                for (level, message) in messages:
                    if level == 'INFO':
                        print(f'\t\033[92m{message}\033[0m')
                    elif level == 'ERROR':
                        print(f'\t\033[91m{message}\033[0m')

    else:
        print('Manually exited pipeline.')

    # # Must add info level items first
    # for d in log_dict:
    #     ds = d['name'].replace('pipeline.', '').replace('.harvester', '').replace(
    #         '.transformation', '').replace('.aggregation', '')
    #     preprocessing_step = d['name'].replace(
    #         'pipeline.', '').replace(f'{ds}.', '')
    #     if len(ds) > 0:
    #         if d['level'] == 'INFO':
    #             dataset_statuses[ds][preprocessing_step].append(('INFO',
    #                                                              d["message"]))
    # # Then add errors
    # for d in log_dict:
    #     ds = d['name'].replace('pipeline.', '').replace('.harvester', '').replace(
    #         '.transformation', '').replace('.aggregation', '')
    #     preprocessing_step = d['name'].replace(
    #         'pipeline.', '').replace(f'{ds}.', '')
    #     if len(ds) > 0:
    #         if d['level'] == 'ERROR':
    #             if ('ERROR', d["message"]) not in dataset_statuses[ds][preprocessing_step]:
    #                 dataset_statuses[ds][preprocessing_step].append(
    #                     ('ERROR', d["message"]))

    # for ds, steps in dataset_statuses.items():
    #     print(f'\033[93mPipeline status for {ds}\033[0m:')
    #     for step, messages in steps.items():
    #         for (level, message) in messages:
    #             if level == 'INFO':
    #                 if 'successful' in message:
    #                     print(f'\t\033[92m{message}\033[0m')
    #                 else:
    #                     print(f'\t\033[91m{message}\033[0m')
    #             elif level == 'ERROR':
    #                 print(f'\t\t\033[91m{message}\033[0m')


def run_harvester(datasets, path_to_harvesters, output_dir, solr_info, grids_to_use):
    print('\n=========================================================')
    print(
        '================== \033[36mRunning harvesters\033[0m ===================')
    print('=========================================================\n')
    for ds in datasets:
        # harv_logger = logging.getLogger(f'pipeline.{ds}.harvester')
        try:
            print(f'\033[93mRunning harvester for {ds}\033[0m')
            print('=========================================================')

            config_path = Path(CONFIG_PATH / f'{ds}/harvester_config.yaml')

            with open(config_path, 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            if 'harvester_type' in config.keys():
                harvester_type = config['harvester_type']

                if harvester_type not in ['podaac', 'osisaf_ftp', 'nsidc_ftp']:
                    print(f'{harvester_type} is not a supported harvester type.')
                    break

                if 'RDEFT4' in ds:
                    path_to_code = Path(
                        f'{path_to_harvesters}/{harvester_type}_harvester/RDEFT4_ftp_harvester/'
                    )
                else:
                    path_to_code = Path(
                        f'{path_to_harvesters}/{harvester_type}_harvester/')

                sys.path.insert(0, str(path_to_code))

                import harvester_local

                status = harvester_local.main(config,
                                              output_dir,
                                              LOG_TIME,
                                              solr_info,
                                              grids_to_use)

                sys.path.remove(str(path_to_code))

            log.info(f'{ds} harvesting complete. {status}')
            print('\033[92mHarvest successful\033[0m')
        except Exception as e:
            sys.path.remove(str(path_to_code))
            print(e)
            log.error(f'{ds} harvesting failed. {e}')
            print('\033[91mHarvesting failed\033[0m')
        print('=========================================================')


def run_transformation(datasets, path_to_preprocessing, output_dir, multiprocessing, user_cpus, wipe, solr_info, grids_to_use):
    print('\n=========================================================')
    print(
        '=============== \033[36mRunning transformations\033[0m =================')
    print('=========================================================\n')
    for ds in datasets:
        # trans_logger = logging.getLogger(f'pipeline.{ds}.transformation')
        try:
            print(f'\033[93mRunning transformation for {ds}\033[0m')
            print('=========================================================')

            config_path = Path(
                CONFIG_PATH / f'{ds}/transformation_config.yaml')

            with open(config_path, 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            path_to_code = Path(
                f'{path_to_preprocessing}/grid_transformation/')

            sys.path.insert(0, str(path_to_code))

            import grid_transformation_local

            status = grid_transformation_local.main(config,
                                                    output_dir,
                                                    LOG_TIME,
                                                    multiprocessing,
                                                    user_cpus,
                                                    wipe,
                                                    solr_info,
                                                    grids_to_use)

            sys.path.remove(str(path_to_code))

            log.info(f'{ds} transformation complete. {status}')
            print('\033[92mTransformation successful\033[0m')
        except Exception as e:
            sys.path.remove(str(path_to_code))
            log.error(f'{ds} transformation failed. {e}')
            print('\033[91mTransformation failed\033[0m')
        print('=========================================================')


def run_aggregation(datasets, path_to_preprocessing, output_dir, solr_info, grids_to_use):
    print('\n=========================================================')
    print(
        '================ \033[36mRunning aggregations\033[0m ===================')
    print('=========================================================\n')
    for ds in datasets:
        # agg_logger = logging.getLogger(f'pipeline.{ds}.aggregation')
        try:
            print(f'\033[93mRunning aggregation for {ds}\033[0m')
            print('=========================================================')

            config_path = Path(CONFIG_PATH / f'{ds}/aggregation_config.yaml')

            with open(config_path, 'r') as stream:
                config = yaml.load(stream, yaml.Loader)

            path_to_code = Path(
                f'{path_to_preprocessing}/aggregation_by_year/')

            sys.path.insert(0, str(path_to_code))

            import aggregation_local

            status = aggregation_local.main(output_dir,
                                            config,
                                            LOG_TIME,
                                            solr_info,
                                            grids_to_use)

            sys.path.remove(str(path_to_code))

            log.info(f'{ds} aggregation complete. {status}')
            print('\033[92mAggregation successful\033[0m')
        except Exception as e:
            sys.path.remove(str(path_to_code))
            log.info(f'{ds} aggregation failed: {e}')
            print('\033[91mAggregation failed\033[0m')
        print('=========================================================')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    print('\n=================================================')
    print('========== ECCO PREPROCESSING PIPELINE ==========')
    print('=================================================')

    # path to harvester and preprocessing folders
    pipeline_path = Path(__file__).resolve()

    path_to_harvesters = Path(f'{pipeline_path.parents[1]}/harvesters')
    path_to_preprocessing = Path(f'{pipeline_path.parents[1]}/preprocessing')
    path_to_grids = Path(f'{pipeline_path.parents[2]}/grids_to_solr')

    # ------------------- Harvested Entry Validation -------------------
    if isinstance(args.harvested_entry_validation, list) and len(args.harvested_entry_validation) in [0, 2]:
        harvested_entry_validation(args=args.harvested_entry_validation)

    # ------------------- Developer Solr -------------------
    if isinstance(args.developer_solr, list) and len(args.developer_solr) in [0, 2]:
        solr_info = {
            'solr_url': args.developer_solr[0], 'solr_collection_name': args.developer_solr[1]}
    else:
        solr_info = {}

    # ------------------- Grids to Use -------------------
    if isinstance(args.grids_to_use, list):
        grids_to_use = args.grids_to_use
        verify_grids = True
    else:
        grids_to_use = []
        verify_grids = False

    # ------------------- Grids to Solr -------------------
    if solr_info:
        solr_host = solr_info['solr_url']
        solr_collection_name = solr_info['solr_collection_name']
    else:
        solr_host = 'http://localhost:8983/solr/'
        solr_collection_name = 'ecco_datasets'
    if args.grids_to_solr or verify_grids or check_solr_grids(solr_host, solr_collection_name):
        try:
            print(f'\n\033[93mRunning grids_to_solr\033[0m')
            print('=========================================================')
            grids_to_solr = 'grids_to_solr'
            sys.path.insert(1, str(path_to_grids))
            try:
                ret_import = importlib.reload(
                    ret_import                  # pylint: disable=used-before-assignment
                )
            except:
                ret_import = importlib.import_module(grids_to_solr)

            grids_not_in_solr = ret_import.main(
                path=str(path_to_grids), grids_to_use=grids_to_use, verify=verify_grids)
            sys.path.remove(str(path_to_grids))

            if grids_not_in_solr:
                for name in grids_not_in_solr:
                    print(
                        f'Grid "{name}" not in Solr. Ensure it\'s file name is present in grids_config.yaml and run pipeline with the --grids_to_solr argument')
                sys.exit()

            print('\033[92mgrids_to_solr successful\033[0m')
        except Exception as e:
            print(e)
            sys.path.remove(str(path_to_grids))
            print('\033[91mgrids_to_solr failed\033[0m')
        print('=========================================================')

    # ------------------- Output directory -------------------
    if args.output_dir or not output_dir:
        print('\nPlease choose your output directory')

        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        output_dir = f'{filedialog.askdirectory()}/'

        if output_dir == '/':
            print('No output directory given. Exiting.')
            sys.exit()
    else:
        if not os.path.exists(output_dir):
            print(f'{output_dir} is an invalid output directory. Exiting.')
            sys.exit()
    print(f'\nUsing output directory: {output_dir}')

   # ------------------- Add new dataset -------------------
    if args.add_dataset:
        try:
            path_to_tools = Path(__file__).resolve().parents[0]
            sys.path.insert(1, str(path_to_tools))
            try:
                ret_import = importlib.reload(
                    ret_import                      # pylint: disable=used-before-assignment
                )
            except:
                ret_import = importlib.import_module('create_directories')
            ret_import.main()
            sys.path.remove(str(path_to_tools))
            print('\033[92mcreate_directories successful\033[0m')
        except Exception as e:
            sys.path.remove(str(path_to_tools))
            print('\033[91mcreate_directories failed\033[0m')
        print('=========================================================')

    # ------------------- Multiprocessing -------------------
    multiprocessing = not args.single_processing
    user_cpus = args.multiprocesses

    if multiprocessing:
        print(f'Using {user_cpus} processes for multiprocess transformations')
    else:
        print('Using single process transformations')

    # ------------------- Run pipeline -------------------
    while True:
        print('\n------------- OPTIONS -------------')
        print('1) Run all')
        print('2) Harvesters only')
        print('3) Up to aggregation')
        print('4) Dataset input')
        chosen_option = input('Enter option number: ')

        if chosen_option in ['1', '2', '3', '4']:
            break
        else:
            print(
                f'Unknown option entered, "{chosen_option}", please enter a valid option\n'
            )

    # Setup file handler to output log file
    fh_formatter = logging.Formatter(
        "{'time': '%(asctime)s', 'level': '%(levelname)s', 'message': '%(message)s'}")
    fh = logging.FileHandler(logs_path / 'pipeline.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    log.addHandler(fh)

    datasets = [ds for ds in os.listdir(CONFIG_PATH) if ds != '.DS_Store']
    datasets.sort()

    wipe = args.wipe_transformations

    # Run all
    if chosen_option == '1':
        for ds in datasets:
            run_harvester([ds], path_to_harvesters,
                          output_dir, solr_info, grids_to_use)
            run_transformation([ds], path_to_preprocessing,
                               output_dir, multiprocessing, user_cpus, wipe, solr_info, grids_to_use)
            run_aggregation([ds], path_to_preprocessing,
                            output_dir, solr_info, grids_to_use)

    # Run harvester
    elif chosen_option == '2':
        run_harvester(datasets, path_to_harvesters,
                      output_dir, solr_info, grids_to_use)

    # Run up through transformation
    elif chosen_option == '3':
        for ds in datasets:
            run_harvester([ds], path_to_harvesters,
                          output_dir, solr_info, grids_to_use)
            run_transformation([ds], path_to_preprocessing,
                               output_dir, multiprocessing, user_cpus, wipe, solr_info, grids_to_use)

    # Manually enter dataset and pipeline step(s)
    elif chosen_option == '4':
        ds_dict = {i: ds for i, ds in enumerate(datasets, start=1)}
        while True:
            print(f'\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')
            ds_index = input('\nEnter dataset number: ')

            if not ds_index.isdigit() or int(ds_index) not in range(1, len(datasets)+1):
                print(
                    f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        wanted_ds = ds_dict[int(ds_index)]
        print(f'\nUsing {wanted_ds} dataset')

        steps = ['harvest', 'transform', 'aggregate',
                 'harvest and transform', 'transform and aggregate', 'all']
        steps_dict = {i: step for i, step in enumerate(steps, start=1)}
        while True:
            print(f'\nAvailable steps:\n')
            for i, step in steps_dict.items():
                print(f'{i}) {step}')
            steps_index = input('\nEnter pipeline step(s) number: ')

            if not steps_index.isdigit() or int(steps_index) not in range(1, len(steps)+1):
                print(
                    f'Invalid step(s), "{steps_index}", please enter a valid selection')
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if 'harvest' in wanted_steps:
            run_harvester([wanted_ds], path_to_harvesters,
                          output_dir, solr_info, grids_to_use)
        if 'transform' in wanted_steps:
            run_transformation([wanted_ds], path_to_preprocessing,
                               output_dir, multiprocessing, user_cpus, wipe, solr_info, grids_to_use)
        if 'aggregate' in wanted_steps:
            run_aggregation([wanted_ds], path_to_preprocessing,
                            output_dir, solr_info, grids_to_use)
        if wanted_steps == 'all':
            run_harvester([wanted_ds], path_to_harvesters,
                          output_dir, solr_info, grids_to_use)
            run_transformation([wanted_ds], path_to_preprocessing,
                               output_dir, multiprocessing, user_cpus, wipe, solr_info, grids_to_use)
            run_aggregation([wanted_ds], path_to_preprocessing,
                            output_dir, solr_info, grids_to_use)

    print_log(output_dir)
