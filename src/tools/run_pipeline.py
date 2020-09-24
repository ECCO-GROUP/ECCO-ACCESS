import os
import sys
import importlib
import numpy as np
from pathlib import Path
from shutil import copyfile

def print_log(log):
	print('\n=========================================================')
	print('===================== \033[36mPrinting log\033[0m ======================')
	print('=========================================================')
	for ds,steps in log.items():
		print(f'\033[93mPipeline status for {ds}\033[0m:')
		if ds == 'obp_AOD1B' or ds == 'ewh_TELLUS_GRAC_MASCON':
			print(f'\033[91mMascon datasets not currently supported\033[0m')
		print(*steps, sep='\n')

def run_harvester(datasets, path_to_harvesters, log):
	print('\n=========================================================')
	print('================== \033[36mRunning harvesters\033[0m ===================')
	print('=========================================================\n')
	for ds in datasets:
		try:
			print(f'\033[93mRunning harvester for {ds}\033[0m')
			print('=========================================================')
			for harvester_folder in os.listdir(path_to_harvesters):
				dataset_path = Path(f'{path_to_harvesters}/{harvester_folder}/{ds}')
				if os.path.exists(dataset_path):
					sys.path.insert(1, str(dataset_path))
					if 'RDEFT4' in ds:
						harvester = 'seaice_harvester_local'
					elif 'podaac' in harvester_folder:
						harvester = 'podaac_harvester_local'
					elif 'osisaf' in harvester_folder:
						harvester = 'osisaf_ftp_harvester_local'
					elif 'nsidc' in harvester_folder:
						harvester = 'nsidc_ftp_harvester_local'
					try:
						ret_import = importlib.reload(ret_import)
					except:
						ret_import = importlib.import_module(harvester)
					ret_import.main(path=dataset_path)
					sys.path.remove(str(dataset_path))
			log.setdefault(ds,[]).append(f'\tHarvest \033[92msuccessful\033[0m')
			print('\033[92mHarvest successful\033[0m')
		except Exception as e:
			sys.path.remove(str(dataset_path))
			print('\033[91mHarvesting failed\033[0m')
			log.setdefault(ds,[]).append(f'\tHarvest \033[91mfailed\033[0m: {e}')
		print('=========================================================')
	return log

def run_transformation(datasets, path_to_preprocessing, log):
	print('\n=========================================================')
	print('=============== \033[36mRunning transformations\033[0m =================')
	print('=========================================================\n')
	for ds in datasets:
		try:
			print(f'\033[93mRunning transformation for {ds}\033[0m')
			print('=========================================================')
			dataset_path = Path(f'{path_to_preprocessing}/{ds}/grid_transformation/')
			if os.path.exists(dataset_path):
				sys.path.insert(1, str(dataset_path))
				transformer = 'grid_transformation_local'
				try:
					ret_import = importlib.reload(ret_import)
				except:
					ret_import = importlib.import_module(transformer)
				ret_import.main(path=dataset_path)
				sys.path.remove(str(dataset_path))
			log.setdefault(ds,[]).append(f'\tTransformation \033[92msuccessful\033[0m')
			print('\033[92mTransformation successful\033[0m')
		except Exception as e:
			sys.path.remove(str(dataset_path))
			print('\033[91mTransformation failed\033[0m')
			log.setdefault(ds,[]).append(f'\tTransform \033[91mfailed\033[0m: {e}')
		print('=========================================================')
	return log

def run_aggregation(datasets, path_to_preprocessing, log):
	print('\n=========================================================')
	print('================ \033[36mRunning aggregations\033[0m ===================')
	print('=========================================================\n')
	for ds in datasets:
		try:
			print(f'\033[93mRunning aggregation for {ds}\033[0m')
			print('=========================================================')
			dataset_path = Path(f'{path_to_preprocessing}/{ds}/aggregation_by_year/')
			if os.path.exists(dataset_path):
				sys.path.insert(1, str(dataset_path))
				aggregator = 'aggregation_local'
				try:
					ret_import = importlib.reload(ret_import)
				except:
					ret_import = importlib.import_module(aggregator)
				ret_import.main(path=dataset_path)
				sys.path.remove(str(dataset_path))
			log.setdefault(ds,[]).append(f'\tAggregation \033[92msuccessful\033[0m')
			print('\033[92mAggregation successful\033[0m')
		except Exception as e:
			sys.path.remove(str(dataset_path))
			print('\033[91mAggregation failed\033[0m')
			log.setdefault(ds,[]).append(f'\tAggregation \033[91mfailed\033[0m: {e}')
		print('=========================================================')
	return log

if __name__ == '__main__':
	# path to harvester and preprocessing folders
	path_to_harvesters = Path(f'{Path(__file__).resolve().parents[1]}/harvesters')
	path_to_preprocessing = Path(f'{Path(__file__).resolve().parents[1]}/preprocessing')
	path_to_grids = Path(f'{Path(__file__).resolve().parents[1]}/grids_to_solr')
	datasets = os.listdir(path_to_preprocessing)

	while True:
		print('------------- OPTIONS -------------')
		print('1) Run all')
		print('2) Harvesters only')
		print('3) Up to aggregation')
		print('4) Dataset input')
		print('5) Y/N for datasets')
		chosen_option = input('Enter option number: ')

		if chosen_option in ['1', '2', '3', '4', '5']:
			break
		else:
			print(f'Unknown option entered, "{chosen_option}", please enter a valid option\n')

	#log = {'ds':['harvest e','transform e','aggregation e']}
	log = {}

	while True:
		run_grids = input('\nUpdate Solr with local grid files? (Y/N): ').upper()
		if run_grids not in ['Y', 'N']:
			print(f'Invalid response, "{run_grids}", please enter a valid response')
		elif run_grids == 'N':
			break
		else:
			try:
				print(f'\n\033[93mRunning grids_to_solr\033[0m')
				print('=========================================================')
				grids_to_solr = 'grids_to_solr'
				sys.path.insert(1, str(path_to_grids))
				try:
					ret_import = importlib.reload(ret_import)
				except:
					ret_import = importlib.import_module(grids_to_solr)
				ret_import.main(path=path_to_grids)
				sys.path.remove(str(path_to_grids))
				log.setdefault('grids',[]).append(f'\tgrids_to_solr \033[92msuccessful\033[0m')
				print('\033[92mgrids_to_solr successful\033[0m')
			except:
				sys.path.remove(str(path_to_grids))
				print('\033[91mgrids_to_solr failed\033[0m')
				log.setdefault('grids',[]).append(f'\tgrids_to_solr \033[91mfailed\033[0m: {e}')
			print('=========================================================')
			break

	if chosen_option == '1':
		for ds in datasets:
			log = run_harvester([ds], path_to_harvesters, log)
			log = run_transformation([ds], path_to_preprocessing, log)
			log = run_aggregation([ds], path_to_preprocessing, log)
	elif chosen_option == '2':
		log = run_harvester(datasets, path_to_harvesters, log)
	elif chosen_option == '3':
		for ds in datasets:
			log = run_harvester([ds], path_to_harvesters, log)
			log = run_transformation([ds], path_to_preprocessing, log)
	elif chosen_option == '4':
		while True:
			wanted_ds = input('\nEnter wanted dataset: ')
			if wanted_ds not in datasets:
				print(f'Invalid dataset, "{wanted_ds}", please enter a valid dataset')
			else:
				break
		while True:
			valid_steps = True
			wanted_steps = input('\nEnter wanted pipeline steps: ').lower().split()
			for step in wanted_steps:
				if (step not in ['harvest', 'transform', 'aggregate', 'all']):
					print(f'Invalid step, "{step}", please enter a valid pipeline step')
					valid_steps = False
			if valid_steps:
				break
		for step in wanted_steps:
			if step == 'harvest':
				log = run_harvester([wanted_ds], path_to_harvesters, log)
			elif step == 'transform':
				log = run_transformation([wanted_ds], path_to_preprocessing, log)
			elif step == 'aggregate':
				log = run_aggregation([wanted_ds], path_to_preprocessing, log)
			elif step == 'all':
				log = run_harvester([wanted_ds], path_to_harvesters, log)
				log = run_transformation([wanted_ds], path_to_preprocessing, log)
				log = run_aggregation([wanted_ds], path_to_preprocessing, log)
	elif chosen_option == '5':
		for ds in datasets:
			while True:
				yes_no = input(f'\nRun pipeline for {ds}? (Y/N): ').upper()
				if (yes_no != 'E') and (yes_no != 'N') and (yes_no != 'Y'):
					print(f'Unknown option entered, "{yes_no}", please enter a valid option')
				else:
					break
			if yes_no == 'Y':
				log = run_harvester([ds], path_to_harvesters, log)
				log = run_transformation([ds], path_to_preprocessing, log)
				log = run_aggregation([ds], path_to_preprocessing, log)
			elif yes_no == 'E':
				break
			else: #yes_no == 'N'
				continue
	print_log(log)