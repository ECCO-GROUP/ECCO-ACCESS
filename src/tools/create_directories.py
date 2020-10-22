import os
import sys
import numpy as np
from pathlib import Path
from shutil import copyfile


def main():
    # all config file paths
    transformation_config_path = Path(
        f"{Path(__file__).resolve().parents[1]}/preprocessing/grid_transformation/grid_transformation_config.yaml"
    )
    aggregation_config_path = Path(
        f"{Path(__file__).resolve().parents[1]}/preprocessing/aggregation_by_year/aggregation_config.yaml"
    )
    podaac_config_path = Path(
        f"{Path(__file__).resolve().parents[1]}/harvesters/podaac_harvester/podaac_harvester_config.yaml"
    )
    osisaf_config_path = Path(
        f"{Path(__file__).resolve().parents[1]}/harvesters/osisaf_ftp_harvester/osisaf_ftp_harvester_config.yaml"
    )
    nsidc_config_path = Path(
        f"{Path(__file__).resolve().parents[1]}/harvesters/nsidc_ftp_harvester/nsidc_ftp_harvester_config.yaml"
    )

    # path to datasets folder
    path_to_datasets = Path(f"{Path(__file__).resolve().parents[2]}/datasets")

    # get ds_name and harvest_type from user
    ds_name = ""
    harvest_type = ""
    ds_name = input("Enter ds_name for dataset: ")
    harvest_type = input(
        "Enter harvester type (PODAAC, NSIDC, OSISAF): ").upper()

    # get correct config file name and path for wanted harvester type
    if harvest_type == "PODAAC":
        harvester_config_path = podaac_config_path
    elif harvest_type == "NSIDC":
        harvester_config_path = nsidc_config_path
    elif harvest_type == "OSISAF":
        harvester_config_path = osisaf_config_path
    else:
        print("Unsupported harvester type")
        return

    # create new dataset directory
    path_to_new_dataset = Path(f"{path_to_datasets}/{ds_name}")
    if not os.path.exists(path_to_new_dataset):
        os.makedirs(path_to_new_dataset)

    # harvester config file copy
    destination_file = Path(f"{path_to_new_dataset}/harvester_config.yaml")
    copyfile(harvester_config_path, destination_file)

    # transformation config file copy
    destination_file = Path(
        f"{path_to_new_dataset}/grid_transformation_config.yaml")
    copyfile(transformation_config_path, destination_file)

    # aggregation config file copy
    destination_file = Path(f"{path_to_new_dataset}/aggregation_config.yaml")
    copyfile(aggregation_config_path, destination_file)


#############################################
if __name__ == "__main__":
    main()
