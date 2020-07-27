import sys
import os
import yaml
from aggregation import run_aggregation


def aggregation_handler(system_path, output_dir):
    run_aggregation(system_path, output_dir=output_dir)


##################################################
if __name__ == "__main__":
    system_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Pull config information
    path_to_yaml = system_path + "/aggregation_config.yaml"
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream)

    output_dir = config['output_dir']

    aggregation_handler(system_path, output_dir)
