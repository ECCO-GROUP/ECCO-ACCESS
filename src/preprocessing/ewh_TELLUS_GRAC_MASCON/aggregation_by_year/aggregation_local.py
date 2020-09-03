import os
import sys
import yaml
from aggregation import run_aggregation


##################################################
if __name__ == "__main__":
    # Pull config information
    path_to_yaml = f'{os.path.dirname(sys.argv[0])}/aggregation_config.yaml'
    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    output_dir = config['output_dir']

    run_aggregation(output_dir)
