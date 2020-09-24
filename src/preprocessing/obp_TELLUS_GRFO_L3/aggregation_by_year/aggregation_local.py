import os
import sys
import yaml
import importlib
from pathlib import Path

def main(path=''):
    import aggregation
    aggregation = importlib.reload(aggregation)

    # Pull config information
    if path:
        path_to_yaml = f'{path}/aggregation_config.yaml'
    else:
        path_to_yaml = f'{os.path.dirname(sys.argv[0])}/aggregation_config.yaml'

    with open(path_to_yaml, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    output_dir = f'{Path(__file__).resolve().parents[3]}/output/'

    if '\\' in output_dir:
        output_dir = output_dir.replace('\\', '/')

    aggregation.run_aggregation(output_dir, path=path)

##################################################
if __name__ == "__main__":
    main()

