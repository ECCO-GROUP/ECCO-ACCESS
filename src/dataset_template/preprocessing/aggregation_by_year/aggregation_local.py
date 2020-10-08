import os
import sys
import yaml
import importlib
from pathlib import Path


def main(path=''):
    import aggregation
    aggregation = importlib.reload(aggregation)

    # Pull config information
    if not path:
        print('No path for configuration file. Can not run aggregation.')
        return

    with open(path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    output_dir = f'{Path(__file__).resolve().parents[3]}/output/'

    if '\\' in output_dir:
        output_dir = output_dir.replace('\\', '/')

    aggregation.run_aggregation(output_dir, path=path)


##################################################
if __name__ == "__main__":
    main()
