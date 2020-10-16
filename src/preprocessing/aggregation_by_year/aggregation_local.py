import os
import sys
import yaml
import importlib
from pathlib import Path


def main(config_path='', output_path=''):
    import aggregation
    aggregation = importlib.reload(aggregation)

    # Pull config information
    if not config_path:
        print('No path for configuration file. Can not run aggregation.')
        return

    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    aggregation.run_aggregation(output_path, config_path=config_path)


##################################################
if __name__ == "__main__":
    main()
