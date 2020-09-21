import os
import sys
import yaml
import importlib


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

    output_dir = config['output_dir']

    aggregation.run_aggregation(output_dir, path=path)


##################################################
if __name__ == "__main__":
    main()
