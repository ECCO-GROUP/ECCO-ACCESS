import os
import sys
import importlib


def main(config_path='', output_path=''):
    import podaac_harvester
    podaac_harvester = importlib.reload(podaac_harvester)
    podaac_harvester.podaac_harvester(
        config_path=config_path, output_path=output_path, on_aws=False)


if __name__ == '__main__':
    main()
