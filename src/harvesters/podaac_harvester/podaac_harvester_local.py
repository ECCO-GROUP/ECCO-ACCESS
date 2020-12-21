import os
import sys
import importlib


def main(config_path='', output_path='', solr_info=''):
    import podaac_harvester
    podaac_harvester = importlib.reload(podaac_harvester)
    podaac_harvester.podaac_harvester(
        config_path=config_path, output_path=output_path, on_aws=False, solr_info=solr_info)


if __name__ == '__main__':
    main()
