import os
import sys
import importlib


def main(config_path='', output_path='', solr_info='', grids_to_use=[]):
    import podaac_harvester
    podaac_harvester = importlib.reload(podaac_harvester)
    podaac_harvester.podaac_harvester(config_path=config_path, output_path=output_path,
                                      solr_info=solr_info, grids_to_use=grids_to_use)


if __name__ == '__main__':
    main()
