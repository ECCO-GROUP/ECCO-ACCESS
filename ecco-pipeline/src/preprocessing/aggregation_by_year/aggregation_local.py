import os
import sys
import yaml
import importlib
from pathlib import Path


def main(config_path='', output_path='', solr_info='', grids_to_use=[]):
    import aggregation
    aggregation = importlib.reload(aggregation)

    # Pull config information
    if not config_path:
        print('No path for configuration file. Can not run aggregation.')
        return

    aggregation.run_aggregation(
        output_path, config_path=config_path, solr_info=solr_info, grids_to_use=grids_to_use)


##################################################
if __name__ == "__main__":
    main()
