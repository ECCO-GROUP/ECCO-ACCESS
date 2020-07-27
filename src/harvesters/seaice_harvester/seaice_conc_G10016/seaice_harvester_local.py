# for local run
import sys
import os

from seaice_harvester import seaice_harvester

if __name__ == '__main__':
    path_to_file_dir = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'
    seaice_harvester(path_to_file_dir, on_aws=False)

