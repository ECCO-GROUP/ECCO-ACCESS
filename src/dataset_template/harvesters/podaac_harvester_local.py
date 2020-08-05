import sys
import os

from podaac_harvester import podaac_harvester

if __name__ == '__main__':
    path_to_file_dir = f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/'
    podaac_harvester(path_to_file_dir, on_aws=False)
