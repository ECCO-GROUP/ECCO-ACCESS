# for local run
import sys
import os

from ftp_harvester import ftp_harvester

if __name__ == '__main__':
    path_to_file_dir = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'
    ftp_harvester(path_to_file_dir, on_aws=False)
