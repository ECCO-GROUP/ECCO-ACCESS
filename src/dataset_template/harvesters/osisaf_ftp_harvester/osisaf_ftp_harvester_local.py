import os
import sys

from osisaf_ftp_harvester import osisaf_ftp_harvester

if __name__ == '__main__':
    path_to_file_dir = f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/'
    osisaf_ftp_harvester(path_to_file_dir, on_aws=False)
