import os
import sys

from nsidc_ftp_harvester import nsidc_ftp_harvester

if __name__ == '__main__':
    path_to_file_dir = f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/'
    nsidc_ftp_harvester(path_to_file_dir, on_aws=False)
