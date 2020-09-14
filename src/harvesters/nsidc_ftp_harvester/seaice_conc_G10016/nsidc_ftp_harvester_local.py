import os
import sys

from nsidc_ftp_harvester import nsidc_ftp_harvester

if __name__ == '__main__':
    nsidc_ftp_harvester(on_aws=False)
