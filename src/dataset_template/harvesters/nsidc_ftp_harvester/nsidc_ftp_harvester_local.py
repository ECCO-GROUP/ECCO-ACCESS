import os
import sys
import importlib

def main(path=''):
	import nsidc_ftp_harvester
	nsidc_ftp_harvester = importlib.reload(nsidc_ftp_harvester)
	nsidc_ftp_harvester.nsidc_ftp_harvester(path=path, on_aws=False)

if __name__ == '__main__':
    main()
