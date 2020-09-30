import os
import sys
import importlib

def main(path=''):
	import osisaf_ftp_harvester
	osisaf_ftp_harvester = importlib.reload(osisaf_ftp_harvester)
	osisaf_ftp_harvester.osisaf_ftp_harvester(path=path, on_aws=False)

if __name__ == '__main__':
    main()
