import os
import sys
import importlib

def main(path=''):
	import seaice_harvester
	seaice_harvester = importlib.reload(seaice_harvester)
	seaice_harvester.seaice_harvester(path=path, on_aws=False)

if __name__ == '__main__':
    main()

