import os
import sys
import importlib

def main(path=''):
	import podaac_harvester
	podaac_harvester = importlib.reload(podaac_harvester)
	podaac_harvester.podaac_harvester(path=path, on_aws=False)

if __name__ == '__main__':
    main()
