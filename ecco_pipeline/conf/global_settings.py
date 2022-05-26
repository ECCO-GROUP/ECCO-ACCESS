import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = Path('/Users/marlis/Developer/ECCO ACCESS/ecco_output')

GRIDS = ['ECCO_llc90_demo.nc', 'ECCO_llc270_demo.nc', 'grid_tpose6_radius.nc']
GRIDS = ['ECCO_llc90_demo.nc']

SOLR_HOST = 'http://localhost:8983/solr/'
SOLR_COLLECTION = 'ecco_datasets'

os.chdir(ROOT_DIR)
