import os
import sys
import numpy as np
from pathlib import Path
from shutil import copyfile

# path to harvester and preprocessing folders
path_to_harvesters = Path(f'{Path(__file__).parents[1]}/harvesters')
path_to_preprocessing = Path(f'{Path(__file__).parents[1]}/preprocessing')

name = input("tell me your name...or else...")
print(name)

# 1) Run all
# 2) Harvesters only
# 3) Up to transformations
# 4) Up to aggregation
# 5) Dataset input
# 6) Y/N for datasets
