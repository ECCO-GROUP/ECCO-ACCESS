#!/bin/bash
echo $0
echo $1

python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20210305' --output_freq_code='AVG_MON' --grouping_to_process=$1 --product_type='native'
python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20210305' --output_freq_code='AVG_MON' --grouping_to_process=$1 --product_type='latlon'
