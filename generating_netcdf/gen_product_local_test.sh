#!/bin/bash

for i in `seq 20`
do
#   python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20201223' --output_freq_code='AVG_DAY' --grouping_to_process=$i --product_type='native'

#   python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20201223' --output_freq_code='AVG_MON' --grouping_to_process=$i --product_type='native'

#   python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20201223' --output_freq_code='SNAPSHOT' --grouping_to_process=$i --product_type='native'

   python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20201223' --output_freq_code='AVG_MON' --grouping_to_process=$i --product_type='latlon'

   python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir='/home/ifenty/tmp/v4r4_nc_output_20201223' --output_freq_code='AVG_DAY' --grouping_to_process=$i --product_type='latlon'
done
