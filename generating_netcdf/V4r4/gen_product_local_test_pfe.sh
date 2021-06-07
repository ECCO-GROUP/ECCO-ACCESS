#!/bin/bash

#    parser.add_argument('--time_steps_to_process', default='by_job', nargs="+",\
#                        help='which time steps to process')

#    parser.add_argument('--num_jobs', type=int, default=1,\
#                        help='the total number of separate jobs you are using to process the files')

#    parser.add_argument('--job_id', type=int, default=0,\
#                        help='the id of this particular job (out of num_jobs)')

#    parser.add_argument('--grouping_to_process', required=True, type=int,\
#                        help='which dataset grouping to process, there are 20 in v4r4')

#    parser.add_argument('--product_type', required=True, type=str, choices=['latlon', 'native'], \
#                        help='one of either "latlon" or "native" ')

#    parser.add_argument('--output_freq_code', required=True, type=str, choices=['AVG_MON','AVG_DAY','SNAPSHOT'],\
#                        help='one of AVG_MON, AVG_DAY, or SNAPSHOT')

#    parser.add_argument('--output_dir', required=True, type=str,\
#                        help='output directory')

export output_dir='/nobackupp2/ifenty/tmp/nc_test/'
export grouping_to_process=13
export product_type='native'
export job_id=0
export num_jobs=312
export output_freq_code='AVG_MON'
 
python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir=$output_dir --output_freq_code=$output_freq_code --grouping_to_process=$grouping_to_process --product_type=$product_type --num_jobs=$num_jobs --job_id=$job_id 


export grouping_to_process=17777777
python ~/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac.py --output_dir=$output_dir --output_freq_code=$output_freq_code --grouping_to_process=$grouping_to_process --product_type=$product_type --num_jobs=$num_jobs --job_id=$job_id 
