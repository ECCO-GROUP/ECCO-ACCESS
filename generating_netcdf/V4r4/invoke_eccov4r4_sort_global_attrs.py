#!/bin/bash

export ea_dir=/home5/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf
export db_dir=/nobackupp2/ifenty/podaac/podaac_20201216/final_delivered_test_3/

echo $i

if [ $1 = "day_inst_native" ]; then
  export tt='day_inst'
  export gt='native'
  echo "${tt} ${gt}"
  for i in `seq 0 4`
  do
    echo $i
    python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i} --debug --grid_type=${gt} --time_type=${tt} > ./sort_out${tt}_${gt}_${i}.txt &
    #python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i} --debug --grid_type=${gt} --time_type=${tt} 
  done
fi

if [ $1 = "day_mean_native" ]; then
  export tt='day_mean'
  export gt='native'
  echo "${tt} ${gt}"
  for i in `seq 0 19`
  do
    echo $i
    python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i}  --debug --grid_type=${gt} --time_type=${tt} > ./sort_out${tt}_${gt}_${i}.txt &
  done
fi

if [ $1 = "mon_mean_native" ]; then
  export tt='mon_mean'
  export gt='native'
  echo "${tt} ${gt}"
  for i in `seq 0 19`
  do
    echo $i
    python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i}  --debug --grid_type=${gt} --time_type=${tt} > ./sort_out${tt}_${gt}_${i}.txt &
  done
fi


if [ $1 = "mon_mean_lat-lon" ]; then
  export tt='mon_mean'
  export gt='lat-lon'
  echo "${tt} ${gt}"
  for i in `seq 0 12`
  do
    echo $i
    python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i}  --debug --grid_type=${gt} --time_type=${tt} > ./sort_out${tt}_${gt}_${i}.txt &
  done
fi

if [ $1 = "day_mean_lat-lon" ]; then
  export tt='day_mean'
  export gt='lat-lon'
  echo "${tt} ${gt}"
  for i in `seq 0 12`
  do
    echo $i
    python ${ea_dir}/eccov4r4_sort_global_attrs.py --dataset_base_dir=${db_dir} --grouping_id ${i}  --debug --grid_type=${gt} --time_type=${tt} > ./sort_out${tt}_${gt}_${i}.txt &
  done
fi

