#!/bin/tcsh -fe

#  --dataset_base_dir DATASET_BASE_DIR
#                        directory containing dataset grouping subdirectories


#seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
#    "cd $PWD; /bin/tcsh ${ecco_access_dir}/invoke_python_valid_minmax_v4.csh ${run_description} ${ecco_access_dir} ${base_dir} ${output_dir} ${time_type} ${grid_type} ${n_workers} ${threads_per_worker} {}"


set run_description = $1
set ecco_access_dir  = $2
set base_dir  = $3
set output_dir = $4
set time_type = $5
set grid_type = $6
set grouping = $7

echo ""
echo " invoke:run_description : $1"
echo " invoke:ecco_access_dir  : $2"
echo " invoke:base_dir  : $3"
echo " invoke:output_dir : $4"
echo " invoke:time_type : $5"
echo " invoke:grid_type : $6"
echo " invoke:grouping : $7"
echo ""

mkdir -p       run_${run_description}
printenv     > run_${run_description}/env_${grouping}

echo "Executing run $1 $2 $3 $4 $5 $6 $7 on $HOST in $PWD" > run_${run_description}/exec_${grouping}

echo `date` >> run_${run_description}/start_date_${grouping}

conda activate /nobackupp2/ifenty/envs/ecco

echo "invoke: arguments $1 $2 $3 $4 $5 $6 $7"
#python ./valid_minmax_v4.py --dataset_base_dir=/nobackupp2/ifenty/podaac/podaac_20201216/final_delivered --grid_type=native --time_type=mon_mean --grouping_id=1 --output_dir=/nobackupp2/ifenty/podaac/podaac_20201216/valid_minmax_20210311b --n_workers=7 --threads_per_worker=2

#cd /nobackupp2/ifenty/podaac/podaac_20201216

#python ${ecco_access_dir}/valid_minmax_v4.py --debug --dataset_base_dir=${base_dir} --grid_type=${grid_type} --time_type=${time_type} --grouping_id=${grouping} --output_dir=${output_dir} --n_workers=${n_workers} --threads_per_worker=${threads_per_worker} > run_${run_description}/output_${grouping}
python ${ecco_access_dir}/valid_minmax_v5.py  --dataset_base_dir=${base_dir} --grid_type=${grid_type} --time_type=${time_type} --grouping_id=${grouping} --output_dir=${output_dir} > run_${run_description}/output_${grouping}

echo `date` >> run_${run_description}/end_date_${grouping}
