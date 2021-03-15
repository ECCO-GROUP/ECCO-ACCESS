#!/bin/tcsh -fe

#  --dataset_base_dir DATASET_BASE_DIR
#                        directory containing dataset grouping subdirectories


#seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
#    "cd $PWD; /bin/tcsh ${ecco_access_dir}/invoke_python_valid_minmax_v4.csh ${run_description} ${ecco_access_dir} ${base_dir} ${output_dir} ${time_type} ${grid_type} ${n_workers} ${threads_per_worker} {}"



set ecco_access_dir = '/home5/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf'
set output_dir = '/nobackupp2/ifenty/podaac/podaac_20201216/valid_minmax_20210312_single_node/native/day_mean'
set base_dir = '/nobackupp2/ifenty/podaac/podaac_20201216/final_delivered/'
set time_type = "day_mean"
set grid_type = "native"
set run_description = 'calc_minmax_native_day_mean_single_node'

echo ""
echo " invoke:run_description : $run_description"
echo " invoke:ecco_access_dir  : $ecco_access_dir"
echo " invoke:base_dir  : $base_dir"
echo " invoke:output_dir : $output_dir"
echo " invoke:time_type : $time_type"
echo " invoke:grid_type : $grid_type"
echo ""

mkdir -p       run_${run_description}

foreach grouping (`seq 0 20`)
    echo " invoke:grouping : ${grouping}"

    printenv     > run_${run_description}/env_${grouping}

    echo `date` >> run_${run_description}/start_date_${grouping}

    conda activate /nobackupp2/ifenty/envs/ecco

    echo "invoking python"
    python ${ecco_access_dir}/valid_minmax_v5.py --dataset_base_dir=${base_dir} --grid_type=${grid_type} --time_type=${time_type} --grouping_id=${grouping} --output_dir=${output_dir} > run_${run_description}/output_${grouping}

    echo `date` >> run_${run_description}/end_date_${grouping}
end
