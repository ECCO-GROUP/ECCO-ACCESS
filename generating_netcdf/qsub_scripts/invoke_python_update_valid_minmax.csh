#!/bin/tcsh -fe

#  --dataset_base_dir DATASET_BASE_DIR
#                        directory containing dataset grouping subdirectories
#  --calculate_valid_minmax [CALCULATE_VALID_MINMAX]
#                        calculate the valid minmax for the groupings in
#                        dataset_base_dir
#  --apply_valid_minmax [APPLY_VALID_MINMAX]
#                        apply the new minmax for the groupings in
#                        dataset_base_dir
#  --valid_minmax_scaling_factor VALID_MINMAX_SCALING_FACTOR
#                        scaling factor by which to inflate the valid min and
#                        valid max
#  --valid_minmax_prec {32,64}
#                        32 or 64 bit precision for valid min and max
#  --grouping_to_process GROUPING_TO_PROCESS
#                        which grouping id to process, -1 = all

set run_description = $1
set ecco_access_dir  = $2
set base_dir  = $3
set calculate_valid_minmax = $4
set calculate_valid_minmax_method = $5
set apply_valid_minmax = $6
set valid_minmax_scaling = $7
set grouping = $8

echo ""
echo " run_description : $1"
echo " ecco_access_dir  : $2"
echo " base_dir  : $3"
echo " calculate_valid_minmax : $4"
echo " calculate_valid_minmax_method : $5"
echo " apply_valid_minmax : $6"
echo " valid_minmax_scaling : $7"
echo " grouping : $8"
echo ""

mkdir -p       run_${run_description}
printenv     > run_${run_description}/env_${grouping}

echo "Executing run $1 $2 $3 $4 $5 $6 $7 $8 on $HOST in $PWD" > run_${run_description}/exec_${grouping}

echo `date` >> run_${run_description}/start_date_${grouping}

conda activate ecco

echo "invoking python $1 $2 $3 $4 $5 $6 $7 $8"

python ${ecco_access_dir}/update_valid_minmax.py --dataset_base_dir ${base_dir} --calculate_valid_minmax ${calculate_valid_minmax} --calculate_valid_minmax_method ${calculate_valid_minmax_method} --apply_valid_minmax ${apply_valid_minmax} --valid_minmax_scaling ${valid_minmax_scaling} --grouping_to_process ${grouping} > run_${run_description}/output_${grouping}

echo `date` >> run_${run_description}/end_date_${grouping}
