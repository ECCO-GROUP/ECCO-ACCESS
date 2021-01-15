#!/bin/tcsh -fe

#  --dataset_base_dir DATASET_BASE_DIR
#                        directory containing dataset grouping subdirectories
#  --show_grouping_numbers [SHOW_GROUPING_NUMBERS]
#                        show id of each grouping (dataset)
#  --fix_ZuZlkukl [FIX_ZUZLKUKL]
#                        flag whether or not to update coordinates
#  --dry_run [DRY_RUN]   flag whether or not to apply changes
#  --grouping_to_process GROUPING_TO_PROCESS [GROUPING_TO_PROCESS ...]
#  --n_jobs N_JOBS
#  --job_id JOB_ID

set run_description = $1
set ecco_access_dir  = $2
set base_dir  = $3
set n_jobs = $4
set job_id = $5
set grouping = $6

echo ""
echo " run_description : $1"
echo " ecco_access_dir  : $2"
echo " base_dir  : $3"
echo " n_jobs : $4"
echo " job_id : $5"
echo " grouping : $6"
echo ""

set grouping_str = `printf "%03d" $6`
set job_id_str = `printf "%03d" $5`

mkdir -p       run_${run_description}
printenv     > run_${run_description}/env_${grouping_str}_${job_id_str}

echo "Executing run $1 $2 $3 $4 $5 $8 on $HOST in $PWD" > run_${run_description}/exec_${grouping_str}_${job_id_str}

echo `date` >> run_${run_description}/start_date_${grouping_str}_${job_id_str}

conda activate ecco


echo "invoking python $1 $2 $3 $4 $5 $6"

python ${ecco_access_dir}/fix_Zul_kul.py --dataset_base_dir=${base_dir} --dry_run=False --grouping_to_process=${grouping} --n_jobs=${n_jobs} --job_id=${job_id} --fix_ZuZlkukl=True --show_grouping_numbers=True > run_${run_description}/output_${grouping_str}_${job_id_str}

echo `date` >> run_${run_description}/end_date_${grouping_str}_${job_id_str}
