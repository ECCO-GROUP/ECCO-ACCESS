#!/bin/tcsh -fe

#echo "in tcsh script $2"

# 1 = num jobs
# 2 = job id
# 3 = grouping id
set numjobs  = `printf "%05d" $1`
set job_id   = `printf "%05d" $2`
set gid      = `printf "%05d" $3`

mkdir -p run_${numjobs}_${gid}

printenv     > run_${numjobs}_${gid}/env_${job_id}

echo "Executing run $1 $2 $3 on $HOST in $PWD" > run_${numjobs}_${gid}/exec_${numjobs}_${gid}
echo `date` >> run_${numjobs}_${gid}/exec_${numjobs}_${gid}

conda activate ecco

echo "invoking python $1 $2 $3"

python /home5/ifenty/git_repos_others/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac_20201022_PFE_as_func.py $1 $2 $3 > run_${numjobs}_${gid}/output_${job_id}
