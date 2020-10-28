#!/bin/tcsh -fe

echo "in tcsh script $2"

rm -fr run_*

mkdir -p run_$1_$3

echo $SHELL  > run_$1_$3/shell_$2
echo $1     >> run_$1_$3/shell_$2
echo $2     >> run_$1_$3/shell_$2
echo `date` >> run_$1_$3/shell_$2

printenv     > run_$1_$3/env_$2

echo "Executing run $1 $2 on $HOST in $PWD" > run_$1_$3/exec_$2

conda activate ecco

echo "invoking python $1 $2 $3"
python /home5/ifenty/git_repos_others/ECCO-ACCESS/generating_netcdf/eccov4r4_gen_for_podaac_20201022_PFE_as_func.py $1 $2 $3 > run_$1_$3/output_$2
