#PBS -S /bin/tcsh
#PBS -W group_list=g26113
#PBS -l select=1:ncpus=5:model=bro
#PBS -q normal
#PBS -l walltime=08:00:00

##PBS -l walltime=02:00:00
##PBS -q devel
##PBS -j oe
##PBS -l select=78:ncpus=4:model=bro
##PBS -l select=40:ncpus=12:model=bro

echo "cd into workdir"
cd $PBS_O_WORKDIR

#https://www.nas.nasa.gov/hecc/resources/pleiades.html

echo "set cpus"

# total number of cpus. 
set totcpus = 5

echo "Total CPUS ${totcpus}"

set cpuspernode = 5 
echo "CPUS per node ${cpuspernode}"

# NATIVE GRID groupings
# ---- 2D
# 0 dynamic sea surface height and model sea level anomaly
# 1 ocean bottom pressure and model ocean bottom pressure anomaly
# 2 ocean and sea-ice surface freshwater fluxes
# 3 ocean and sea-ice surface heat fluxes
# 4 atmosphere surface temperature, humidity, wind, and pressure
# 5 ocean mixed layer depth
# 6 ocean and sea-ice surface stress
# 7 sea-ice and snow concentration and thickness
# 8 sea-ice velocity
# 9 sea-ice and snow horizontal volume fluxes

# ---- 3D
# 10 Gent-McWilliams ocean bolus transport streamfunction
# 11 ocean three-dimensional volume fluxes
# 12 ocean three-dimensional potential temperature fluxes
# 13 ocean three-dimensional salinity fluxes
# 14 sea-ice salt plume fluxes
# 15 ocean potential temperature and salinity
# 16 ocean density, stratification, and hydrostatic pressure
# 17 ocean velocity
# 18 Gent-McWilliams ocean bolus velocity
# 19 ocean three-dimensional momentum tendency

rm pbs_nodefile
cat "$PBS_NODEFILE" > pbs_nodefile

set cpu_start = 0
echo "cpu_start ${cpu_start}"

set cpu_end = `expr $totcpus - 1`
echo "cpu_end ${cpu_end}"

set ecco_access_dir = '/home5/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf'
set base_dir = '/nobackupp2/ifenty/podaac_20201216/native/day_inst/'
set run_description = 'update_valid_minmax_calculate_and_apply_native_day_inst'
set calculate_valid_minmax = 'True'
set calculate_valid_minmax_method = 'easy'
set apply_valid_minmax = 'True'
set valid_minmax_scaling = 10.0

# arguments for invoke_python_update_valid_minmax.csh
#>> run_description, ecco_access_dir, base_dir, calculate_valid_minmax, calculate_valid_minmax_method, apply_valid_minmax, valid_minmax_scaling, grouping_to_process		

seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
    "cd $PWD; /bin/tcsh ${ecco_access_dir}/invoke_python_update_valid_minmax.csh ${run_description} ${ecco_access_dir} ${base_dir} ${calculate_valid_minmax} ${calculate_valid_minmax_method} ${apply_valid_minmax} ${valid_minmax_scaling} {}"
