#PBS -S /bin/tcsh
#PBS -W group_list=g26113

### DEVEL
#PBS -l select=5:ncpus=6:model=bro
#PBS -l walltime=02:00:00
#PBS -q devel
##PBS -j oe

### NORMAL
##PBS -q normal
##PBS -l walltime=08:00:00

## 2D
##PBS -l select=26:ncpus=12:model=bro

## 3D
##PBS -l select=78:ncpus=4:model=bro

echo "cd into workdir"
cd $PBS_O_WORKDIR

#https://www.nas.nasa.gov/hecc/resources/pleiades.html

#rm -fr run_*

echo "set cpus"

# total number of cpus. 
set totcpus = 30 

echo "Total CPUS ${totcpus}"

# 28 maximum possible cpus per node on Broadwell
set cpuspernode = 6
echo "CPUS per node ${cpuspernode}"

# NATIVE GRID groupings
# (in alphabetical order)

# 0 ATM_SURFACE_TEMP_HUM_WIND_PRES
# 1 OCEAN_3D_MOMENTUM_TEND
# 2 OCEAN_3D_SALINITY_FLUX
# 3 OCEAN_3D_TEMPERATURE_FLUX
# 4 OCEAN_3D_VOLUME_FLUX
# 5 OCEAN_AND_ICE_SURFACE_FW_FLUX
# 6 OCEAN_AND_ICE_SURFACE_HEAT_FLUX
# 7 OCEAN_AND_ICE_SURFACE_STRESS
# 8 OCEAN_BOLUS_STREAMFUNCTION
# 9 OCEAN_BOLUS_VELOCITY
# 10 OCEAN_BOTTOM_PRESSURE
# 11 OCEAN_DENS_STRAT_PRESS
# 12 OCEAN_MIXED_LAYER_DEPTH
# 13 OCEAN_TEMPERATURE_SALINITY
# 14 OCEAN_VELOCITY
# 15 SEA_ICE_CONC_THICKNESS
# 16 SEA_ICE_HORIZ_VOLUME_FLUX
# 17 SEA_ICE_SALT_PLUME_FLUX
# 18 SEA_ICE_VELOCITY
# 19 SEA_SURFACE_HEIGHT

# 3D LIST (9 datasets)
# 1 2 3 4 8 9 11 13 14

rm pbs_nodefile
cat "$PBS_NODEFILE" > pbs_nodefile

set cpu_start = 0
echo "cpu_start ${cpu_start}"

set cpu_end = `expr $totcpus - 1`
echo "cpu_end ${cpu_end}"

set run_description = 'fix_ZuZl_native_snapshot'
set ecco_access_dir = '/home5/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/generating_netcdf'
set dataset_base_dir = '/nobackupp2/ifenty/podaac_20201216/native/day_inst'
#set dataset_base_dir = '/nobackupp2/ifenty/podaac_tmp/mon_mean'

foreach cur_grouping (1)
#foreach cur_grouping (0 1 2) #1 2 3 4 8 9 11 13 14)
	echo "Current Grouping : ${cur_grouping}"
	
	seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
		"cd $PWD; /bin/tcsh ${ecco_access_dir}/invoke_python_fix_ZuZl.csh ${run_description} ${ecco_access_dir} ${dataset_base_dir} ${totcpus} {} ${cur_grouping}"
	
end
