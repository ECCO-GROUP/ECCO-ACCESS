#PBS -S /bin/tcsh
#PBS -W group_list=g26113
#PBS -l select=78:ncpus=4:model=bro
#PBS -l walltime=02:00:00
#PBS -q devel
##PBS -j oe
### FOR 2D
##PBS -l select=26:ncpus=12:model=bro

echo "cd into workdir"
cd $PBS_O_WORKDIR

#https://www.nas.nasa.gov/hecc/resources/pleiades.html

rm -fr run_*

echo "set cpus"

# total number of cpus. 
# for 2D fields totcpus = 312 (26x12)
# FOR 3D FIELDS TOTCPUS = 312 (78X4)
# alternative = 468 (78x6)
set totcpus = 312

# 28 maximum possible cpus per node on Broadwell
# for 2D fields cpuspernode = 12
#set cpuspernode = 12
# for 3D fields cpuspernode = 4
# alternative 6
set cpuspernode = 4


echo ${totcpus}
echo ${cpuspernode}

rm pbs_nodefile
cat "$PBS_NODEFILE" > pbs_nodefile

# latlon fields have 13 groups
# 0..8 are 2D
# 9..12 are 13

set cur_grouping  = 11
# end grouping (if one grouping, 1+cur_grouping)
set num_groupings = 12 

set cpu_start = 0
set cpu_end = `expr $totcpus - 1`

echo "cpu seq: $cpu_start $cpu_end"

while ($cur_grouping < $num_groupings)
	echo "Current Grouping : $cur_grouping"
	echo `date`
		
	seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
		"cd $PWD; /bin/tcsh ./invoke_python_podaac.csh ${totcpus} {} ${cur_grouping}"
	
	set cur_grouping = `expr $cur_grouping + 1`
	echo "incremented cur_grouping ${cur_grouping}"
	echo `date`
end