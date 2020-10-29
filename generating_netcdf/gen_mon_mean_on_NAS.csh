#PBS -S /bin/tcsh
#PBS -W group_list=g26113
#PBS -l select=26:ncpus=14:model=bro
#PBS -l walltime=02:00:00
#PBS -q devel
##PBS -j oe
##PBS -l select=13:ncpus=28:model=bro

echo "cd into workdir"
cd $PBS_O_WORKDIR

#https://www.nas.nasa.gov/hecc/resources/pleiades.html

echo "set cpus"

# total number of cpus. 
# 26x14 = 364
#set totcpus = 364
set totcpus = 364

# 28 is how many copies for each node (not the total copies)
#set cpuspernode = 28
set cpuspernode = 14

echo ${totcpus}
echo ${cpuspernode}

rm pbs_nodefile
cat "$PBS_NODEFILE" > pbs_nodefile

set cur_grouping=0
set num_groupings = 13

set cpu_start = 0
set cpu_end = `expr $totcpus - 1`

echo "cpu seq: $cpu_start $cpu_end"

while ($cur_grouping < $num_groupings)
	echo "Current Grouping : $cur_grouping"
	seq ${cpu_start} ${cpu_end} | parallel -j ${cpuspernode} -u --sshloginfile "$PBS_NODEFILE" \
		"cd $PWD; /bin/tcsh ./invoke_python_podaac.csh ${totcpus} {} ${cur_grouping}"
	@ cur_grouping++