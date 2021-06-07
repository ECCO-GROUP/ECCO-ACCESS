#!/bin/bash


# Run from top-level directory with datasets (e.g., native/mon_mean, lat-lon/day_mean)
export p=`pwd`

# loop through datasets
for j in * 
do
    cd $j

    echo ""
    echo $j 

    ncdump valid_minmax.nc |grep -vi fill

    echo "MIN"
    for i in `ls *ECCO*nc |head -n 2`; do echo $i; ncdump -h $i |grep valid_min;done
    for i in `ls *ECCO*nc |tail -n 2`; do echo $i; ncdump -h $i |grep valid_min;done

    echo ""
    echo "MAX"
    for i in `ls *ECCO*nc |head -n 2`; do echo $i; ncdump -h $i |grep valid_max;done
    for i in `ls *ECCO*nc |tail -n 2`; do echo $i; ncdump -h $i |grep valid_max;done
    cd $p
done
