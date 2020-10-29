#!/bin/bash
while true
do
echo "qstat query"
qstat -u ifenty
echo ""
echo `date`
export x=`find . |grep SEA_SURFACE_HEIGHT |wc -l`
echo " 1. SSH  : $x"

export x=`find . |grep OCEAN_BOTTOM_PRES |wc -l`
echo " 2. OBP  : $x"

export x=`find . |grep FW |wc -l`
echo " 3. FW   : $x"

export x=`find . |grep HEAT |wc -l`
echo " 4. HEAT : $x"

export x=`find . |grep ATM |wc -l`
echo " 5. ATM  : $x"

export x=`find . |grep MIXED |wc -l`
echo " 6. MIXED: $x"

export x=`find . |grep STRESS |wc -l`
echo " 7. STRES: $x"

export x=`find . |grep CONC |wc -l`
echo " 8. CONC : $x"

export x=`find . |grep ICE_VEL |wc -l`
echo " 9. ICEVE: $x"

export x=`find . |grep OCEAN_TEMP | wc -l`
echo "10. TS   : $x"

export x=`find . |grep DENS |wc -l`
echo "11. DENS : $x"

export x=`find . |grep OCEAN_VEL |wc -l`
echo "12. VEL  : $x"

export x=`find . |grep BOLUS |wc -l`
echo "13. BOLUS: $x"

sleep $1 
done
