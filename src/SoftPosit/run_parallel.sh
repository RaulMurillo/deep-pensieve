#!/bin/bash

START=$(date +%s)
BATCH=256
INIT=0
for i in {71..100}
do
        ARG1=$(( INIT  + i * BATCH ))
        python svhn_inference_quire.py $ARG1 $BATCH
done

END=$(date +%s)
DIFF=$(( $END - $START ))
S=$(( $DIFF % 60 ))
Maux=$(( $DIFF / 60 ))
M=$(( $Maux % 60 ))
Haux=$(( $Maux / 60 ))
H=$(( $Haux % 24 ))
D=$(( $Haux / 24 ))
echo "It took $D days, $H hours, $M mins, $S seconds"
