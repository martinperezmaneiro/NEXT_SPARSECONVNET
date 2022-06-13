#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

source /lhome/ext/ific020/ific0201/.bashrc

# might need to setup more environmental variables
cd /lhome/ext/ific020/ific0201/NEXT_SPARSECONVNET/
source setup.sh
cd -
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=4


main.py -a train -conf /lhome/ext/ific020/ific0201/my_scripts/segnet_jul09_5mm.conf
