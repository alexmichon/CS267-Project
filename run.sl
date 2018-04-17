#!/bin/bash
#SBATCH -q regular
#SBATCH -N 20
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH --image=nersc/spark-2.3.0:v1
#SBATCH --volume="/global/cscratch1/sd/wjm9696/tmpfiles:/tmp:perNodeCache=size=200G"
export EXEC_CLASSPATH=path_to_any_extra_needed_jars #Only required if you're using external libraries or jarfiles
module load spark/2.3.0
start-all.sh
shifter spark-submit strong_scaling_ridge.py
stop-all.sh