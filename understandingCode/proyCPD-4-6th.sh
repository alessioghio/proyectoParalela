#!/bin/bash
#SBATCH -J proyCPD
#SBATCH -p investigacion
#SBATCH -N 2
#SBATCH --tasks-per-node=2
#SBATCH --mem-per-cpu=1GB
#SBTACH --array=1-10

module load openmpi/2.1.6
mpirun -np 4 cpu-6th
module unload openmpi/2.1.6
