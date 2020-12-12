#!/bin/bash
#SBATCH -J proyCPD
#SBATCH -p investigacion
#SBATCH -N 2
#SBATCH --tasks-per-node=4
#SBATCH --mem-per-cpu=1GB

module load openmpi/2.1.6 
mpirun -np 2 ./cpu-4th
module unload openmpi/2.1.6
