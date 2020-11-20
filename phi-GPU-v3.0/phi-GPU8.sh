#PBS -N run-025k-H8

#PBS -q gpu

#PBS -r n
#PBS -m n

#PBS -k oe

#PBS -V

###PBS -l walltime=9000:00:00
###PBS -l nodes=2

#PBS -l nodes=box13:ppn=1+box14:ppn=1
###+box11:ppn=1+box12:ppn=1+box13:ppn=1+box14:ppn=1+box15:ppn=1+box16:ppn=1+box17:ppn=1+box18:ppn=1

#!/bin/sh

cd $PBS_O_WORKDIR
export PATH=$PATH:$PBS_O_WORKDIR

OUT_FILE8=run-025k-H8.out
>$OUT_FILE8

  /usr/local/bin/mpiexec -comm mpich2-pmi -n 2 ./gpu-8th >> $OUT_FILE8

exit 0
