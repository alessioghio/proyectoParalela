#PBS -N run-100k-010-0100

#PBS -q gpu

#PBS -r n
#PBS -m n

#PBS -k oe

#PBS -V

###PBS -l walltime=9000:00:00
###PBS -l nodes=8

#PBS -l nodes=box11:ppn=1+box12:ppn=1+box13:ppn=1+box14:ppn=1+box15:ppn=1+box16:ppn=1+box17:ppn=1+box18:ppn=1

###+box16:ppn=1+box15:ppn=1
###+box14:ppn=1+box13:ppn=1
###+box12:ppn=1+box11:ppn=1
###+box10:ppn=1

#!/bin/sh

cd $PBS_O_WORKDIR
export PATH=$PATH:$PBS_O_WORKDIR

OUT_FILE4=run-100k-H4.out
###>$OUT_FILE4

  /usr/local/bin/mpiexec -comm mpich2-pmi -n 8 ./gpu-4th >> $OUT_FILE4

exit 0
