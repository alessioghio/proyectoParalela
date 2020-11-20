#PBS -N phi-GPU-test

#PBS -q gpu

#PBS -r n
#PBS -m n

#PBS -k oe

#PBS -V

#PBS -l walltime=9000:00:00
###PBS -l nodes=4

#PBS -l nodes=box18:ppn=1+box17:ppn=1+box16:ppn=1+box15:ppn=1
###+box14:ppn=1+box13:ppn=1+box12:ppn=1
###+box10:ppn=1

#!/bin/sh

cd $PBS_O_WORKDIR
###/home/mao1/berczik/phi-GRAPE-cluster-test/run-new
export PATH=$PATH:$PBS_O_WORKDIR
###/home/mao1/berczik/phi-GRAPE-cluster-test/run-new

OUT_FILE=phi-GPU-test.out
>$OUT_FILE

for Nk in 1 2 4 8 16 32 64 128
do
  ./gen-plum.exe ${Nk}
  /usr/local/bin/mpiexec -comm mpich2-pmi -n 4 ./phi-GPU.exe >> $OUT_FILE
###  cp timing.dat timing.dat.${Nk}
  cp contr.dat contr.dat.${Nk}
done

###/usr/local/bin/mpiexec -comm mpich2-pmi -n 4 ./phi-GPU.exe >> $OUT_FILE

###/usr/local/bin/mpiexec -n 8 ./phi-GPU.exe
###/opt/openmpi-intel/bin/mpiexec -machinefile $PBS_NODEFILE -n 4 ./phi-GPU.exe
###/opt/openmpi-intel/bin/mpiexec -n 8 ./phi-GPU.exe

###mpiexec -comm mpich-ib -np 4 ./phi-GRAPE.exe
###>> $OUT_FILE

exit 0
