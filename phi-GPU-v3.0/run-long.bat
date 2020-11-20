#!/bin/sh

OUT_FILE=phi-GPU.out
###>$OUT_FILE

for Nk in 256
###16 32 64 128
###256 512 1024 2048 4096 8192
do

./gen-plum.exe ${Nk}

for Np in 1 2 4 8 16 32 64
###1
do

/usr/mpi/gcc/openmpi-1.2.6/bin/mpirun -machinefile machines.len.32 -np ${Np} --mca btl openib,self,sm ./phi-GPU.exe  > $OUT_FILE.${Nk}.${Np}

done
done

for Nk in 512
###16 32 64 128
###256 512 1024 2048 4096 8192
do

./gen-plum.exe ${Nk}

for Np in 2 4 8 16 32 64
###1
do

/usr/mpi/gcc/openmpi-1.2.6/bin/mpirun -machinefile machines.len.32 -np ${Np} --mca btl openib,self,sm ./phi-GPU.exe  > $OUT_FILE.${Nk}.${Np}

done
done


for Nk in 1024
###16 32 64 128
###256 512 1024 2048 4096 8192
do

./gen-plum.exe ${Nk}

for Np in 4 8 16 32 64
###1
do

/usr/mpi/gcc/openmpi-1.2.6/bin/mpirun -machinefile machines.len.32 -np ${Np} --mca btl openib,self,sm ./phi-GPU.exe  > $OUT_FILE.${Nk}.${Np}

done
done
