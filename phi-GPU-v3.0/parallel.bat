#!/bin/sh
for Nk in 16 32 64 128 256 512 1024
#for Nk in 4
do
  ./gen-plum.exe ${Nk}
  scp data.inp g02:berczik/phi-GPU/
  for Np in 1 2 4 8 16 #32
  do
    mpdrun -machinefile ~/mpd.hosts -n ${Np} ./gpu-6th 1> ${Nk}.${Np}.log 2> ${Nk}.${Np}.log
  done
done
