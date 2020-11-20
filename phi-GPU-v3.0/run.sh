#!/bin/sh
for Nk in 16 32 64 128 256 512 1024
#for Nk in 256 512 1024
#for Nk in 2048
do
	mpdrun -machinefile hosts.$1 -n 1 ./gen-plum.exe ${Nk}
	mpdrun -machinefile hosts.$1 -n $1 ./gpu-6th 1> ${Nk}.$1.log 2> ${Nk}.$1.err
#cat hosts.$1
#  ./gen-plum.exe ${Nk}
#  scp data.inp g02:berczik/phi-GPU/
#mpdrun -machinefile ~/mpd.hosts -n ${Np} ./gpu-6th 1> ${Nk}.${Np}.log 2> ${Nk}.${Np}.log
done
