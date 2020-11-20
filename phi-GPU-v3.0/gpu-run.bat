#!/bin/sh
for Nk in 8 16 32 64 128
do
	./gen-plum.exe ${Nk} 1
	time ./gpu-4th 2> gpu-4th.log.${Nk}  1> gpu-4th.log.${Nk}
#	time ./gpu-6th 2> gpu-6th.log.${Nk}  1> gpu-6th.log.${Nk}
#	time ./gpu-8th 2> gpu-8th.log.${Nk}  1> gpu-8th.log.${Nk}
done
