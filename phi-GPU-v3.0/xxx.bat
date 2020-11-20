#!/bin/sh
for Nk in 64
### 128 256 512 1024
do
	./gen-plum.exe ${Nk}
	./gpu-4th 2>> gpu-4th.log  1>> gpu-4th.log
	./gpu-6th 2>> gpu-6th.log  1>> gpu-6th.log
done
