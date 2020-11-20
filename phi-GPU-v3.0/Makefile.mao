CUDA_PATH=/usr/local/cuda
SDK_PATH=/usr/local/cuda_sdk

#CUDA_PATH=/export/opt/cuda
#SDK_PATH=/export/opt/cuda/NVIDIA_CUDA_SDK

#SDK_PATH=/Developer/CUDA

CXXFLAGS = -O3 -Wall -fopenmp

all: gpu-4th gpu-6th

asm: gpu-4th.s gpu-6th.s

cubin: hermite4-gpu.cubin hermite6-gpu.cubin

cpu: cpu-4th cpu-6th

clean:
	rm -f *.o *.s *.cubin

gpu-6th: phi-GPU.cpp hermite6-gpu.o
	mpicxx $(CXXFLAGS) -DSIXTH -DGPU -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib -lcuda -lcudart -o $@ $^

gpu-4th: phi-GPU.cpp hermite4-gpu.o
	mpicxx $(CXXFLAGS) -DFOURTH -DGPU -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib -lcuda -lcudart -o $@ $^

gpu-6th.s: phi-GPU.cpp
	mpicxx $(CXXFLAGS) -DSIXTH -DGPU -I$(CUDA_PATH)/include  -S -o $@ $<

gpu-4th.s: phi-GPU.cpp
	mpicxx $(CXXFLAGS) -DFOURTH -DGPU -I$(CUDA_PATH)/include  -S -o $@ $<

hermite6-gpu.o: hermite6-gpu.cu hermite6-gpu.h
	nvcc -I $(SDK_PATH)/common/inc -Xcompiler "-O3" -c $<

hermite4-gpu.o: hermite4-gpu.cu hermite4-gpu.h
	nvcc -I $(SDK_PATH)/common/inc -Xcompiler "-O3" -c $<

hermite6-gpu.cubin: hermite6-gpu.cu
	nvcc -I $(SDK_PATH)/common/inc -Xcompiler "-O3" -cubin $<

hermite4-gpu.cubin: hermite4-gpu.cu
	nvcc -I $(SDK_PATH)/common/inc -Xcompiler "-O3" -cubin $<

cpu-6th: phi-GPU.cpp hermite6-gpu.o
	mpicxx $(CXXFLAGS) -DSIXTH  -o $@ $^

cpu-4th: phi-GPU.cpp hermite4-gpu.o
	mpicxx $(CXXFLAGS) -DFOURTH -o $@ $^

