#define __CUDA
#include <cstdio>
// #include <iostream>
#include <cutil.h>
#include "hermite4-gpu.h"

#define NTHREADS 128

// #define NJBLOCKS 16
// #define NJBLOCKS_ 16

#define NJBLOCKS  28 // GTX470
#define NJBLOCKS_ 32

#define NREDUCE (NTHREADS/NJBLOCKS_)
#define NIBLOCKS 32
#define NIMAX (NTHREADS * NIBLOCKS) // 2048
#define GPU_REDUCE

struct Force_dev{
	float2 acc[3];
	// float2 pot;
	float jrk[3];
	// float pad0;
	float pad[3];
	__device__ Force_dev(){
		// acc[0] = acc[1] = acc[2] = pot = make_float2(0.f, 0.f);
		acc[0] = acc[1] = acc[2] = make_float2(0.f, 0.f);
		jrk[0] = jrk[1] = jrk[2] =  0.f;
	}
};

__device__ void force_reduce(Force_dev &fl, Force_dev &fr){
#pragma unroll
	for(int k=0; k<3; k++){
		fl.acc[k] = float2_accum(fl.acc[k], fr.acc[k].x);
		fl.acc[k] = float2_accum(fl.acc[k], fr.acc[k].y);
		fl.jrk[k] += fr.jrk[k];
	}
	// fl.pot = float2_accum(fl.pot, fr.pot.x);
	// fl.pot = float2_accum(fl.pot, fr.pot.y);
}

__device__ void h4_kernel(
		const Predictor &ip,
		const Predictor &jp,
		Force_dev &fo,
		float eps2){
#if 0
	float dx = float2_sub(jp.pos[0], ip.pos[0]);
	float dy = float2_sub(jp.pos[1], ip.pos[1]);
	float dz = float2_sub(jp.pos[2], ip.pos[2]);

	float dvx = jp.vel[0] - ip.vel[0];
	float dvy = jp.vel[1] - ip.vel[1];
	float dvz = jp.vel[2] - ip.vel[2];

	float dax = jp.acc[0] - ip.acc[0];
	float day = jp.acc[1] - ip.acc[1];
	float daz = jp.acc[2] - ip.acc[2];
#else
	float dx = (jp.posH.x - ip.posH.x) + (jp.posL.x - ip.posL.x);
	float dy = (jp.posH.y - ip.posH.y) + (jp.posL.y - ip.posL.y);
	float dz = (jp.posH.z - ip.posH.z) + (jp.posL.z - ip.posL.z);

	float dvx = jp.vel.x - ip.vel.x;
	float dvy = jp.vel.y - ip.vel.y;
	float dvz = jp.vel.z - ip.vel.z;
#endif

	float r2 = eps2 + dx*dx + dy*dy + dz*dz;
	float drdv =  dx*dvx +  dy*dvy +  dz*dvz;

	float rinv1 = rsqrtf(r2);
	float rinv2 = rinv1 * rinv1;
	float alpha = (drdv)*rinv2;
	// rinv1 *= jp.mass;
	rinv1 *= jp.posH.w;
	float rinv3 = rinv1 * rinv2;

	// float pot = rinv1;
	float ax = rinv3*dx;
	float ay = rinv3*dy;
	float az = rinv3*dz;
	float jx = rinv3*dvx + (-3.f*alpha)*ax;
	float jy = rinv3*dvy + (-3.f*alpha)*ay;
	float jz = rinv3*dvz + (-3.f*alpha)*az;

#if 0
	if(r2 != eps2){
		fo.pot = float2_accum(fo.pot, pot);
	}
#endif
	fo.acc[0] = float2_accum(fo.acc[0], ax);
	fo.acc[1] = float2_accum(fo.acc[1], ay);
	fo.acc[2] = float2_accum(fo.acc[2], az);
	fo.jrk[0] += jx;
	fo.jrk[1] += jy;
	fo.jrk[2] += jz;
}

__global__ void h4_gravity(
		int ni,
		int nj,
		Predictor ipred[],
		Predictor jpred[],
		Force_dev force[][NJBLOCKS_],
		float eps2){
	int ibid = blockIdx.x;
	int jbid = blockIdx.y;
	int tid = threadIdx.x;
	int iaddr = tid + NTHREADS * ibid;
	int jstart = (nj * (jbid  )) / NJBLOCKS;
	int jend   = (nj * (jbid+1)) / NJBLOCKS;

	// small kernel opt
	int nskip = 1;
	int niloc = ni - NTHREADS * ibid;
	if(niloc <= NTHREADS/2) nskip = 2;
	if(niloc <= NTHREADS/4) nskip = 4;
	if(niloc <= NTHREADS/8) nskip = 8;
	if(niloc <= NTHREADS/16) nskip = 16;
	if(niloc <= NTHREADS/32) nskip = 32;
	int joff = tid / (NTHREADS/nskip);

	__shared__ Predictor jpshare[NTHREADS];
	Force_dev fo;
	Predictor ip = ipred[tid % (NTHREADS/nskip) + NTHREADS * ibid];
	for(int j=jstart; j<jend; j+=NTHREADS){
		__syncthreads();
#if 0
		jpshare[tid] = jpred[j+tid];
#else
		float4 *src = (float4 *)&jpred[j];
		float4 *dst = (float4 *)jpshare;
		for(int it=0; it<sizeof(Predictor)/sizeof(float4); it++){
			dst[tid] = src[tid];
			dst += NTHREADS;
			src += NTHREADS;
		}
#endif
		__syncthreads();
		if(jend-j < NTHREADS){
			for(int jj=0; jj<jend-j; jj+=nskip){
				Predictor &jp = jpshare[jj+joff];
				if(jj+joff < jend-j) h4_kernel(ip, jp, fo, eps2);
			}
		}else{
#if 0
#pragma unroll
			for(int jj=0; jj<NTHREADS; jj+=nskip){
				Predictor &jp = jpshare[jj+joff];
				h6_kernel(ip, jp, fo, eps2);
			}
#else
			for(int jj=0; jj<NTHREADS; jj+=4*nskip){
				Predictor &jp0 = jpshare[0*nskip+jj+joff];
				Predictor &jp1 = jpshare[1*nskip+jj+joff];
				Predictor &jp2 = jpshare[2*nskip+jj+joff];
				Predictor &jp3 = jpshare[3*nskip+jj+joff];
				h4_kernel(ip, jp0, fo, eps2);
				h4_kernel(ip, jp1, fo, eps2);
				h4_kernel(ip, jp2, fo, eps2);
				h4_kernel(ip, jp3, fo, eps2);
			}
#endif
		}
	}
	// horizontal reduce
	// __shared__ Force_dev foshare[NTHREADS];
	Force_dev *foshare = (Force_dev *)jpshare;
	__syncthreads();
	foshare[tid] = fo;
	__syncthreads();
	if(nskip > 1){
		if(tid < NTHREADS/2){
			force_reduce(foshare[tid], foshare[tid + NTHREADS/2]);
		}
		__syncthreads();
	}
	if(nskip > 2){
		if(tid < NTHREADS/4){
			force_reduce(foshare[tid], foshare[tid + NTHREADS/4]);
		}
		__syncthreads();
	}
	if(nskip > 4){
		if(tid < NTHREADS/8){
			force_reduce(foshare[tid], foshare[tid + NTHREADS/8]);
		}
		__syncthreads();
	}
	if(nskip > 8){
		if(tid < NTHREADS/16){
			force_reduce(foshare[tid], foshare[tid + NTHREADS/16]);
		}
		__syncthreads();
	}
	if(nskip > 16){
		if(tid < NTHREADS/32){
			force_reduce(foshare[tid], foshare[tid + NTHREADS/32]);
		}
		__syncthreads();
	}
	// store
	if(tid < niloc){
		fo = foshare[tid];
		force[iaddr][jbid] = fo;
	}
}

#ifdef GPU_REDUCE
__global__ void reduce_kernel(
		Force_dev fo_dev[][NJBLOCKS_],
		Force_dev fo_reduce[])
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int ioff = bid * NREDUCE;
#if 0
	__shared__ Force_dev fo_share[NTHREADS];
#else
	__shared__ Predictor jpshare[NTHREADS];
	Force_dev *fo_share = (Force_dev *)jpshare;
#endif
#if 0
	fo_share[tid] = fo_dev[ioff][tid];
#else
	float4 *src = (float4 *)fo_dev[ioff];
	float4 *dst = (float4 *)fo_share;
	for(int it=0; it<sizeof(Force_dev)/sizeof(float4); it++){
		dst[tid] = src[tid];
		dst += NTHREADS;
		src += NTHREADS;
	}
#endif
	__syncthreads();

	int n = NJBLOCKS_;
	while(n > 1){
		n /= 2;
		if(tid % NJBLOCKS_ < n){
			force_reduce(fo_share[tid], fo_share[tid + n]);
		}
	}
	__syncthreads();

	if(tid % NJBLOCKS_ == 0){
		// fo_reduce[ioff + tid / NJBLOCKS_] = fo_share[tid];
		fo_share[tid / NJBLOCKS_] = fo_share[tid];
	}
	__syncthreads();
#if 0
	if(tid < NREDUCE){
		fo_reduce[ioff + tid] = fo_share[tid];
	}
#else
	if(tid < NREDUCE * sizeof(Force_dev) / sizeof(float)){ // (tid < 96)
		float *dst = (float *)&fo_reduce[ioff];
		float *src = (float *)fo_share;
		dst[tid] = src[tid];
	}
#endif
}
#endif

extern double wtime();

void calc_force(
		int nitot, 
		int nj, 
		float eps2,
		Predictor ipred[],
		Predictor jpred[],
		Force     force[],
		double &t1,
		double &t_send,
		double &t_recv){
	static Predictor *jp_dev = NULL;
	static Predictor *ip_dev = NULL;
	static Force_dev (*fo_dev)[NJBLOCKS_] = NULL;
#ifdef GPU_REDUCE
	static Force_dev (*fo_reduce) = NULL;
	static Force_dev (*fo_host) = NULL;
#else
	static Force_dev (*fo_host)[NJBLOCKS_] = NULL;
#endif

	if(jp_dev == NULL){ // first call
		/*
		   const int dev = 0;
		   CUDA_SAFE_CALL(cudaSetDevice(dev));
		   cudaDeviceProp deviceProp;
		   CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
		   printf("GPU: %s\n", deviceProp.name);
		 */
		cudaMalloc((void **)&jp_dev, (nj + NTHREADS) * sizeof(Predictor));
		cudaMalloc((void **)&ip_dev, NIMAX * sizeof(Predictor));
		cudaMalloc((void **)&fo_dev, NIMAX * sizeof(*fo_dev));
		cudaMemset(fo_dev, 0, NIMAX * sizeof(*fo_dev));
#ifdef GPU_REDUCE
		cudaMalloc((void **)&fo_reduce, NIMAX * sizeof(*fo_reduce));
#endif
		cudaMallocHost((void **)&fo_host, NIMAX * sizeof(*fo_host));
	}

	cudaMemcpy(jp_dev, jpred, nj * sizeof(Predictor), cudaMemcpyHostToDevice);
	t1 = wtime();

	int nimax = NIMAX;
	for(int ioff=0; ioff<nitot; ioff+=nimax){
		int ni = std::min(nimax, nitot-ioff);
		double t2 = wtime();
		cudaMemcpy(ip_dev, ipred+ioff, ni * sizeof(Predictor), cudaMemcpyHostToDevice);
		double t3 = wtime();
		t_send += t3 - t2;
		// kernel call
		int niblocks = 1 + (ni-1) / NTHREADS;
		dim3 grid(niblocks, NJBLOCKS, 1);
		dim3 threads(NTHREADS, 1, 1);
		// std::cerr << "call h6_gravity " << niblocks << std::endl;
		//int sharedMemSize = NTHREADS * sizeof(Predictor);
		// h6_gravity <<< grid, threads, sharedMemSize >>>
		//	(ni, nj, ip_dev, jp_dev, fo_dev, eps2);
		h4_gravity <<< grid, threads >>>
			(ni, nj, ip_dev, jp_dev, fo_dev, eps2);

#ifdef GPU_REDUCE
		dim3 grid_reduce(1 + (ni-1)/NREDUCE, 1, 1);
		reduce_kernel <<< grid_reduce, threads >>> (fo_dev, fo_reduce);
		cudaThreadSynchronize();
		double t4 = wtime();
		cudaMemcpy(fo_host, fo_reduce, ni * sizeof(*fo_reduce), cudaMemcpyDeviceToHost);
		double t5 = wtime();
		t_recv += t5 - t4;
		for(int i=0; i<ni; i++){
			Force f; // 0 flashed by the constructer
			Force_dev &fo = fo_host[i];
			f.acc.x = float2_reduce(fo.acc[0]);
			f.acc.y = float2_reduce(fo.acc[1]);
			f.acc.z = float2_reduce(fo.acc[2]);
			// f.pot   = float2_reduce(fo.pot);
			f.jrk.x = fo.jrk[0];
			f.jrk.y = fo.jrk[1];
			f.jrk.z = fo.jrk[2];
			force[ioff + i] = f;
		}
#else
		cudaMemcpy(fo_host, fo_dev, ni * sizeof(*fo_dev), cudaMemcpyDeviceToHost);
		// std::cerr << "done" << std::endl;
		for(int i=0; i<ni; i++){
			Force f; // 0 flashed by the constructer
			for(int jb=0; jb<NJBLOCKS; jb++){
				Force_dev &fo = fo_host[i][jb];
				f.acc.x += float2_reduce(fo.acc[0]);
				f.acc.y += float2_reduce(fo.acc[1]);
				f.acc.z += float2_reduce(fo.acc[2]);
				f.pot   -= float2_reduce(fo.pot);
				f.jrk.x += fo.jrk[0];
				f.jrk.y += fo.jrk[1];
				f.jrk.z += fo.jrk[2];
				f.snp.x += fo.snp[0];
				f.snp.y += fo.snp[1];
				f.snp.z += fo.snp[2];
			}
			force[ioff + i] = f;
		}
#endif
	}
}

__global__ void pot_kernel(
		int js,
		int je,
		float eps2,
		Posm posm[],
		float2 pot[]){
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int iaddr = tid + NTHREADS * bid;
	Posm ip = posm[iaddr];
	float2 poti = make_float2(0.f, 0.f);
	for(int j=js; j<je; j+=NTHREADS){
		__shared__ Posm posmshare[NTHREADS];
		__syncthreads();
		posmshare[tid] = posm[j + tid];
		__syncthreads();
		int njj = NTHREADS < je-j ? NTHREADS : je-j;
		for(int jj=0; jj< njj; jj++){
			Posm &jp = posmshare[jj];
			float dx = float2_sub(jp.pos[0], ip.pos[0]);
			float dy = float2_sub(jp.pos[1], ip.pos[1]);
			float dz = float2_sub(jp.pos[2], ip.pos[2]);
			float r2 = eps2 + dx*dx + dy*dy + dz*dz;
			float mrinv = jp.mass * rsqrtf(r2);
			if(r2 > eps2) poti = float2_accum(poti, mrinv);
		}
	}
	pot[iaddr] = poti;
}

void calc_pot(
		int ni,
		int js,
		int je,
		float eps2,
		Posm posm[],
		double dpot[]){
	Posm *posm_dev;
	float2 *pot, *pot_dev;
	cudaMalloc((void **)&posm_dev, (ni+NTHREADS) * sizeof(Posm));
	cudaMalloc((void **)&pot_dev, (ni+NTHREADS) * sizeof(float2));
	cudaMallocHost((void **)&pot, (ni+NTHREADS) * sizeof(float2));

	cudaMemcpy(posm_dev, posm, ni * sizeof(Posm), cudaMemcpyHostToDevice);

	int nblocks = 1 + (ni-1) / NTHREADS;
	dim3 grid(nblocks, 1, 1);
	dim3 threads(NTHREADS, 1, 1);
	int sharedMemSize = NTHREADS * sizeof(Posm);

	pot_kernel <<< grid, threads, sharedMemSize >>>
		(js, je, eps2, posm_dev, pot_dev);

	cudaMemcpy(pot, pot_dev, ni * sizeof(float2), cudaMemcpyDeviceToHost);
	for(int i=0; i<ni; i++){
		dpot[i] = -float2_reduce(pot[i]);
	}

	cudaFree(posm_dev);
	cudaFree(pot_dev);
	cudaFreeHost(pot);
}

void CUDA_MPI_Init(int myRank){
	int numGPU;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&numGPU));
	const int dev = myRank % numGPU;
	CUDA_SAFE_CALL(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Rank %02d : GPU %d : %s\n", myRank, dev, deviceProp.name); 

	cudaFuncSetCacheConfig(h4_gravity,    cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(reduce_kernel, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(pot_kernel,    cudaFuncCachePreferShared);
}
