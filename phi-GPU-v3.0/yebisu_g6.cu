#include <cutil.h>
#include <omp.h>
#include "cuda_pointer.h"

enum{
	MAX_GPU    = 4,
	MAX_CPU    = 8,
	NBODY_MAX  = (1<<18),
	NB_MAX     = 256,     // per block
	MAX_NB_BUF = (1<<18),
};

#include "gtx470.h"

#include "particle.h"

#define _out_

__global__ void kernel_jp_scatter(
		const int nj,
		const Jparticle jpsrc[],
		_out_ Jparticle jpdst[])
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < nj){
		const Jparticle jp = jpsrc[tid];
		jpdst[jp.addr] = jp;
	}
}

__global__ void kernel_predict(
		const int       nj,
		const float2    ti,
		const Jparticle jptcl[],
		_out_ Jppred    jpred[])
{
#if 0
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < nj){
		jpred[tid].predict(jptcl[tid], ti);
	}
#else
	const int tid = threadIdx.x;
	const int off = blockDim.x * blockIdx.x;
	const int nth = blockDim.x;
	__shared__ float4 sbuf[NTHREADS*5];
	Jparticle *sptcl = (Jparticle *)sbuf;
	Jppred    *spred = (Jppred    *)sbuf;

	{   // LOAD
		float4 *src = (float4 *)(jptcl + off);
		float4 *dst = (float4 *)(sptcl);
#pragma unroll
		for(int k=0; k<5; k++, src+=nth, dst+=nth){
			dst[tid] = src[tid];
		}
	}

	// Predict
	__syncthreads();
	Jppred pp;
	pp.predict(sptcl[tid], ti);
	__syncthreads();
	spred[tid] = pp;
	__syncthreads();

	{   // STORE
		float4 *src = (float4 *)(spred);
		float4 *dst = (float4 *)(jpred + off);
#pragma unroll
		for(int k=0; k<3; k++, src+=nth, dst+=nth){
			dst[tid] = src[tid];
		}
	}
#endif
}

#define INTERACTION Interaction_NB
__global__ void kernel_gravity(
		const int ni,
		const int nj,
		const Iparticle ipbuf[],
		const Jppred    jpbuf[],
		_out_ Force     fodev[][NJBLOCKS],
		_out_ int       nbbuf[][NJBLOCKS][NB_MAX],
		const bool      with_neib)
{
	int ibid = blockIdx.x;
	int jbid = blockIdx.y;
	int tid = threadIdx.x;
	int iaddr = tid + blockDim.x * ibid;
	int jstart = (nj * (jbid  )) / NJBLOCKS;
	int jend   = (nj * (jbid+1)) / NJBLOCKS;
	int *nbdst = nbbuf[iaddr][jbid];

	__shared__ Jppred jpshare[NJPSHRE]; // 32

	const Iparticle ip = ipbuf[iaddr];
	Force fo;
	fo.clear();
	
	if(with_neib){
		for(int j=jstart; j<jend; j+=NJPSHRE){
			const int jsize = NJPSHRE * Jppred::SIZE_F4; // 96

			__syncthreads();
			if(tid < jsize){ // 96 of 128
				float4 *src = (float4 *)(jpbuf + j);
				float4 *dst = (float4 *)(jpshare  );
				dst[tid] = src[tid];
			}
			if(tid+32 < jsize){ // for the case of 64 threads
				float4 *src = (float4 *)(jpbuf + j);
				float4 *dst = (float4 *)(jpshare  );
				dst[tid+32] = src[tid+32];
			}
			__syncthreads();

			if(jend-j < NJPSHRE){
#pragma unroll 4
				for(int jj=0; jj<jend-j; jj++){
					const Jppred jp = jpshare[jj];
					const Interaction_NB inter(ip, jp);
					inter.set_neib(nbdst[fo.num_neib % NB_MAX]);
					fo += inter;
				}
			}else{
#pragma unroll 32
				for(int jj=0; jj<NJPSHRE; jj++){
					const Jppred jp = jpshare[jj];
					const Interaction_NB inter(ip, jp);
					inter.set_neib(nbdst[fo.num_neib % NB_MAX]);
					fo += inter;
				}
			}
		}
	}else{ // no neib
		for(int j=jstart; j<jend; j+=NJPSHRE){
			const int jsize = NJPSHRE * Jppred::SIZE_F4; // 96

			__syncthreads();
			if(tid < jsize){ // 96 of 128
				float4 *src = (float4 *)(jpbuf + j);
				float4 *dst = (float4 *)(jpshare  );
				dst[tid] = src[tid];
			}
			if(tid+32 < jsize){ // for the case of 64 threads
				float4 *src = (float4 *)(jpbuf + j);
				float4 *dst = (float4 *)(jpshare  );
				dst[tid+32] = src[tid+32];
			}
			__syncthreads();

			if(jend-j < NJPSHRE){
#pragma unroll 4
				for(int jj=0; jj<jend-j; jj++){
					const Jppred jp = jpshare[jj];
					const Interaction inter(ip, jp);
					inter.set_neib(nbdst[fo.num_neib % NB_MAX]);
					fo += inter;
				}
			}else{
#pragma unroll 32
				for(int jj=0; jj<NJPSHRE; jj++){
					const Jppred jp = jpshare[jj];
					const Interaction inter(ip, jp);
					inter.set_neib(nbdst[fo.num_neib % NB_MAX]);
					fo += inter;
				}
			}
		}
	}

	if(iaddr < ni){
		fodev[iaddr][jbid] = fo;
	}
}

__global__ void kernel_reduce(
		const int ni,
		const Force fodev[][NJBLOCKS],
		_out_ Force fosum[]){
	const int xid = threadIdx.x;
	const int yid = threadIdx.y;
	const int bid = blockIdx.x;
	const int iaddr = yid + blockDim.y * bid;

	__shared__ Force fshare[NYREDUCE][NXREDUCE];
	if(xid < NJBLOCKS){
		fshare[yid][xid] = fodev[iaddr][xid];
	}else{
		fshare[yid][xid].clear();
	}
	Force *fs = fshare[yid];

	if(32 == NXREDUCE){
		if(xid < 16) fs[xid] += fs[xid + 16];
	}
	if(xid < 8) fs[xid] += fs[xid + 8];
	if(xid < 4) fs[xid] += fs[xid + 4];
	if(xid < 2) fs[xid] += fs[xid + 2];
	if(xid < 1) fs[xid] += fs[xid + 1];
	
	if(iaddr < ni && 0 == xid){
		fosum[iaddr] = fs[0];
	}
}

__global__ void kernel_gather_nb(
		const int   ni,
		const Force fodev[][NJBLOCKS],
		const int2  nbcnt[],
		const int   nbbuf[][NJBLOCKS][NB_MAX],
		_out_ int   nblst[])
{
	const int xid = threadIdx.x;
	const int yid = threadIdx.y;
	const int bid = blockIdx.x;
	const int iaddr = yid + blockDim.y * bid;
	if(iaddr >= ni) return;
	if(nbcnt[iaddr].x < 0) return; // overflow

	const int mynnb = (xid < NJBLOCKS) ? fodev[iaddr][xid].num_neib
	                                   : 0;

	// now performe prefix sum
	__shared__ int ishare[NYREDUCE][NXREDUCE];
	ishare[yid][xid] = mynnb;
	int *ish = ishare[yid];
	if(xid>=1)  ish[xid] += ish[xid-1];
	if(xid>=2)  ish[xid] += ish[xid-2];
	if(xid>=4)  ish[xid] += ish[xid-4];
	if(xid>=8)  ish[xid] += ish[xid-8];
	if(32 == NXREDUCE){
		if(xid>=16)  ish[xid] += ish[xid-16];
	}

	const int off = (xid == 0) ? 0 
	                           : ish[xid-1];
	int *nbdst = nblst + nbcnt[iaddr].y + off;
	if(xid < NJBLOCKS){
		for(int k=0; k<mynnb; k++){
			const int nbid = nbbuf[iaddr][xid][k];
			nbdst[k] = nbid;
		}
	}
}

class Resource{
private:
	bool   is_open;
	bool   predicted;
	bool   grav_called;
	bool   jp_flushed;
	int    gpid;
	int    njp_in_que;
	int    ni_save;
	float2 ti;

	cudaPointer<Iparticle>              ipbuf;
	cudaPointer<Jparticle>              jpbuf;
	cudaPointer<Jparticle>              jpque;
	cudaPointer<Jppred>                 jpred;
	cudaPointer <Force[NJBLOCKS]>       fodev;
	cudaPointer <Force>                 fosum;
	cudaPointer <int[NJBLOCKS][NB_MAX]> nbbuf;
	cudaPointer <int>                   nblst;
	cudaPointer <int2>                  nbcnt; // {num, off}

	void allocate(){
		ipbuf.allocate(NIMAX);
		jpbuf.allocate(NBODY_MAX);
		jpque.allocate(NBODY_MAX);
		jpred.allocate(NBODY_MAX);
		fodev.allocate(NIMAX);
		fosum.allocate(NIMAX);
		nbbuf.allocate(NIMAX);
		nblst.allocate(MAX_NB_BUF);
		nbcnt.allocate(NIMAX);
	}
	void free(){
		ipbuf.free();
		jpbuf.free();
		jpque.free();
		jpred.free();
		fodev.free();
		fosum.free();
		nbbuf.free();
		nblst.free();
		nbcnt.free();
	}
public:
	void set_gpid(const int id){
		gpid = id;
		int dev;
		cudaGetDevice(&dev);
		assert(dev == gpid);
	}
	void open(){
		assert(!is_open);
		allocate();
		is_open     = true;
		njp_in_que  = 0;
		predicted   = false;
		grav_called = false;
		jp_flushed  = false;
	}
	void close(){
		assert(is_open);
		free();
		is_open = false;
	}
	void set_ip(
			const int    ni,
			const double pos[][3],
			const double vel[][3],
			const double eps2[],
			const double h2  [],
			const int    id  [])
	{
		assert(is_open);
		assert(ni <= NIMAX);
		
		for(int i=0; i<ni; i++){
			ipbuf[i].read(pos[i], vel[i], eps2[i], h2[i], id[i]);
		}
		ipbuf.htod(ni);
		ni_save = ni;
	}
	void push_jp(
			const double pos [3],
			const double vel [3],
			const double acc2[3],
			const double jrk6[6],
			const double mass,
			const double tj,
			const int    id,
			const int    addr)
	{
		assert(is_open);
		assert(addr < NBODY_MAX);

		jpque[njp_in_que].read(pos, vel, acc2, jrk6, mass, tj, id, addr);
		njp_in_que++;
		jp_flushed = false;
	}
	void transter_jp(){
		assert(is_open);

		const int njq = njp_in_que;
		jpque.htod(njq);
		njp_in_que = 0;
		const int Blocks = 1 + (njq-1)/NTHSCAT;
		kernel_jp_scatter <<< Blocks, NTHSCAT >>>
			(njq, jpque, jpbuf);
		jp_flushed = true;
		predicted  = false;
	}
	void set_ti(const double dbl_ti){
		assert(is_open);

		ti = float2_split(dbl_ti);
		predicted = false;
	}
	void predict_all(const int nj){
		assert(is_open);
		
		const int Blocks = 1 + (nj-1)/NTHREADS;
		kernel_predict <<< Blocks, NTHREADS >>>
			(nj, ti, jpbuf, jpred);
		predicted = true;
	}
	void launch_gravity(
			const int  ni,
			const int  nj,
			const bool with_neib)
	{
		assert(is_open);
		assert(ni == ni_save);
		assert(ni <= NIMAX);
		assert(nj < NBODY_MAX);

		if(!jp_flushed) transter_jp();
		if(!predicted ) predict_all(nj);
		if(ni <= 64){
			dim3 grid   ( 1, NJBLOCKS, 1);
			dim3 threads(64,        1, 1);
			kernel_gravity <<< grid, threads >>>
				(ni, nj, ipbuf, jpred, fodev, nbbuf, with_neib);
		}else{
			const int niblocks = 1 + (ni-1) / NTHREADS;
			dim3 grid   (niblocks, NJBLOCKS, 1);
			dim3 threads(NTHREADS,        1, 1);
			kernel_gravity <<< grid, threads >>>
				(ni, nj, ipbuf, jpred, fodev, nbbuf, with_neib);
		}
		grav_called = true;
	}
	void get_force(
			const int    ni,
			_out_ double acc   [][3],
			_out_ double jrk   [][3],
			_out_ double pot   [],
			_out_ int    nnb_id[])
	{
		assert(is_open);
		assert(grav_called);
		assert(ni == ni_save);
		assert(ni <= NIMAX);

		const int ni8 = 1 + (ni-1) / NYREDUCE;
		dim3 grid   (ni8, 1, 1);
		dim3 threads(NXREDUCE, NYREDUCE, 1);
		kernel_reduce <<< grid, threads >>>
			(ni, fodev, fosum);
		fosum.dtoh(ni);
		grav_called = false;

		for(int i=0; i<ni; i++){
			fosum[i].write(acc[i], jrk[i], pot[i], nnb_id[i], nbcnt[i].x);
		}
	}
	void receive_neighbor_list(){
		assert(is_open);

		const int ni = ni_save;
		int nbsum = 0;
		for(int i=0; i<ni; i++){
			nbcnt[i].y = nbsum;
			if(nbcnt[i].x >= 0) nbsum += nbcnt[i].x;
		}
		assert(nbsum <= MAX_NB_BUF);
		nbcnt.htod(ni);

		const int ni8 = 1 + (ni-1) / NYREDUCE;
		dim3 grid   (ni8, 1, 1);
		dim3 threads(NXREDUCE, NYREDUCE, 1);
		kernel_gather_nb <<< grid, threads >>>
			(ni, fodev, nbcnt, nbbuf, nblst);
		nblst.dtoh(nbsum);
	}
	void get_neighbor_list(
			const int ipipe,
			const int maxlen,
			_out_ int *num_neib,
			_out_ int list[])
	{
		assert(is_open);
		assert(ipipe < ni_save);

		const int nnb = nbcnt[ipipe].x;
		const int off = nbcnt[ipipe].y;
		const int *src = &nblst[off];
		if(nnb > 0 && maxlen >= nnb){
			for(int k=0; k<nnb; k++){
				list[k] = src[k];
			}
			*num_neib = nnb;
		}else{
			*num_neib = -1;
		}
	}

	void DEBUG_read_pred(
			const int    nj,
			const int    addr,
			_out_ double pos [3],
			_out_ double vel [3],
			_out_ double mass[1],
			_out_ int    id  [1])
	{
		jpred.dtoh(nj);
		const Jppred &p = jpred[addr];
		for(int k=0; k<3; k++){
			pos[k] = p.pos[k].x + p.pos[k].y;
			vel[k] = p.vel[k];
		}
		mass[0] = p.mass;
		id  [0] = p.id;
	}

};

static Resource resource[MAX_GPU];
static int numGPU, numCPU;
static bool initialized = false;

static void lib_initialize(){
	if(initialized) return;
	initialized = true;

	assert(NXREDUCE >= NJBLOCKS);
	assert(NXREDUCE <= 32);
	assert(sizeof(Jppred) % sizeof(float4) == 0);
	assert(sizeof(Jppred) / sizeof(float4) == Jppred::SIZE_F4);
	assert(NJPSHRE * Jppred::SIZE_F4 <= NTHREADS);

	int devid[MAX_GPU];
	cudaGetDeviceCount(&numGPU);
	assert(numGPU <= MAX_GPU);
	char *gpu_list = getenv("GPU_LIST");
	if(gpu_list){
		// get GPU list from environment variable
		numGPU = 0;
		char *p = strtok(gpu_list, " ");
		while(p){
			devid[numGPU++] = atoi(p);
			p = strtok(NULL, " ");
			assert(numGPU <= MAX_GPU);
		}
	}else{
		// use all GPUs
		for(int i=0; i<numGPU; i++){
			devid[i] = i;
		}
	}

	// numGPU = 1;
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		if(tid == 0) numCPU = omp_get_num_threads();
	}
	assert(numCPU <= MAX_CPU);
	assert(numGPU <= numCPU);
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		if(tid < numGPU){
			cudaSetDevice(devid[tid]);
			resource[tid].set_gpid(devid[tid]);
		}
	}
	fprintf(stderr, "***********************\n");
	fprintf(stderr, "Initializing Yebisu/G6 library\n");
	fprintf(stderr, "#CPU %d, #GPU %d\n", numCPU, numGPU);
	fprintf(stderr, " device:");
	for(int i=0; i<numGPU; i++){
		fprintf(stderr, " %d", devid[i]);
	}
	fprintf(stderr, "\n");
	fprintf(stderr, "***********************\n");

#if 1
	cudaFuncSetCacheConfig(kernel_jp_scatter, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernel_predict,    cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(kernel_gravity,    cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernel_reduce,     cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(kernel_gather_nb,  cudaFuncCachePreferL1);
#endif
}

#include "yebisu_g6.h"
extern "C"{
	void yebisu_g6_open (const int gpid){
		lib_initialize();
		resource[gpid].open();
	}
	void yebisu_g6_close(const int gpid){
		lib_initialize();
		resource[gpid].close();
	}
	void yebisu_g6_set_ip(
			const int    gpid,
			const int    ni,
			const double pos[][3],
			const double vel[][3],
			const double eps2[],
			const double h2  [],
			const int    id  [])
	{
		lib_initialize();
		resource[gpid].set_ip(ni, pos, vel, eps2, h2, id);
	}
	void yebisu_g6_push_jp(
			const int    gpid,
			const double pos [3],
			const double vel [3],
			const double acc2[3],
			const double jrk6[6],
			const double mass,
			const double tj,
			const int    id,
			const int    addr)
	{
		lib_initialize();
		resource[gpid].push_jp( pos, vel, acc2, jrk6, mass, tj, id, addr);
	}
	void yebisu_g6_transfer_jp(const int gpid){
		lib_initialize();
		resource[gpid].transter_jp();
	}
	void yebisu_g6_set_ti(
			const int    gpid,
			const double ti)
	{
		lib_initialize();
		resource[gpid].set_ti(ti);
	}
	void yebisu_g6_predict_all(
			const int gpid,
			const int nj)
	{
		lib_initialize();
		resource[gpid].predict_all(nj);
	}
	void yebisu_g6_launch_gravity(
			const int gpid,
			const int ni,
			const int nj,
			const int with_neib)
	{
		lib_initialize();
		resource[gpid].launch_gravity(ni, nj, bool(with_neib));
	}
	void yebisu_g6_get_force(
			const int    gpid,
			const int    ni,
			_out_ double acc   [][3],
			_out_ double jrk   [][3],
			_out_ double pot   [],
			_out_ int    nnb_id[])
	{
		lib_initialize();
		resource[gpid].get_force(ni, acc, jrk, pot, nnb_id);
	}
	void yebisu_g6_receive_neighbor_list(const int gpid){
		lib_initialize();
		resource[gpid].receive_neighbor_list();
	}
	void yebisu_g6_get_neighbor_list(
			const int gpid,
			const int ipipe,
			const int maxlen,
			_out_ int *num_neib,
			_out_ int list[])
	{
		lib_initialize();
		resource[gpid].get_neighbor_list(ipipe, maxlen, num_neib, list);
	}

	void yebisu_g6_DEBUG_read_pred(
			const int    gpid,
			const int    nj,
			const int    addr,
			_out_ double pos [3],
			_out_ double vel [3],
			_out_ double mass[1],
			_out_ int    id  [1])
	{
		resource[gpid].DEBUG_read_pred(nj, addr, pos, vel, mass, id);
	}

	int yebisu_g6_get_nimax(){
		return NIMAX;
	}

	int yebisu_g6_get_njmax(){
		return NBODY_MAX;
	}
}
