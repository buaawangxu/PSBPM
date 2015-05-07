#ifndef _GA_CUH_
#define _GA_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#define MAX_CHRM_LEN 1000
#define MAX_SHARED_MEM 49152

void cuGaEvolve();
void gaAllocMem();
void gaFreeMem();
void dbDisplayWorld();
void gaEvolve(size_t npop, size_t ngen);
static void dbPrintPerson(int * person, size_t n, char * tag);

__global__ void gaSetPara(size_t npop, size_t ngen);
__global__ void gaInit(int * h_chrm, unsigned long * h_hashv, float * h_fitv);
__global__ void gaCrossover(int * h_chrm, unsigned long * h_hashv, float * h_fitv);

__device__ void crossover(int * dad, int * mom, int * bro, int * sis);
__device__ bool swapBits(size_t a, size_t b, int * person);
__device__ unsigned long hashfunc(int * person, size_t num);
__device__ float gaObject(int * person, float * occupy);
__device__ bool check(int * person);
__device__ void scheFCFS(int * person, float * occupy);
__device__ void personMoveForward(int * person, size_t ele, size_t step);
__device__ void fixPerson(int * person);



#endif // ! _GA_CUH_
