/**
 * File: ga.cuh
 * Author: Jeanhwea
 * Email: hujinghui@buaa.edu.cn
 */

#ifndef _GA_CUH_
#define _GA_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#define MAX_CHRM_LEN 1000
#define MAX_SHARED_MEM 49152

#define PROB_CROSSOVER (0.8f)
#define PROB_MUTATION (0.005f)

void cuGaEvolve();
void gaAllocMem();
void gaFreeMem();
void dbDisplayWorld();
void dbPrintResult(FILE * out);
void gaEvolve(size_t npop, size_t ngen);
void gaSelection();
void gaStatistics(FILE * out);

static void dbPrintPerson(int * person, size_t n, char * tag);
static int fitvalueCompare(const void *a, const void *b);

__global__ void gaSetPara(size_t npop, size_t ngen, size_t * h_order);
__global__ void gaInit(int * h_chrm, unsigned long * h_hashv, float * h_fitv);
__global__ void gaCrossover(int * h_chrm, unsigned long * h_hashv, float * h_fitv);
__global__ void gaMutation(int * h_chrm, unsigned long * h_hashv, float * h_fitv);

__device__ void crossover(int * dad, int * mom, int * bro, int * sis);
__device__ void mutation(int * person);
__device__ bool swapBits(size_t a, size_t b, int * person);
__device__ unsigned long hashfunc(int * person, size_t num);
__device__ float gaObject(int * person, float * occupy);
__device__ bool check(int * person);
__device__ void scheFCFS(int * person, float * occupy);
__device__ void personMoveForward(int * person, size_t ele, size_t step);
__device__ void fixPerson(int * person);


#endif // ! _GA_CUH_
