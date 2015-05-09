#ifndef _HELPER_CUH_
#define _HELPER_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

#include "data.h"

const size_t npop = 200;
const size_t ngen = 100;

#define INF_DURATION FLT_MAX

void allocMemOnDevice();
void freeMemOnDevice();
void moveDataToDevice();

#define IS_DEPEND(ipred, isucc) \
    (d_depd[((ipred)-1)+((isucc)-1)*d_ntask])

#define IS_ASSIGN(itask, ireso) \
    (d_asgn[((itask)-1)+((ireso)-1)*d_ntask])

#define DURATION(itask) \
    (d_dura[(itask)-1])

__device__ size_t randInt(const size_t min, const size_t max);
__device__ float randProb();__device__ void clearResouceOccupy(float * occupy);
__device__ float allocResouce(size_t ireso, float duration, float * occupy);
__device__ float getTotalOccupy(size_t ireso, float * occupy);
__device__ float getMaxTotalOccupy(float * occupy);

// ------------------- here are implemention -------------------


// -- data structure --
__device__ size_t d_ntask;
__device__ size_t d_nreso;

// host handles
float * h_dura;
bool  * h_depd;
bool  * h_asgn;
curandState_t * h_states;

// device handles
__device__ float * d_dura;
__device__ bool  * d_depd;
__device__ bool  * d_asgn;
__device__ curandState_t * d_states;

// -- functions --
__global__ void setupRandSeed(unsigned long long seed);
__global__ void setInitPara(size_t ntask, size_t nreso, float * h_dura, bool * h_depd, bool * h_asgn, curandState_t * h_states);

void allocMemOnDevice()
{
    size_t m_size;

    m_size = ntask * sizeof(float);
    checkCudaErrors(cudaMalloc((void **)&h_dura, m_size));
    m_size = ntask * ntask * sizeof(bool);
    checkCudaErrors(cudaMalloc((void **)&h_depd, m_size));
    m_size = ntask * nreso * sizeof(bool);
    checkCudaErrors(cudaMalloc((void **)&h_asgn, m_size));

    m_size = npop * sizeof(curandState_t);
    checkCudaErrors(cudaMalloc((void **)&h_states, m_size));
}

void freeMemOnDevice()
{
    checkCudaErrors(cudaFree(h_dura));
    checkCudaErrors(cudaFree(h_depd));
    checkCudaErrors(cudaFree(h_asgn));

    checkCudaErrors(cudaFree(h_states));
}

__global__ void setInitPara(size_t ntask, size_t nreso, float * h_dura, bool * h_depd, bool * h_asgn, curandState_t * h_states)
{
    d_ntask = ntask;
    d_nreso = nreso;

    d_dura = h_dura;
    d_depd = h_depd;
    d_asgn = h_asgn;

    d_states = h_states;
}

void moveDataToDevice()
{
    size_t m_size;

    m_size = ntask * sizeof(float);
    checkCudaErrors(cudaMemcpy(h_dura, dura, m_size, cudaMemcpyHostToDevice));
    m_size = ntask * ntask * sizeof(bool);
    checkCudaErrors(cudaMemcpy(h_depd, depd, m_size, cudaMemcpyHostToDevice));
    m_size = ntask * nreso * sizeof(bool);
    checkCudaErrors(cudaMemcpy(h_asgn, asgn, m_size, cudaMemcpyHostToDevice));
    
    // setup parameter in device
    setInitPara<<<1, 1>>>(ntask, nreso, h_dura, h_depd, h_asgn, h_states);

    // setup random seed for each thread
    setupRandSeed<<<1, npop>>>(time(0));

    // free memory on host
    // dataFreeMemory();
}

__device__ void clearResouceOccupy(float * occupy)
{
    size_t i;
    for (i = 0; i < d_nreso; i++) {
        occupy[i] = 0.0f;
    }
}

__device__ float allocResouce(size_t ireso, float duration, float * occupy)
{
    occupy[ireso-1] += duration;
    return occupy[ireso-1];
}

__device__ float getTotalOccupy(size_t ireso, float * occupy)
{
    return occupy[ireso-1];
}

__device__ float getMaxTotalOccupy(float * occupy)
{
    size_t i, ireso;
    float max;
    
    ireso = 1;
    max = occupy[0];
    for (i = 0; i < d_nreso; i++) {
        if (occupy[i] > max) {
            ireso = i+1;
            max = occupy[i];
        }
    }

    return max;
}

// call once, setup seed for each thread
__global__ void setupRandSeed(unsigned long long seed)
{
    size_t tid = threadIdx.x;

    /* we have to initialize the state */
    curand_init(seed, /* the seed controls the sequence of random values that are produced */
                tid, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &d_states[tid]);
}

__device__ size_t randInt(const size_t min, const size_t max) 
{
    size_t tid = threadIdx.x;

    /* curand works like rand - except that it takes a state as a parameter */
    return (min + curand(&d_states[tid]) % (max - min));
}

__device__ float randProb()
{
    unsigned int result; 
    size_t tid = threadIdx.x;
    result = curand(&d_states[tid]) % 100000;

    return (float) result / 100000.0f;
}

#endif // !_HELPER_CUH_

