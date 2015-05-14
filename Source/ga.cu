/**
 * File: ga.cu
 * Author: Jeanhwea
 * Email: hujinghui@buaa.edu.cn
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include "helper.cuh"
#include "ga.cuh"

__device__ size_t d_npop;
__device__ size_t d_ngen;

// chromosome for [sz_taks * d_npop*2]
int * h_chrm;
/************************************************************************/
/* hash value for each person                                           */
/************************************************************************/
unsigned long * h_hashv;
/************************************************************************/
/* you can get fitness value like this:                                 */
/*      h_fitv[ itask-1 ]];                                             */
/************************************************************************/
float * h_fitv;
/************************************************************************/
/* ordering of each person in our population                            */
/************************************************************************/
size_t * h_order;
__device__ size_t * d_order;

// host data for display result
int * chrm;
unsigned long * hashv;
float * fitv;
size_t * order;

void cuGaEvolve()
{
    StopWatchInterface * timer = NULL;
    float elapse_time_inMs = 0.0f;
    cudaEvent_t start, stop;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Allocate GPU buffer
    allocMemOnDevice();

    // transfer data to GPU
    moveDataToDevice();

    if (ntask > MAX_CHRM_LEN) {
        fprintf(stderr, "ntask = %d (> MAX_CHRM_LEN)\n", ntask);
        exit(1);
    }
    
    gaAllocMem();

    // starting timer ...
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaThreadSynchronize());
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    // Launch a kernel on the GPU with one thread for each element.
    gaEvolve(npop, ngen);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapse_time_inMs, start, stop));
    elapse_time_inMs = sdkGetTimerValue(&timer);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());

    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());

    printf("total time in GPU = %f ms\n", elapse_time_inMs);

    gaFreeMem();
    freeMemOnDevice();
}

__global__ void gaSetPara(size_t npop, size_t ngen, size_t * h_order)
{
    d_npop = npop;
    d_ngen = ngen;
    d_order = h_order;
}

void gaAllocMem()
{
    size_t m_size;

    // chromosome attribution of a person
    m_size = 2 * npop * ntask * sizeof(int);
    checkCudaErrors(cudaMalloc((void **)&h_chrm, m_size));

    
    // hash value attribution of a person
    m_size = 2 * npop * sizeof(unsigned long);
    checkCudaErrors(cudaMalloc((void **)&h_hashv, m_size));

    // fitness value attribution of a person
    m_size = 2 * npop * sizeof(float);
    checkCudaErrors(cudaMalloc((void **)&h_fitv, m_size));

    // ordering after each selection
    m_size = 2 * npop * sizeof(size_t);
    checkCudaErrors(cudaMalloc((void **)&h_order, m_size));


    chrm = (int *) calloc(2 * npop * ntask, sizeof(int));
    assert(chrm != 0);
    hashv = (unsigned long *) calloc(2 * npop, sizeof(unsigned long));
    assert(hashv != 0);
    fitv = (float *) calloc(2 * npop, sizeof(float));
    assert(fitv != 0);
    order = (size_t *) calloc(2 * npop, sizeof(size_t));
    assert(order != 0);

}

void gaFreeMem()
{
    checkCudaErrors(cudaFree(h_chrm));
    checkCudaErrors(cudaFree(h_hashv));
    checkCudaErrors(cudaFree(h_fitv));
    checkCudaErrors(cudaFree(h_order));

    free(chrm);
    free(hashv);
    free(fitv);
    free(order);
}

static void dbPrintPerson(int * person, size_t n, char * tag)
{
    size_t i;

    printf("%s : ", tag);
    for (i = 0; i < n; i++) {
        printf("%d", person[i]);
        if (i < n-1) {
            printf("->");
        } else {
            printf("\n");
        }
    }

}

void dbDisplayWorld()
{
    size_t i;

    checkCudaErrors(cudaMemcpy(chrm, h_chrm, 2*npop * ntask * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hashv, h_hashv, 2*npop * sizeof(unsigned long), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fitv, h_fitv, 2*npop * sizeof(float), cudaMemcpyDeviceToHost));

    printf("parent----\n");
    for (i = 0; i < npop; i++) {;
        char tag[100];
        sprintf(tag, "i%04d\th%08u\tf%f\t",i, hashv[order[i]], fitv[order[i]]);
        dbPrintPerson(chrm+ntask*order[i], ntask, tag);
    }
    printf("children----\n");
    for (i = npop; i < 2*npop; i++) {;
        char tag[100];
        sprintf(tag, "i%04d\th%08u\tf%f\t",i, hashv[order[i]], fitv[order[i]]);
        dbPrintPerson(chrm+ntask*order[i], ntask, tag);
    }
}

void dbPrintResult(FILE * out)
{
    size_t i, j;

    checkCudaErrors(cudaMemcpy(chrm, h_chrm, 2*npop * ntask * sizeof(int), cudaMemcpyDeviceToHost));
    for (i = 0; i < 3; i++) {
        int * person = chrm+ntask*order[i];
        for (j = 0; j < ntask; j++) {
            fprintf(out, "%d%c", person[j] ,j==(ntask-1)? '\n': ' ');
        }
    }
}

void gaEvolve(size_t npop, size_t ngen)
{
    size_t i;
    FILE * fd_info, * fd_resu;

    fd_info = fopen("output.txt", "w");
    fd_resu = fopen("result.txt", "w");
    assert(fd_info != 0);

    gaSetPara<<<1, 1>>>(npop, ngen, h_order);
    size_t msize_occupy;
    msize_occupy = npop * nreso * sizeof(float);
    if (msize_occupy > MAX_SHARED_MEM) {
        fprintf(stderr, "msize_occupy = %d (> MAX_SHARED_MEM(%d))\n", msize_occupy, MAX_SHARED_MEM);
        exit(1);
    }

    for (i = 0; i < 2*npop; i++) {
        order[i] = i;
    }

    checkCudaErrors(cudaMemcpy(h_order, order, 2*npop * sizeof(size_t), cudaMemcpyHostToDevice));

    gaInit<<<1, npop, msize_occupy>>>(h_chrm, h_hashv, h_fitv);
    // dbDisplayWorld();

    for (i = 0; i < ngen; ++i) {
        // printf("%d generation----------------------\n", i+1);
        gaCrossover<<<1, npop/2, msize_occupy>>>(h_chrm, h_hashv, h_fitv);
        gaMutation<<<1, npop, msize_occupy>>>(h_chrm, h_hashv, h_fitv);
        gaSelection();
        gaStatistics(fd_info);
        // dbDisplayWorld();
        // checkCudaErrors(cudaDeviceSynchronize());
    }
    
    // dbDisplayWorld();
    dbPrintResult(fd_resu);

    fclose(fd_info);
    fclose(fd_resu);
}


/************************************************************************/
/* Initialize a person                                                  */
/************************************************************************/
__global__ void gaInit(int * h_chrm, unsigned long * h_hashv, float * h_fitv)
{
    int * person;
    size_t tid = threadIdx.x;
    extern __shared__ float sh_occupys[];
    float * occupy;

    person = h_chrm + d_ntask * d_order[tid];
    occupy = sh_occupys + tid * d_nreso;

    size_t i;
    for (i = 0; i < d_ntask; i++) {
        person[i] = i+1;
    }

    size_t a, b;
    for (i = 0; i < d_ntask; i++) {
        a = randInt(0, d_ntask-1);
        b = i; 
        if (a > b) {
            int tmp;
            tmp=a; a=b; b=tmp;
        }
        swapBits(a, b, person);
    }

    h_hashv[d_order[tid]] = hashfunc(person, d_ntask);
    h_fitv[d_order[tid]] = gaObject(person, occupy);
    // printf("%d %08u %f\n", d_order[tid], h_hashv[d_order[tid]], h_fitv[d_order[tid]]);
    __syncthreads();
}


__global__ void gaCrossover(int * h_chrm, unsigned long * h_hashv, float * h_fitv)
{
    int * dad, * mom, * bro, * sis, * person;
    size_t a, b, tid;
    size_t j, k;
    bool needCrossover;

    float * occupy;
    extern __shared__ float sh_occupys[];

    tid = threadIdx.x;
    occupy = sh_occupys + tid * d_nreso;
    
    needCrossover = true;
    while (needCrossover) { 
        a = randInt(0, d_npop-1);
        b = randInt(0, d_npop-1);
        dad = h_chrm + d_ntask*d_order[a];
        mom = h_chrm + d_ntask*d_order[b];
        bro = h_chrm + d_ntask*d_order[d_npop+2*tid];
        sis = h_chrm + d_ntask*d_order[d_npop+2*tid+1];

        crossover(dad, mom, bro, sis);

        if (!check(bro)) {
            fixPerson(bro);
        }
        if (!check(sis)) {
            fixPerson(sis);
        }

        unsigned long bro_hash, sis_hash;
        bro_hash = hashfunc(bro, d_ntask);
        sis_hash = hashfunc(sis, d_ntask);
        h_hashv[d_order[d_npop+2*tid]]   = bro_hash;
        h_hashv[d_order[d_npop+2*tid+1]] = sis_hash;

        needCrossover = false;
        for (j = 0; j < d_npop; j++) {
            // pick j-th person (parent)
            person = h_chrm + d_ntask*d_order[j];

            // check for brother
            if (bro_hash == h_hashv[d_order[j]]) {
                for (k = 0; k < d_ntask; k++) {
                    if (bro[k] != person[k])
                        break;
                }
                if (k == d_ntask) {
                    // need re-crossover
                    needCrossover = true;
                    break;
                }
            }
            // check for sister
            if (sis_hash == h_hashv[d_order[j]]) {
                for (k = 0; k < d_ntask; k++) {
                    if (sis[k] != person[k])
                        break;
                }
                if (k == d_ntask) {
                    // need re-crossover
                    needCrossover = true;
                    break;
                }
            }
        }
    }

    if (!needCrossover) {
        h_fitv[d_order[d_npop+2*tid]]   = gaObject(bro, occupy);
        h_fitv[d_order[d_npop+2*tid+1]] = gaObject(sis, occupy);
    }

    __syncthreads();
}

/************************************************************************/
/* ordering-based two points crossover                                  */
/************************************************************************/
__device__ void crossover(int * dad, int * mom, int * bro, int * sis)
{
    size_t i, j, k, a, b;
    int dad_new[MAX_CHRM_LEN], mom_new[MAX_CHRM_LEN];
    a = randInt(0, d_ntask-1);
    b = randInt(0, d_ntask-1);
    if (a > b) {
        size_t tmp;
        tmp=a; a=b; b=tmp;
    }

    for (i = 0; i < d_ntask; i++) {
        dad_new[i] = dad[i];
        mom_new[i] = mom[i];
        bro[i] = 0;
        sis[i] = 0;
    }

    // copy selected continuous region first (part1)
    for (i = a; i <= b; i++) {
        bro[i] = mom[i];
        sis[i] = dad[i];
    }

    // remove duplicated items
    for (k = 0; k < d_ntask; k++) {
        for (i = a; i <= b; i++) {
            if (dad_new[k] == mom[i]) {
                dad_new[k] = 0;
                break;
            }
        }
        for (i = a; i <= b; i++) {
            if (mom_new[k] == dad[i]) {
                mom_new[k] = 0;
                break;
            }
        }
    }

    
    // copy remainder region (part2)
    i = j = 0;
    for (k = 0; k < d_ntask; k++) {
        if (bro[k] == 0) {
            for (; i < d_ntask; i++) {
                if (dad_new[i] != 0) {
                    bro[k] = dad_new[i++];
                    break;
                }
            }
        }
        if (sis[k] == 0) {
            for (; j < d_ntask; j++) {
                if (mom_new[j] != 0) {
                    sis[k] = mom_new[j++];
                    break;
                }
            }
        }
    }

}

__global__ void gaMutation(int * h_chrm, unsigned long * h_hashv, float * h_fitv)
{
    int * person;
    size_t tid;

    float * occupy;
    extern __shared__ float sh_occupys[];

    tid = threadIdx.x;
    occupy = sh_occupys + tid * d_nreso;

    if (randProb() < PROB_MUTATION) {
        // mutate n-th parent
        person = h_chrm + d_ntask*d_order[tid];
        mutation(person);
        h_hashv[d_order[tid]] = hashfunc(person, d_ntask);
        h_fitv[d_order[tid]]  = gaObject(person, occupy);
    }
    if (randProb() < PROB_MUTATION) {
        // mutate n-th child
        person = h_chrm + d_ntask*d_order[tid+d_npop];
        mutation(person);
        h_hashv[d_order[tid+d_npop]] = hashfunc(person, d_ntask);
        h_fitv[d_order[tid+d_npop]]  = gaObject(person, occupy);
    }
    
    __syncthreads();
}

/************************************************************************/
/* two points swap mutation                                             */
/************************************************************************/
__device__ void mutation(int * person)
{
    size_t a, b;

    a = randInt(0, d_ntask-1);
    b = randInt(0, d_ntask-1);
    if (a > b) {
        size_t tmp;
        tmp=a; a=b; b=tmp;
    }

    swapBits(a, b, person);

}

/************************************************************************/
/* calculate fitness value, and move the bests to the parent of next    */
/*     generation.                                                      */
/************************************************************************/
void gaSelection()
{
    // copy fitness value form device
    checkCudaErrors(cudaMemcpy(fitv, h_fitv, 2*npop * sizeof(float), cudaMemcpyDeviceToHost));

    // sort individual by fitness value
    qsort(order, 2*npop, sizeof(size_t), fitvalueCompare);

    // transfer the order after sorting
    checkCudaErrors(cudaMemcpy(h_order, order, 2*npop * sizeof(size_t), cudaMemcpyHostToDevice));

}

/************************************************************************/
/* Statistics of some important information.                            */
/************************************************************************/
void gaStatistics(FILE * out)
{
    size_t i;

    for (i = 0; i < npop; i++) {
        fprintf(out, "%f%c", fitv[order[i]], i==npop-1 ? '\n': ' ');
    }
}

/****************************************************************************/
/* return true, if a-th task swap with b-th task; otherwise, return false.  */
/****************************************************************************/
__device__ bool swapBits(size_t a, size_t b, int * person)
{
    bool ret = true;
    // notice that, a < b
    if (a >= b) {
        ret = false;
    } else {
        size_t i, a_itask, b_itask, k_itask;
        a_itask = person[a];
        b_itask = person[b];
        for (i = a; i <= b; i++) {
            k_itask = person[i];
            if ( (i!=a) && IS_DEPEND(a_itask, k_itask) ){
                ret = false;
                break;
            }
            if ( (i!=b) && IS_DEPEND(k_itask, b_itask) ) {
                ret = false;
                break;
            }
        }
    }
    
    if (ret) {
        int tmp;
        tmp=person[a]; person[a]=person[b]; person[b]=tmp;
    }

    return ret;
}


#define HASH_SHIFT (3)
#define HASH_SIZE (19921104)
__device__ unsigned long hashfunc(int * person, size_t num)
{
    unsigned long hash_value;
    hash_value = 0;
    for (size_t i = 0; i < num; i++) {
        hash_value = (((unsigned long)person[i] + hash_value) << HASH_SHIFT ) % HASH_SIZE;
    }
    return hash_value;
}

__device__ float gaObject(int * person, float * occupy)
{
    float score;
    if (check(person)) {
        scheFCFS(person, occupy);
        score = getMaxTotalOccupy(occupy);
        if (score == 0.0f) {
            score = INF_DURATION;
        }
    } else {
        score = INF_DURATION;
    }
    return score;
}

/************************************************************************/
/*   feasibility check for <chromo_id>-th chromosome.                   */
/*       return true, if pass; otherwise, return false                  */
/************************************************************************/
__device__ bool check(int * person)
{
    size_t i, j;
    for (i = 0; i < d_ntask; i++) {
        for (j = i+1; j < d_ntask; j++) {
            int i_itask, j_itask;
            i_itask = person[i];
            j_itask = person[j];

            if (IS_DEPEND(j_itask, i_itask)) {
                // printf("failed depend %d -> %d\n", j_itask, i_itask);
                return false;
            }
        }
    }
    return true;
}

/************************************************************************/
/* scheduler, implement FCFS (first come, first service).               */
/************************************************************************/
__device__ void scheFCFS(int * person, float * occupy)
{
    size_t i, r, itask;

    // set temporary data struct as 0
    clearResouceOccupy(occupy);
    for (i = 0; i < d_ntask; i++) {
        itask = person[i];
        float dura = DURATION(itask);

        size_t min_id = 0;
        float min_occ, occ;
        for (r = 1; r <= d_nreso; r++) { // search all resources
            if (IS_ASSIGN(itask,r)) {
                if (min_id == 0) {
                    min_occ = getTotalOccupy(r, occupy);
                    min_id = r;
                } else {
                    occ = getTotalOccupy(r, occupy);
                    if (occ < min_occ) {
                        min_occ = occ;
                        min_id = r;
                    }
                }
            }
        }

        if (min_id > 0) {
            allocResouce(min_id, dura, occupy);
        } else {
            allocResouce(1, dura, occupy);
        }
    }
}

/************************************************************************/
/* move a person[ele] several steps forward                             */
/************************************************************************/
__device__ void personMoveForward(int * person, size_t ele, size_t step)
{
    int tmp;
    size_t i;
    tmp = person[ele];
    for (i = ele; i < ele + step; i++) {
        person[i] = person[i+1];
    }
    person[ele+step] = tmp;
}

__device__ void fixPerson(int * person)
{
    size_t i, j, step;
    i = 0;
    while (i < d_ntask) {                       // FOR all tasks listed in person array

        // Number of steps to move elements forward?
        step = 0;
        for (j = i+1; j < d_ntask; j++) {
            if (IS_DEPEND(person[j], person[i]))
                step = j-i;
        }

        if (step > 0) {
            personMoveForward(person, i, step);
        } else {
            // if no use to move, then i++
            i++;
        }

    }
}

static int fitvalueCompare(const void *a, const void *b)
{
    return (fitv[(*(size_t *)a)] > fitv[(*(size_t *)b)]) ? 1: -1;
}

