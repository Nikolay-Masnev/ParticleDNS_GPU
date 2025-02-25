#include <math.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

#include "dataio.h"
#include "input_params.h"
#include "langevin.h"

#define THREADS 1000
#define BLOCKS  1

uint64_t nBins = 100;

void printArray(float* arr, uint64_t size)
{
    for(uint64_t i = 0; i < size; ++i)
    {
        printf("%lli: %f\n", i, arr[i]);
    }
}

void normalizeArray(float* arr, uint64_t* uint_arr, uint64_t size)
{
    uint64_t sum = 0;

    for(uint64_t i = 0; i < size; ++i)
    {
        sum+=uint_arr[i];
    }

    for(uint64_t i = 0; i < size; ++i)
    {
        arr[i] =float(uint_arr[i]) / sum;
    }
}

int main(int argc, char *argv[])
{ 
    // reading params
    printf("reading params\n");
    std::string paramsPath(argv[1]);
    input_params data;
    readParams(data, paramsPath);

    // allocate HOST memory
    printf("allocate HOST memory\n");
    uint64_t nbytes = nBins * sizeof(float);    
    float *concentration = 0;
    checkCudaErrors(cudaMallocHost((void **)&concentration, nbytes));
    memset(concentration, 0, nbytes);

    uint64_t *uint64_t_concentration = 0;
    checkCudaErrors(cudaMallocHost((void **)&uint64_t_concentration, nBins * sizeof(uint64_t)));
    memset(concentration, 0, nBins * sizeof(uint64_t));

    // allocate DEVISE memory
    printf("allocate DEVISE memory\n");
    uint64_t *d_concentration = 0;
    checkCudaErrors(cudaMalloc((void **)&d_concentration, nBins * sizeof(uint64_t)));
    cudaMemset(d_concentration, 0, nBins * sizeof(uint64_t));

    // create cuda event handles
    printf("create cuda event handles\n");
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    checkCudaErrors(cudaDeviceSynchronize());

    // set random number generator
    printf("set random number generator\n");
    curandState *devState;
    checkCudaErrors(cudaMalloc((void**)&devState, THREADS * BLOCKS * sizeof(curandState)));
    time_t t;
    time(&t);
    setup_kernel<<<THREADS, BLOCKS, 0, 0>>>(devState, (unsigned long) t);

    // copy data from host to devise
    printf("copy data from host to devise\n");
    cudaMemcpy(d_concentration, uint64_t_concentration, nBins * sizeof(uint64_t), cudaMemcpyHostToDevice);

    //start
    printf("start\n");
    float gpu_time = 0.0f;
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    // main loop
    printf("main loop\n");
    numericalProcedure<<<THREADS, BLOCKS, 0, 0>>>(d_concentration, data,  nBins, devState);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(uint64_t_concentration, d_concentration, nBins * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // stop
    printf("stop\n");
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    printf("time spent executing by the GPU: %.2f\n", gpu_time);

    checkCudaErrors(cudaDeviceSynchronize());

    // normalize
    uint64_t sum = 0;

    for(int i = 0; i < nBins; ++i)
    {
        sum += uint64_t_concentration[i];
    }

    printf("sum = %lli\n", sum);

    normalizeArray(concentration, uint64_t_concentration,nBins);

    // save distribution
    printf("save dist\n");
    saveHist(concentration, argv[2], nBins);

    // free memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(concentration));
    checkCudaErrors(cudaFreeHost(uint64_t_concentration));
    checkCudaErrors(cudaFree(d_concentration));
    checkCudaErrors(cudaFree(devState));

    return EXIT_SUCCESS;
}