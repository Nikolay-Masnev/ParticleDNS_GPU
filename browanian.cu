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

void normalizeArray(float* arr, uint64_t size)
{
    float sum = 0;

    for(uint64_t i = 0; i < size; ++i)
    {
        sum+=arr[i];
    }

    for(uint64_t i = 0; i < size; ++i)
    {
        arr[i] /= sum;
    }
}

int main(int argc, char *argv[])
{ 
    int devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

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

    // allocate DEVISE memory
    printf("allocate DEVISE memory\n");
    float *d_concentration = 0;

    checkCudaErrors(cudaMalloc((void **)&d_concentration, nbytes));

    cudaMemset(d_concentration, 0, nbytes);

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
    cudaMemcpy(d_concentration, concentration, nbytes, cudaMemcpyHostToDevice);

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

    cudaMemcpy(concentration, d_concentration, nbytes, cudaMemcpyDeviceToHost);

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

    for(uint64_t i = 0; i < uint64_t(nBins); ++i)
    {
        sum+= uint64_t(concentration[i]);
    }

    printf("sum = %lli\n", sum);

    normalizeArray(concentration, nBins);

    // save distribution
    printf("save dist\n");
    saveHist(concentration, argv[2], nBins);

    // free memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(concentration));
    checkCudaErrors(cudaFree(d_concentration));
    checkCudaErrors(cudaFree(devState));

    return EXIT_SUCCESS;
}