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

#define THREADS 1
#define BLOCKS  1

unsigned long long int nBins = 1000;
unsigned long long int tr_points = 1e4;

void printArray(float* arr, unsigned long long int size)
{
    for(unsigned long long int i = 0; i < size; ++i)
    {
        printf("%lli: %f\n", i, arr[i]);
    }
}

void normalizeArray(float* arr, unsigned long long int* uint_arr, unsigned long long int size)
{
    unsigned long long int sum = 0;

    for(unsigned long long int i = 0; i < size; ++i)
    {
        sum+=uint_arr[i];
    }

    for(unsigned long long int i = 0; i < size; ++i)
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

    unsigned long long int *uint64_t_concentration = 0;
    checkCudaErrors(cudaMallocHost((void **)&uint64_t_concentration, nBins * sizeof(unsigned long long int)));
    memset(concentration, 0, nBins * sizeof(unsigned long long int));
    
   
#ifdef TRAJECTORY
    uint64_t tr_nbytes = tr_points * sizeof(float);
    float *tr_x = 0;
    float *tr_y = 0;
    float *tr_wx = 0;
    float *tr_wy = 0;
    checkCudaErrors(cudaMallocHost((void **)&tr_x, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_y, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_wx, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_wy, tr_nbytes ));
    memset(tr_x, 0, nbytes);
    memset(tr_y, 0, nbytes);
    memset(tr_wx, 0, nbytes);
    memset(tr_wy, 0, nbytes);
#endif /* TRAJECTORY */
    
    // allocate DEVISE memory
    printf("allocate DEVISE memory\n");
    unsigned long long int *d_concentration = 0;
    checkCudaErrors(cudaMalloc((void **)&d_concentration, nBins * sizeof(unsigned long long int)));
    cudaMemset(d_concentration, 0, nBins * sizeof(unsigned long long int));
    
#ifdef TRAJECTORY
    float *d_tr_x = 0;
    float *d_tr_y = 0;
    float *d_tr_wx = 0;
    float *d_tr_wy = 0;
    
    checkCudaErrors(cudaMalloc((void **)&d_tr_x, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_y, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_wx, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_wy, tr_nbytes));
    
    cudaMemset(d_tr_x, 0, tr_nbytes);
    cudaMemset(d_tr_y, 0, tr_nbytes);
    cudaMemset(d_tr_wx, 0, tr_nbytes);
    cudaMemset(d_tr_wy, 0, tr_nbytes);
#endif /* TRAJECTORY */

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
    cudaMemcpy(d_concentration, uint64_t_concentration, nBins * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    //start
    printf("start\n");
    float gpu_time = 0.0f;
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    // main loop
    printf("main loop\n");
    
#ifdef TRAJECTORY
    numericalProcedure<<<THREADS, BLOCKS, 0, 0>>>(d_concentration, data,  nBins, devState, d_tr_x, d_tr_y, d_tr_wx, d_tr_wy, tr_points);
#else 
    numericalProcedure<<<THREADS, BLOCKS, 0, 0>>>(d_concentration, data,  nBins, devState, NULL, NULL, NULL, NULL, NULL);
#endif /* TRAJECTORY */

    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(uint64_t_concentration, d_concentration, nBins * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
#ifdef TRAJECTORY
    cudaMemcpy(tr_x, d_tr_x, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_y, d_tr_y, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_wx, d_tr_wx, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_wy, d_tr_wy, tr_nbytes, cudaMemcpyDeviceToHost);
#endif /* TRAJECTORY */

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
    unsigned long long int sum = 0;

    for(unsigned long long int i = 0; i < nBins; ++i)
    {
        sum += uint64_t_concentration[i];
    }

    printf("sum = %lli\n", sum);

    normalizeArray(concentration, uint64_t_concentration,nBins);

    // save distribution
    printf("save dist\n");
    saveHist(concentration, argv[2], nBins);
    
#ifdef TRAJECTORY   
    saveHist(tr_x, argv[3], tr_points);
    saveHist(tr_y, argv[4], tr_points);
    saveHist(tr_wx, argv[5], tr_points);
    saveHist(tr_wy, argv[6], tr_points);
#endif /* TRAJECTORY */

    // free memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(concentration));
    checkCudaErrors(cudaFreeHost(uint64_t_concentration));
    checkCudaErrors(cudaFree(d_concentration));
    checkCudaErrors(cudaFree(devState));


#ifdef TRAJECTORY
    checkCudaErrors(cudaFreeHost(tr_x));
    checkCudaErrors(cudaFreeHost(tr_y));
    checkCudaErrors(cudaFreeHost(tr_wx));
    checkCudaErrors(cudaFreeHost(tr_wy));
    checkCudaErrors(cudaFree(d_tr_x));
    checkCudaErrors(cudaFree(d_tr_y));
    checkCudaErrors(cudaFree(d_tr_wx));
    checkCudaErrors(cudaFree(d_tr_wy));
#endif /* TRAJECTORY */

    return EXIT_SUCCESS;
}
