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
#define BLOCKS  100

unsigned long long int nBins = 100;
unsigned long long int tr_points = 1e4;

template <typename T>
void printArray(T* arr, unsigned long long int size)
{
    for(unsigned long long int i = 0; i < size; ++i)
    {
        std::cout << arr[i] << ' '; 
    }

    std::cout << '\n';
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

void normalizeVariance(double* arr, unsigned long long int* uint_arr, unsigned long long int size)
{
    printArray(arr, size);
    printArray(uint_arr, size);

    for(unsigned long long int i = 0; i < size; ++i)
    {
        arr[i] = arr[i] / (uint_arr[i]+1);
	arr[i] = std::sqrt(arr[i]);
    }
}

int main(int argc, char *argv[])
{ 
    // reading params
    printf("reading params\n");
    std::string paramsPath(argv[1]);
    input_params data;
    readParams(data, paramsPath);
    printParams(data);

    // allocate HOST memory
    printf("allocate HOST memory\n");

    float *concentration = nullptr;
    float *concentration_2D = nullptr;
    unsigned long long int *uint64_t_concentration = nullptr;
    unsigned long long int *uint64_t_concentration_2D = nullptr;
    float *tr_x = nullptr;
    float *tr_y = nullptr;
    float *tr_wx = nullptr;
    float *tr_wy = nullptr;
    double *velocity_variance = nullptr;
    unsigned long long int *variance_counter = nullptr;

#ifdef CONCENTRATION
    checkCudaErrors(cudaMallocHost((void **)&concentration, nBins * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&uint64_t_concentration, nBins * sizeof(unsigned long long int)));
    memset(concentration, 0, nBins * sizeof(float));
    memset(uint64_t_concentration, 0, nBins * sizeof(unsigned long long int));
#endif /* CONCENTRATION */

#ifdef _2D_HISTOGRAM
    checkCudaErrors(cudaMallocHost((void **)&concentration_2D, nBins * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&uint64_t_concentration_2D, 4 * nBins * nBins * sizeof(unsigned long long int)));
    memset(concentration_2D, 0, 4 * nBins * nBins * sizeof(float));
    memset(uint64_t_concentration_2D, 0, 4 * nBins * nBins * sizeof(unsigned long long int));
#endif /* 2D_HISTOGRAM */

#ifdef TRAJECTORY
    uint64_t tr_nbytes = tr_points * sizeof(float);
    checkCudaErrors(cudaMallocHost((void **)&tr_x, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_y, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_wx, tr_nbytes ));
    checkCudaErrors(cudaMallocHost((void **)&tr_wy, tr_nbytes ));
    memset(tr_x, 0, nbytes);
    memset(tr_y, 0, nbytes);
    memset(tr_wx, 0, nbytes);
    memset(tr_wy, 0, nbytes);
#endif /* TRAJECTORY */

#ifdef VELOCITY_VARIANCE
    checkCudaErrors(cudaMallocHost((void **)&velocity_variance, nBins * sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&variance_counter, nBins * sizeof(unsigned long long int)));
    memset(velocity_variance, 0, nBins * sizeof(double));
    memset(variance_counter, 0, nBins * sizeof(unsigned long long int));
#endif /* VELOCITY_VARIANCE */
    
    // allocate DEVISE memory
    printf("allocate DEVISE memory\n");

    unsigned long long int *d_concentration = nullptr;
    unsigned long long int *d_concentration_2D = nullptr;
    float *d_tr_x = nullptr;
    float *d_tr_y = nullptr;
    float *d_tr_wx = nullptr;
    float *d_tr_wy = nullptr;
    double *d_velocity_variance = nullptr;
    unsigned long long int *d_variance_counter = nullptr;

#ifdef CONCENTRATION
    checkCudaErrors(cudaMalloc((void **)&d_concentration, nBins * sizeof(unsigned long long int)));
    cudaMemset(d_concentration, 0, nBins * sizeof(unsigned long long int));
#endif /* CONCENTRATION */

#ifdef _2D_HISTOGRAM
    checkCudaErrors(cudaMalloc((void **)&d_concentration_2D, 4 * nBins * nBins * sizeof(unsigned long long int)));
    cudaMemset(d_concentration_2D, 0, 4 * nBins * nBins * sizeof(unsigned long long int));
#endif /* 2D_HISTOGRAM */

#ifdef TRAJECTORY
    checkCudaErrors(cudaMalloc((void **)&d_tr_x, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_y, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_wx, tr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_tr_wy, tr_nbytes));
    
    cudaMemset(d_tr_x, 0, tr_nbytes);
    cudaMemset(d_tr_y, 0, tr_nbytes);
    cudaMemset(d_tr_wx, 0, tr_nbytes);
    cudaMemset(d_tr_wy, 0, tr_nbytes);
#endif /* TRAJECTORY */

#ifdef VELOCITY_VARIANCE
    checkCudaErrors(cudaMalloc((void **)&d_velocity_variance, nBins * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_variance_counter, nBins * sizeof(unsigned long long int)));
    cudaMemset(d_velocity_variance, 0, nBins * sizeof(double));
    cudaMemset(d_variance_counter, 0, nBins * sizeof(unsigned long long int));
#endif /* VELOCITY_VARIANCE */

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

#ifdef CONCENTRATION
    cudaMemcpy(d_concentration, uint64_t_concentration, nBins * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
#endif /* CONCENTRATION */

#ifdef _2D_HISTOGRAM
    cudaMemcpy(d_concentration_2D, uint64_t_concentration_2D, 4 * nBins * nBins * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
#endif /* 2D_HISTOGRAM */

#ifdef TRAJECTORY
    cudaMemcpy(d_tr_x, tr_x, tr_nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tr_y, tr_y, tr_nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tr_wx, tr_wx, tr_nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tr_wy, tr_wy, tr_nbytes, cudaMemcpyHostToDevice);
#endif /* TRAJECTORY */

#ifdef VELOCITY_VARIANCE
    cudaMemcpy(d_velocity_variance, velocity_variance, nBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance_counter, variance_counter, nBins * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
#endif /* VELOCITY_VARIANCE */

    //start
    printf("start\n");
    float gpu_time = 0.0f;
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    // main loop
    printf("main loop\n");

    int nBins_2D =  4 * nBins * nBins;
    printf("Launch kernel\n");   
    numericalProcedure<<<THREADS, BLOCKS, 0, 0>>>(d_concentration, data,  nBins, devState, 
        d_tr_x, d_tr_y, d_tr_wx, d_tr_wy, tr_points, d_concentration_2D, nBins_2D, d_velocity_variance, d_variance_counter, nBins);
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef CONCENTRATION
    cudaMemcpy(uint64_t_concentration, d_concentration, nBins * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
#endif /* CONCENTRATION */
    
#ifdef TRAJECTORY
    cudaMemcpy(tr_x, d_tr_x, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_y, d_tr_y, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_wx, d_tr_wx, tr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tr_wy, d_tr_wy, tr_nbytes, cudaMemcpyDeviceToHost);
#endif /* TRAJECTORY */

#ifdef _2D_HISTOGRAM
    cudaMemcpy(uint64_t_concentration_2D, d_concentration_2D, 4 * nBins * nBins * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
#endif /* 2D_HISTOGRAM */

#ifdef VELOCITY_VARIANCE
    cudaMemcpy(velocity_variance, d_velocity_variance, nBins * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(variance_counter, d_variance_counter, nBins * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
#endif /* VELOCITY_VARIANCE */

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
#ifdef CONCENTRATION 
    normalizeArray(concentration, uint64_t_concentration,nBins);
#endif /* CONCENTRATION */

#ifdef _2D_HISTOGRAM
    normalizeArray(concentration_2D, uint64_t_concentration_2D, 4 * nBins * nBins);
#endif /* 2D_HISTOGRAM */

#ifdef VELOCITY_VARIANCE
    normalizeVariance(velocity_variance, variance_counter, nBins);
#endif /* VELOCITY_VARIANCE */

    // save distribution
    printf("save dist\n");

#ifdef CONCENTRATION  
    saveHist(concentration, argv[2], nBins);
#endif /* CONCENTRATION3 */

#ifdef TRAJECTORY   
    saveHist(tr_x, argv[3], tr_points);
    saveHist(tr_y, argv[4], tr_points);
    saveHist(tr_wx, argv[5], tr_points);
    saveHist(tr_wy, argv[6], tr_points);
#endif /* TRAJECTORY */

#ifdef _2D_HISTOGRAM
    saveHist(concentration_2D, argv[7], 4 * nBins * nBins);
#endif /* 2D_HISTOGRAM */

#ifdef VELOCITY_VARIANCE
    saveHist(velocity_variance, argv[8], nBins);
#endif /* VELOCITY_VARIANCE */

    // free memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(devState));

#ifdef CONCENTRATION  
    checkCudaErrors(cudaFreeHost(concentration));
    checkCudaErrors(cudaFreeHost(uint64_t_concentration));
    checkCudaErrors(cudaFree(d_concentration));
#endif /* CONCENTRATION */

#ifdef _2D_HISTOGRAM
    checkCudaErrors(cudaFreeHost(concentration_2D));
    checkCudaErrors(cudaFreeHost(uint64_t_concentration_2D));
    checkCudaErrors(cudaFree(d_concentration_2D));
#endif /* 2D_HISTOGRAM */

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

#ifdef VELOCITY_VARIANCE
    checkCudaErrors(cudaFreeHost(velocity_variance));
    checkCudaErrors(cudaFreeHost(variance_counter));
    checkCudaErrors(cudaFree(d_velocity_variance));
    checkCudaErrors(cudaFree(d_variance_counter));
#endif /* VELOCITY_VARIANCE */

    return EXIT_SUCCESS;
}
