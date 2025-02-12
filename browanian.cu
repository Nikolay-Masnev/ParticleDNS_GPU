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

#define THREADS 100
#define BLOCKS  1

uint64_t nBins = 100;
uint64_t autocorr_nBins = 1000;

void normalizeArray(double* arr, uint64_t size)
{
    double sum = 0;

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
    uint64_t nbytes = nBins * sizeof(double);
    uint64_t autocorr_nbytes = autocorr_nBins * sizeof(double);

    double *concentration = 0;
    double *velocityVariance = 0;
    double *angleVariance = 0;
    double *pdf_vel = 0;
    double *w_autocorrelator = 0;
    double *phi_autocorrelator = 0;

    checkCudaErrors(cudaMallocHost((void **)&concentration, nbytes));
    checkCudaErrors(cudaMallocHost((void **)&velocityVariance, nbytes));
    checkCudaErrors(cudaMallocHost((void **)&angleVariance, nbytes));
    checkCudaErrors(cudaMallocHost((void **)&pdf_vel, nbytes));
    checkCudaErrors(cudaMallocHost((void **)&w_autocorrelator, autocorr_nbytes));
    checkCudaErrors(cudaMallocHost((void **)&phi_autocorrelator, autocorr_nbytes));

    memset(concentration, 0, nbytes);
    memset(velocityVariance, 0, nbytes);
    memset(angleVariance, 0, nbytes);
    memset(pdf_vel, 0, nbytes);
    memset(w_autocorrelator, 0, autocorr_nbytes);
    memset(phi_autocorrelator, 0, autocorr_nbytes);

    // allocate DEVISE memory
    printf("allocate DEVISE memory\n");
    double *d_concentration = 0;
    double *d_velocityVariance = 0;
    double *d_angleVariance = 0;
    double *d_pdf_vel = 0;
    double *d_w_autocorrelator = 0;
    double *d_phi_autocorrelator = 0;

    checkCudaErrors(cudaMalloc((void **)&d_concentration, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_velocityVariance, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_angleVariance, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_pdf_vel, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_w_autocorrelator, autocorr_nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_phi_autocorrelator, autocorr_nbytes));

    cudaMemset(d_concentration, 0, nbytes);
    cudaMemset(d_velocityVariance, 0, nbytes);
    cudaMemset(d_angleVariance, 0, nbytes);
    cudaMemset(d_pdf_vel, 0, nbytes);
    cudaMemset(d_w_autocorrelator, 0, autocorr_nbytes);
    cudaMemset(d_phi_autocorrelator, 0, autocorr_nbytes);

    //set kernel launch configuration
    printf("set kernel launch configuration\n");
    dim3 threads = dim3(THREADS, 1, 1);
    dim3 blocks = dim3(BLOCKS, 1, 1);

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
    cudaMemcpy(d_velocityVariance, velocityVariance, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_angleVariance, angleVariance, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pdf_vel, pdf_vel, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_autocorrelator, w_autocorrelator, autocorr_nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_autocorrelator, phi_autocorrelator, autocorr_nbytes, cudaMemcpyHostToDevice);

    //start
    printf("start\n");
    float gpu_time = 0.0f;
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    // main loop
    printf("main loop\n");
    numericalProcedure<<<THREADS, BLOCKS, 0, 0>>>(d_concentration, d_velocityVariance, d_pdf_vel, 
    d_w_autocorrelator, d_phi_autocorrelator, data,  nBins,  autocorr_nBins , devState);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(concentration, d_concentration, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityVariance, d_velocityVariance, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(pdf_vel, d_pdf_vel, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_autocorrelator, d_w_autocorrelator, autocorr_nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(phi_autocorrelator, d_phi_autocorrelator, autocorr_nbytes, cudaMemcpyDeviceToHost);

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

    // norm pdf
    printf("norm pdf\n");
    normalizeArray(concentration, nBins);
    //normalizeArray(velocityVariance, nBins);
    //normalizeArray(pdf_vel, nBins);

    //normalizeArray(w_autocorrelator, autocorr_nBins);
    //normalizeArray(phi_autocorrelator, autocorr_nBins);

    // save distribution
    printf("save dist\n");
    saveHist(concentration, argv[2], nBins);
    //saveHist(velocityVariance, argv[3], nBins);
    //saveHist(pdf_vel, argv[4], nBins);

    //saveHist(w_autocorrelator, argv[5], autocorr_nBins);
    //saveHist(phi_autocorrelator, argv[6], autocorr_nBins);

    // free memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(concentration));
    checkCudaErrors(cudaFreeHost(velocityVariance));
    checkCudaErrors(cudaFreeHost(pdf_vel));
    checkCudaErrors(cudaFreeHost(w_autocorrelator));
    checkCudaErrors(cudaFreeHost(phi_autocorrelator));

    checkCudaErrors(cudaFree(d_concentration));
    checkCudaErrors(cudaFree(d_velocityVariance));
    checkCudaErrors(cudaFree(d_pdf_vel));
    checkCudaErrors(cudaFree(d_w_autocorrelator));
    checkCudaErrors(cudaFree(d_phi_autocorrelator));

    checkCudaErrors(cudaFree(devState));

    return EXIT_SUCCESS;
}