#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

__constant__ double r_pdf = 3;
__constant__ double dr_pdf = 0.1;
__constant__ double w_max = 1;
__constant__ double Re = 2500;
__constant__ uint64_t BUFFER_SIZE = 1e7;
__constant__ double maxTimeStep = 1e5;
__constant__ double twoPi = 2.0 * 3.141592865358979;

__device__ void printArray( uint64_t *v, int size)
{
    for(int i = 0; i < size; ++i)
        printf("%lli\n", v[i]);
}

__global__ void setup_kernel(curandState * state, unsigned long seed )
{
    int id = threadIdx.x  + blockIdx.x * blockDim.x;
    curand_init(seed, id , 0, &state[id]);
}

__device__ double Sigma(double r)
{
    return 10 * (0.2 * tanh(0.5 * r) - 0.1 * tanh(0.1 * r));
}

__device__ double D(double r, double L)
{
    return 10 * sqrt(0.1 + pow(r/L, 2));
}

__device__ double M(double r, double L)
{
    return sqrt(0.1 + pow(r/L, 2));
}

__global__ void numericalProcedure(uint64_t *d_concentration,
    const input_params params, const uint64_t size, curandState *state)
{
    double L = params.BoxSize;
    double a = params.a;
    double r_bin = L / size;
    double sqrt12 = sqrt((float)12);
    double tau_invert = pow(L, 2) / (pow(a,2) * Re);
    double tau = 1/tau_invert;
    double dt = tau / 10;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;
    double sqrt_dt_12 = sqrt_dt * sqrt12;

    double dr = 0;
    double w_r = 0;
    double w_phi = 0;
    double k_r1 = 0;
    double k_r2 = 0;
    double k_wr_1 = 0;
    double k_wr_2 = 0;
    double k_wphi_1 = 0;
    double k_wphi_2 = 0;

    uint64_t steps = params.numSteps;

    double W1 = 0;
    double W2 = 0;
    double W3 = 0;
    double W4 = 0;

    uint64_t ind = 0;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    double r = params.BoxSize * curand_uniform(&localState);

    curand_init(idx, 0, 0, &state[idx]);

    __syncthreads();

    for(uint64_t i = 0; i < steps; ++i)
    {   
        W1 = curand_uniform(&localState) - 0.5;
        W2 = curand_uniform(&localState) - 0.5;
        W3 = curand_uniform(&localState) - 0.5;
        W4 = curand_uniform(&localState) - 0.5;

        k_r1 = dt * w_r;

        k_wr_1 = - dt_tau_invert * w_r + sqrt_dt_12 * W1 * sqrt(D(r,L));

        k_wphi_1 = -dt_tau_invert * w_phi + sqrt_dt_12 * W2 * sqrt(D(r,L)) - dt * w_r * Sigma(r);

        k_r2 = dt * (w_r + k_wr_1);

        k_wr_2 = - dt_tau_invert * (w_r + k_wr_1) + sqrt_dt_12 * W3 * sqrt(D(r+k_r1,L));

        k_wphi_2 = -dt_tau_invert * (w_phi + k_wphi_1) 
        + sqrt_dt_12 * W4 * sqrt(D(r+k_r1,L))  - dt * (w_r + k_wr_1) * Sigma(r + k_r1);

        dr = 0.5 * (k_r1 + k_r2);

        w_r += 0.5 * (k_wr_1 + k_wr_2);
        w_phi += 0.5 * (k_wphi_1 + k_wphi_2);

        if(r + dr > L)
        {
            r = 2 * L - r - dr;
            w_r *= -1;
        }
        else if (r + dr < 0)
        {
            r = - r - dr;
            w_r *= -1;
        }
        else
            r = r + dr;

#ifdef CONCENTRATION
        ind = min(uint64_t(r / r_bin), size-1);
        atomicAdd(&d_concentration[ind], 1);
#endif // CONCENTRATION
    }

    __syncthreads();
}



