#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

__constant__ double Re = 2500;

__device__ void printArray( uint64_t *v, unsigned long long int size)
{
    for(unsigned long long int i = 0; i < size; ++i)
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
    return 0.01 * sqrt(0.1 + pow(r/L, 2));
}

__device__ double M(double r, double L)
{
    return 1e2 * (1 + 1e4 * pow(r/L, 2) );
}

__device__ double tau_corr(double r, double L)
{
    return 1/sqrt(15 * M(r, L));
}

__global__ void numericalProcedure(unsigned long long int *d_concentration,
    const input_params params, const unsigned long long int size, curandState *state,
    float *d_tr_x, float *d_tr_y, float *d_tr_wx, float *d_tr_wy, unsigned long long int tr_points)
{
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    curand_init(idx, 0, 0, &state[idx]);

    double L = params.BoxSize;
    double a = params.a;
    double r_bin = L / size / 2;
    double sqrt12 = sqrt((float)12);
    double tau_invert = pow(L, 2) / (pow(a,2) * Re);
    double tau = 1/tau_invert;
    double dt = tau / 10;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;
    double sqrt_dt_12 = sqrt_dt * sqrt12;
    double dx, dy, w_x, w_y, kx_1, kx_2, ky_1, ky_2, 
    kwx_1, kwx_2, kwy_1, kwy_2;

    dx = 0;
    dy = 0;
    w_x = 0;
    w_y = 0;
    kx_1 = 0;
    kx_2 = 0;
    ky_1 = 0;
    ky_2 = 0;
    kwx_1 = 0;
    kwx_2 = 0;
    kwy_1 = 0;
    kwy_2 = 0;

    unsigned long long int steps = params.numSteps;
    unsigned long long int ind = 0;

    double x = L * (curand_uniform(&localState) - 0.5);
    double y = L * (curand_uniform(&localState) - 0.5);
    double r = sqrt(x*x + y*y);

    double W1 = 0;
    double W2 = 0;
    double W3 = 0;
    double W4 = 0;
    double W5 = 0;
    double W6 = 0;
    double W7 = 0;
    double W8 = 0;
    double W1_old = 0;
    double W2_old = 0;
    double W3_old = 0;
    double W4_old = 0;
    double W5_old = 0;
    double W6_old = 0;
    double W7_old = 0;
    double W8_old = 0;
    
    double rho = 0;
    double sqrt_one_rho = 0;

    __syncthreads();

    for(unsigned long long int i = 0; i < steps; ++i)
    {   
        // rho = exp(-dt/tau_corr(r, L));
        // sqrt_one_rho = sqrt(1 - rho * rho);
        rho = 0;
        sqrt_one_rho = 1;

        W1 = W1_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W2 = W2_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W3 = W3_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W4 = W4_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W5 = W5_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W6 = W6_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
       	W7 = W7_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
	    W8 = W8_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W1_old = W1;
        W2_old = W2;
        W3_old = W3;
        W4_old = W4;
        W5_old = W5;
        W6_old = W6;
        W7_old = W7;
        W8_old = W8;
        
        kx_1 = 0 * dt * w_x + sqrt_dt_12 * W5 * sqrt(D(r, L));
        ky_1 = 0 * dt * w_y + sqrt_dt_12 * W6 * sqrt(D(r, L));
        kwx_1 = - dt_tau_invert * w_x + sqrt_dt_12 * W1 * sqrt(D(r,L));
        kwy_1 = - dt_tau_invert * w_y + sqrt_dt_12 * W2 * sqrt(D(r,L));

        r = sqrt((x+kx_1)*(x+kx_1) + (y+ky_1)*(y+ky_1));

        kx_2 = 0 * dt * (w_x + kwx_1) + sqrt_dt_12 * W7 * sqrt(D(r, L));
        ky_2 = 0 * dt * (w_y + kwy_1) + sqrt_dt_12 * W8 * sqrt(D(r, L));
        kwx_2 = - dt_tau_invert * (w_x + kwx_1) + sqrt_dt_12 * W3 * sqrt(D(r,L));
        kwy_2 = - dt_tau_invert * (w_y + kwy_1) + sqrt_dt_12 * W4 * sqrt(D(r,L));

        dx = 0.5 * (kx_1 + kx_2);
        dy = 0.5 * (ky_1 + ky_2);
        w_x += 0.5 * (kwx_1 + kwx_2);
        w_y += 0.5 * (kwy_1 + kwy_2);

        if(x + dx > L/2)
        {
            x = L - x - dx;
            w_x *= -1;
            W1 *= -1;
            W3 *= -1;
        }
        else if (x + dx < -L/2)
        {
            x = -L  - x - dx;
	        w_x *= -1;
	        W1 *= -1;
            W3 *= -1;
        }
        else
        {
            x += dx;
        }
        
        if(y + dy > L/2)
        {
            y = L - y - dy;
            w_y *= -1;
            W2 *= -1;
            W4 *= -1;
        }
        else if (y + dy < -L/2)
        {
            y = -L  - y - dy;
	        w_y *= -1;
	        W2 *= -1;
            W4 *= -1;
        }
        else
        {
            y += dy;
        }

        r = sqrt(abs(x*x + y*y));

#ifdef CONCENTRATION
        ind = min(int(r / r_bin), int(size-1));
        atomicAdd(&d_concentration[ind], 1);
#endif // CONCENTRATION

#ifdef TRAJECTORY
        if( i < tr_points)
        {
            d_tr_x[i] = x;
            d_tr_y[i] = y;
            d_tr_wx[i] = w_x;
            d_tr_wy[i] = w_y;
        }
#endif // TRAJECTORY
    }

    __syncthreads();
}
