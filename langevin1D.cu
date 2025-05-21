
#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

__constant__ double Re = 2500;
#define  M_PI  3.14159265358979323846
__constant__ double a = 1.0;

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
    return (1 + a * r*r / (L*L));
}

__device__ double D_k(int n, double L)
{
    if(n==0)
	return 1 + a/12;
    else
	return (12 + a)/24 + a/ (16 * n * n * M_PI * M_PI);
}

__device__ double tau_corr(double r, double L)
{
    return 10.;
}

__global__ void numericalProcedure(unsigned long long int *d_concentration,
    const input_params params, const unsigned long long int size, curandState *state,
    float *d_tr_x, float *d_tr_y, float *d_tr_wx, float *d_tr_wy, unsigned long long int tr_points,
    unsigned long long int *d_concentration_2D, unsigned long long int size_2D,
    double *d_velocity_variance, unsigned long long *d_variance_counter, unsigned long long int variance_size)
{
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    curand_init(idx, 0, 0, &state[idx]);

    double L = params.BoxSize;
    double a = params.a;
    double r_bin = L / size;
    double sqrt12 = sqrt((float)12);
    double tau_invert = pow(L, 2) / (pow(a,2) * Re);
    double tau = 1/tau_invert;
    double dt = tau/10;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;
    const int fourier_size = 100;
    double k = 2 * M_PI / L; 

    double dx = 0, x = 0;
    double dw_x = 0, w_x = 0;

    unsigned long long int steps = params.numSteps;
    unsigned long long int ind = 0;

    x = L * (curand_uniform(&localState)-0.5);

    double W[100];
    
    for(int i = 0; i < fourier_size; ++i)
	W[i] = 0;

    double u = 0;

    __syncthreads();

    double tmp = 0;

    double t_c_inv = 1 / (10 * tau);

    for(unsigned long long int i = 0; i < steps; ++i)
    {
	for (int j = 0; j < fourier_size; ++j)
	{
	    tmp =  - dt * W[j] * t_c_inv + sqrt(2 * D_k(j,L)) * sqrt_dt * curand_normal(&localState) * t_c_inv;
	    W[j] += tmp;
	    u += tmp * 2 * cos(k * j * x) / fourier_size;
	}

	w_x = w_x - tau_invert * w_x * dt + tau_invert * u * dt;
	x += w_x * dt;

        if(x > L/2)
        {
            x = L/2 - abs(x-L/2);
	    w_x = -w_x;
        }
	else if(x < -L/2)
	{
	    x = -L/2 + abs(x + L/2);
	    w_x = -w_x;
	}

#ifdef CONCENTRATION
        ind = min(int((x+L/2) / r_bin), int(size-1));
        atomicAdd(&d_concentration[ind], 1);
#endif // CONCENTRATION
    }

    __syncthreads();
}
