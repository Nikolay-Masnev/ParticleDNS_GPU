
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

__device__ double sqrt_D(double r, double L)
{
    return sqrt(1 + a * (r-L/2)*(r-L/2) / (L*L/4));
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
    double dt = tau/100;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;

    double dx = 0, x = 0;
    double dw_x = 0, w_x = 0;
    double u = 0, du=0;
    double sqrt_K = 0.1;

    unsigned long long int steps = params.numSteps;
    unsigned long long int ind = 0;

    x = L * curand_uniform(&localState);

    __syncthreads();

    double t_c_inv = 1 / (0.1 * tau);

    for(unsigned long long int i = 0; i < steps; ++i)
    {
	u += t_c_inv * (-u * dt + sqrt_D(x,L) * sqrt_dt * curand_normal(&localState));
	w_x += (- tau_invert * w_x * dt + sqrt_K * sqrt_dt * curand_normal(&localState));
	x += (w_x + u) * dt;
	//x += sqrt_D(x,L) * sqrt_dt * curand_normal(&localState);

        if(x > L)
        {
            x = L - abs(x-L);
	    w_x = -w_x;
	    u = 0;
        }
	else if(x < 0)
	{
	    x = -x;
	    w_x = -w_x;
	    u = 0;
	}

#ifdef CONCENTRATION
        ind = min(int(x/r_bin), int(size-1));
        atomicAdd(&d_concentration[ind], 1);
#endif // CONCENTRATION
    }

    __syncthreads();
}
