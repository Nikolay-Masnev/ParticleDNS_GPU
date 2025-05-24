#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

__constant__ double Re = 2500.;
#define  M_PI  3.14159265358979323846

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

__device__ double D(double r, double L)
{
    return (1 + pow(r/L, 2));
}

__device__ double K(double r, double L)
{
    return 0.1 * (1 + pow(r/L, 2));
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
    double tau_invert = 10 * pow(L, 2) / (pow(a,2) * Re);
    double tau = 1/tau_invert;
    double dt = tau/10;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;

    double dx = 0, dy = 0, x = 0, y = 0;
    double dw_x = 0, dw_y = 0, w_x = 0, w_y = 0;

    unsigned long long int steps = params.numSteps;
    unsigned long long int ind = 0;
    unsigned long long int indx = 0;
    unsigned long long int indy = 0;

    double r = L * curand_uniform(&localState);
    double phi = 2 * M_PI * curand_uniform(&localState);

    x = r * sin(phi);
    y = r * cos(phi);

    double dW1 = 0;
    double dW2 = 0;
    double dW3 = 0;
    double dW4 = 0;

    double rho = 0;
    double sqrt_one_rho = 0;
    int ind_2d = 0;

    double t_c = tau/10;

    __syncthreads();

    for(unsigned long long int i = 0; i < steps; ++i)
    {   
	    rho = exp(-dt/t_c);
        sqrt_one_rho = sqrt(1 - rho * rho);

        dW1 = sqrt_one_rho * curand_normal(&localState) + rho * dW1;
        dW2 = sqrt_one_rho * curand_normal(&localState) + rho * dW2;
        dW3 = curand_normal(&localState);
        dW4 = curand_normal(&localState);

        dx = w_x * dt + sqrt(2 * D(r, L)) * sqrt_dt * dW1;
        dy = w_y + dt * sqrt(2 * D(r, L)) * sqrt_dt * dW2;

        dw_x = -tau_invert * w_x * dt + sqrt(2 * K(r, L)) * sqrt_dt * dW3;
        dw_y = -tau_invert * w_y * dt + sqrt(2 * K(r, L)) * sqrt_dt * dW4;

        x += dx;
        y += dy;
        w_x += dw_x;
        w_y += dw_y;

	    if(w_x * (w_x - dw_x) < 0)
		    w_x = 0;
    	if(w_y * (w_y - dw_y) < 0)
		    w_y = 0;

        r = sqrt(x*x + y*y);

        if(r > L)
        {
            x -= dx;
            y -= dy;
            w_x = -w_x;
            w_y = -w_y;
            dW1 = 0;
            dW2 = 0;
            dW3 = 0;
            dW4 = 0;
        }

        r = sqrt(x*x + y*y);

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

#ifdef _2D_HISTOGRAM
        indx = min(int((x + L) / r_bin), int(2 * size-1));
        indy = min(int((L - y) / r_bin), int(2 * size-1));
        ind_2d = indy * (2 * size) + indx;
        atomicAdd(&d_concentration_2D[ind_2d], 1);
#endif /* _2D_HISTOGRAM */

#ifdef VELOCITY_VARIANCE
	ind = min(int(r/r_bin), int(size-1));
    	atomicAdd(&d_velocity_variance[ind], w_x * w_x + w_y * w_y);
    	atomicAdd(&d_variance_counter[ind], 1);  
#endif /* VELOCITY_VARIANCE */
    }

    __syncthreads();
}
