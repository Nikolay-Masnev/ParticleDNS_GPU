#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

__constant__ double Re = 2500;
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

__device__ double Sigma(double r)
{
    return 10 * (0.2 * tanh(0.5 * r) - 0.1 * tanh(0.1 * r));
}

__device__ double D(double r, double L)
{
    return 0.1 * (1 + pow(r/L, 2));
}

__device__ double div_D(double x, double y, double L)
{
    return 0.1 * (x + y) / (2 * L * L);
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
    double r_bin = L / size;
    double sqrt12 = sqrt((float)12);
    double tau_invert = pow(L, 2) / (pow(a,2) * Re);
    double tau = 1/tau_invert;
    double dt = tau / 10;
    double sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;
    double sqrt_dt_12 = sqrt_dt * sqrt12;

    double dx = 0, dy = 0, dw_x = 0, dw_y = 0, x = 0, y = 0, w_x = 0, w_y = 0, r = 0, phi = 0;
    double c_x_0 = 0, c_x_1 = 0, b_x_0 = 0, b_x_1 = 0, b_x_2 = 0, b_x_3 = 0;
    double c_y_0 = 0, c_y_1 = 0, b_y_0 = 0, b_y_1 = 0, b_y_2 = 0, b_y_3 = 0;
    double c_wx_0 = 0, c_wx_1 = 0, b_wx_0 = 0, b_wx_1 = 0, b_wx_2 = 0, b_wx_3 = 0;
    double c_wy_0 = 0, c_wy_1 = 0, b_wy_0 = 0, b_wy_1 = 0, b_wy_2 = 0, b_wy_3 = 0;

    unsigned long long int steps = params.numSteps;
    unsigned long long int ind = 0;

    r = L * curand_uniform(&localState);
    phi = 2 * M_PI * curand_uniform(&localState);
    x = r * cos(phi);
    y = r * sin(phi);

    double W1 = 0;
    double W2 = 0;
    double W3 = 0;
    double W4 = 0;

    double W1_old = 0;
    double W2_old = 0;
    double W3_old = 0;
    double W4_old = 0;
    
    double dW1 = 0;
    double dW2 = 0;
    double dW3 = 0;
    double dW4 = 0;

    double rho = 0;
    double sqrt_one_rho = 0;

    __syncthreads();

    for(unsigned long long int i = 0; i < steps; ++i)
    {   
        // rho = exp(-dt/tau_corr(r, L));
        // sqrt_one_rho = sqrt(1 - rho * rho);
        rho = 0;
        sqrt_one_rho = 1;

        W1_old = W1;
        W2_old = W2;
        W3_old = W3;
        W4_old = W4;

        W1 = W1_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W2 = W2_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W3 = W3_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);
        W4 = W4_old * rho + sqrt_one_rho * (curand_uniform(&localState) - 0.5);

        dW1 = (W1 - W1_old) * sqrt_dt_12;
        dW2 = (W2 - W2_old) * sqrt_dt_12;
        dW3 = (W3 - W3_old) * sqrt_dt_12;
        dW4 = (W4 - W4_old) * sqrt_dt_12;

        c_x_0 =  - div_D(x, y, L);
        c_y_0 =  - div_D(x, y, L);

        b_x_0 = sqrt(D(r, L));
        b_y_0 = b_x_0;

        b_x_1 = sqrt(D(sqrt(pow(x + 0.5 * b_x_0 * dW1,2) + pow(y + 0.5 * b_y_0 * dW2,2)), L));
        b_y_1 = b_x_1;

        b_x_2 = sqrt(D(sqrt(pow(x + 0.25 * c_x_0 * (3 * dt + dW1*dW1) + 0.5 * b_x_1 * dW1,2) 
        + pow(y + 0.5 *  0.25 * c_y_0 * (3 * dt + dW2*dW2) + 0.5 * b_y_1 * dW2,2)), L));
        b_y_2 = b_x_2;

        b_x_3 = sqrt(D(sqrt(pow(x + 0.5 * c_x_0 * (3 * dt - dW1*dW1) + b_x_2 * dW1,2) 
        + pow(y + 0.5 * c_y_0 * (3 * dt - dW2*dW2) + b_y_2 * dW2,2)), L));
        b_y_3 = b_x_3;

        c_x_1 = - div_D(x + 0.5 * c_x_0 * (3 * dt - dW1*dW1) + b_x_2 * dW1, y + 0.5 * c_y_0 * (3 * dt - dW2*dW2) + b_y_2 * dW2, L);
        c_y_1 = - div_D(x + 0.5 * c_x_0 * (3 * dt - dW1*dW1) + b_x_2 * dW1, y + 0.5 * c_y_0 * (3 * dt - dW2*dW2) + b_y_2 * dW2, L);

        dx = 0.5 * (c_x_0 + c_x_1) * dt + (b_x_0 +  2 * b_x_1 + 2 * b_x_2 + b_x_3) * dW1 / 6;
        dy = 0.5 * (c_y_0 + c_y_1) * dt + (b_y_0 +  2 * b_y_1 + 2 * b_y_2 + b_y_3) * dW2 / 6;

        x += dx;
        y += dy;
        r = sqrt(x*x + y*y);

        if(r > L)
        {
            x -= 2 * dx;
            y -= 2 * dy;

            // w_x = (8 * x * y * w_y) / (L*L) - (4*(x*x -y*y)*w_x)/(L*L);
            // w_y = (4*(x*x - y*y)*w_x)/(L*L) + (8*x*y*w_y)/(L*L); 
        }

        r = sqrt(x*x + y*y);

#ifdef CONCENTRATION
        ind = min(int(r / r_bin), int(size-1));
        atomicAdd(&d_concentration[ind], 1);

	    if(int(r/r_bin) > size)
        {
            printf("r = %f\n", r);
        }
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
