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

__device__ void printArray( double *v, int size)
{
    for(int i = 0; i < size; ++i)
        printf("%f\n", v[i]);
}

__global__ void setup_kernel(curandState * state, unsigned long seed )
{
    int id = threadIdx.x  + blockIdx.x * blockDim.x;
    curand_init(seed, id , 0, &state[id]);
}

// __global__ void calcCorr(std::vector<double> &W_BUFFER, std::vector<double> &autocorrelator)
// {
//     std::cout << "Here calcCorr\n";
//     //uint64_t dt = uint64_t(maxTimeStep / autocorrelator.size()); // p точек через каждые dt
//     const uint64_t dt = 1;
//     const uint64_t autocorr_size = uint64_t(autocorrelator.size());
//     const uint64_t buff_size =  uint64_t(W_BUFFER.size());

//     for(uint64_t k = 0; k < autocorr_size; ++k)
//     {
//         for(uint64_t n = 0; n <buff_size - k * dt; ++n)
//         {
//             autocorrelator[k] += W_BUFFER[n] * W_BUFFER[n + k * dt];
//         }
//     }
// }

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

__global__ void numericalProcedure(double *d_concentration, double *d_velocityVariance,
double *d_pdf_vel, double *d_w_autocorrelator, double *d_phi_autocorrelator,
const input_params params, uint64_t size, uint64_t autocorr_size , curandState *state)
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
    double W5 = 0;
    double W6 = 0;

    int64_t ind = 0;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    double r = params.BoxSize * curand_uniform(&localState);
    printf("idx = %i, r = %f\n", idx, r);

#ifdef DEBUG
    printf("r=%f, sqrt_dt=%f, r_bin=%f, numBins=%i, ind=%i, tau^-1=%f, tau=%f, dt=%f\n", 
    r, sqrt_dt, r_bin, numBins, ind, tau_invert, tau, dt);
#endif

    double w_var = 0;
    double w_r_var = 0;
    uint64_t w_count = 0;

    curand_init(idx, 0, 0, &state[idx]);

    for(uint64_t i = 1; i < params.numSteps; ++i)
    {
        W1 = curand_uniform(&localState) - 0.5;
        W2 = curand_uniform(&localState) - 0.5;
        W3 = curand_uniform(&localState) - 0.5;
        W4 = curand_uniform(&localState) - 0.5;
        W5 = curand_uniform(&localState) - 0.5;
        W6 = curand_uniform(&localState) - 0.5;

        k_r1 = dt * w_r;

        k_wr_1 = - dt_tau_invert * w_r + sqrt_dt * sqrt12 * (W1 * sqrt(D(r,L)) + (W2 * w_r + W3 * w_phi) * sqrt(M(r, L)));

        k_wphi_1 = -dt_tau_invert * w_phi + sqrt_dt * sqrt12 * (W4 * sqrt(D(r,L)) + (W5 * w_r + W6 * w_phi) * sqrt(M(r,L))) - dt * w_r * Sigma(r);

        k_r2 = dt * (w_r + k_wr_1);

        k_wr_2 = - dt_tau_invert * (w_r + k_wr_1) + sqrt_dt * sqrt12 * (W1 * sqrt(D(r+k_r1,L))
         + (W2 * (w_r+k_wr_1) + W3 * (w_phi + k_wphi_1)) * sqrt(M(r + k_r1, L)));

        k_wphi_2 = -dt_tau_invert * (w_phi + k_wphi_1) 
        + sqrt_dt * sqrt12 * (W4 * sqrt(D(r+k_r1,L)) 
        + (W5 * (w_r + k_wr_1) + W6 * (w_phi + k_wphi_1)) * sqrt(M(r+k_r1,L))) 
        - dt * (w_r + k_wr_1) * Sigma(r + k_r1);

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
        d_concentration[ind]++;
#endif // CONCENTRATION
    }
}



