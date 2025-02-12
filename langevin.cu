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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    double r = params.BoxSize * curand_uniform(&localState);
    printf("idx = %i, r = %f\n", idx, r);

    double dr = 0;
    double phi = 0;
    double w = 0;
    double dphi = 0;
    double dw = 0;
    double kr1 = 0;
    double kr2 = 0;
    double kw1 = 0;
    double kw2 = 0;
    double kphi1 = 0;
    double kphi2 = 0;
    double sqrt12 = sqrt((float)12);
    double sqrt_dt = 0;
    double dt = 0;
    double r_bin = params.BoxSize / size;
    double W1 = 0;
    double W2 = 0;
    uint64_t numBins = size;
    uint64_t ind = 0;
    uint64_t ind_pdf = 0;

    double tau_invert = pow(params.BoxSize, 2) / (pow(params.a,2) * Re);
    double tau = 1/tau_invert;
    double L = params.BoxSize;
    dt = tau/10;
    sqrt_dt = sqrt(dt);
    double dt_tau_invert = dt * tau_invert;
    double sqrt_dt_12 = sqrt_dt * sqrt12;

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
        
        kr1 = dt * w * sin(phi);

        kw1 = - dt_tau_invert * w + sqrt_dt_12 * W1 * sqrt(D(r, L));

        kphi1 = Sigma(r) * pow(sin(phi),2) + sqrt_dt_12 * W2 * sqrt(M(r, L));

        kr2 = dt * (w + kw1) * sin(phi + kphi1);

        kw2 = - dt_tau_invert * (w + kw1) + sqrt_dt_12 * W1 * sqrt(D(r + kr1, L));

        kphi2 = Sigma(r + kr1) * pow(sin(phi + kphi1),2) + sqrt_dt_12 * W2 * sqrt(M(r+kr1, L));

        dr = 0.5 * (kr1 + kr2);
        dw = 0.5 * (kw1 + kw2);
        dphi = 0.5 * (kphi1 + kphi2);

        if(r + dr > L)
            r = 2 * L - r - dr;
        else if (r + dr < 0)
            r = - r - dr;
        else
            r = r + dr;

        w = abs(w + dw);
        phi = phi + dphi;

#ifdef CONCENTRATION
        ind = min(uint64_t(r / r_bin), numBins-1);
        d_concentration[ind]++;
#endif // CONCENTRATION
    }
}



