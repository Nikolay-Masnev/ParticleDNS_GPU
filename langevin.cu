#include "langevin.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "helper_functions.h" 

const double r_pdf = 3;
const double dr_pdf = 0.1;
const double w_max = 1;
const double Re = 2500;
const uint64_t BUFFER_SIZE = 1e7;
const double maxTimeStep = 1e5;
const double twoPi = 2.0 * 3.141592865358979;

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

__global__ void calcCorr(std::vector<double> &W_BUFFER, std::vector<double> &autocorrelator)
{
    std::cout << "Here calcCorr\n";
    //uint64_t dt = uint64_t(maxTimeStep / autocorrelator.size()); // p точек через каждые dt
    const uint64_t dt = 1;
    const uint64_t autocorr_size = uint64_t(autocorrelator.size());
    const uint64_t buff_size =  uint64_t(W_BUFFER.size());

    for(uint64_t k = 0; k < autocorr_size; ++k)
    {
        for(uint64_t n = 0; n <buff_size - k * dt; ++n)
        {
            autocorrelator[k] += W_BUFFER[n] * W_BUFFER[n + k * dt];
        }
    }
}

__global__ void numericalProcedure(double *d_concentration, double *d_velocityVariance,
double *d_pdf_vel, double *d_w_autocorrelator, double *d_phi_autocorrelator,
const input_params params, uint64_t size, uint64_t autocorr_size , curandState *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    double r = params.BoxSize * curand_uniform(&localState);
    printf("idx = %i, r = %f\n", idx, r);

    double r = params.r_0;
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
    double sqrt12 = sqrt(12);
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
    dt = tau/10;
    sqrt_dt = sqrt(dt);
    double D_sqrt = sqrt(10);
    double Sigma = 10;

    uint64_t *counter = 0;
    checkCudaErrors(cudaMalloc((void **)&counter, size * sizeof(uint64_t)));
    memset(counter, 0, size * sizeof(uint64_t));

#ifdef DEBUG
    printf("r=%f, sqrt_dt=%f, r_bin=%f, numBins=%i, ind=%i, tau^-1=%f, tau=%f, dt=%f\n", 
    r, sqrt_dt, r_bin, numBins, ind, tau_invert, tau, dt);
#endif

    double *W_BUFFER = 0;
    double *PHI_BUFFER = 0;

    uint64_t nBytes = BUFFER_SIZE * sizeof(double);

    checkCudaErrors(cudaMalloc((void **)&W_BUFFER, nBytes));
    checkCudaErrors(cudaMalloc((void **)&PHI_BUFFER, nBytes));

    memset(W_BUFFER, 0, nBytes);
    memset(PHI_BUFFER, 0, nBytes);

    uint64_t tmp_ind = 0;
    uint64_t phi_tmp_ind = 0;  

    int autocorr_counter = 0;
    int phi_autocorr_counter = 0;

    double w_var = 0;
    double w_r_var = 0;
    int64_t w_count = 0;

    curand_init(idx, 0, 0, &state[idx]);

    for(uint64_t i = 1; i < params.numSteps; ++i)
    {
        W1 = curand_uniform(&localState) - 0.5;
        W2 = curand_uniform(&localState) - 0.5;
        
        kr1 = dt * w * sin(phi);

        kw1 = - dt * tau_invert * w + D_sqrt * sqrt_dt * W1 
        * sqrt12 * sqrt(0.1 + pow(r/params.BoxSize, 2));

        kphi1 = Sigma * (0.2 * tanh(0.5 * r) 
        - 0.1 * tanh(0.1 * r)) * pow(sin(phi),2) 
        + sqrt_dt * W2 * sqrt12 * sqrt(0.1 + pow(r/params.BoxSize, 2));

        kr2 = dt * (w + kw1) * sin(phi + kphi1);

        kw2 = - dt * tau_invert * (w + kw1) 
        + D_sqrt * sqrt_dt * W1 * sqrt12 * sqrt(0.1 + pow((r + kr1)/params.BoxSize, 2));

        kphi2 = Sigma * (0.2 * tanh(0.5 * (r + kr1)) 
        - 0.1 * tanh(0.1 * (r + kr1))) * pow(sin(phi + kphi1),2) 
        + sqrt_dt * W2 * sqrt12 * sqrt(0.1 + pow((r + kr1)/params.BoxSize, 2));

        dr = 0.5 * (kr1 + kr2);
        dw = 0.5 * (kw1 + kw2);
        dphi = 0.5 * (kphi1 + kphi2);

        if(r + dr > params.BoxSize)
            r = 2 * params.BoxSize - r - dr;
        else if (r + dr < 0)
            r = - r - dr;
        else
            r = r + dr;

        w = abs(w + dw);
        phi = phi + dphi;

#ifdef CONCENTRATION
        ind = min(int64_t(r / r_bin), numBins-1);
        d_concentration[ind]++;
#endif // CONCENTRATION

#ifdef PDF_VELOCITY
        if (r > r_pdf - dr_pdf && r < r_pdf + dr_pdf)
        {
            ind_pdf = min(int64_t(abs(w * sin(phi)) * size/w_max), int64_t(size-1));
            d_pdf_vel[ind_pdf]++;

            w_var += pow(w, 2);
            w_r_var += pow(w * sin(phi), 2);
            w_count++;
        }
#endif // PDF_VELOCITY

#ifdef VELOCITY_STAT
        d_velocityVariance[ind] += pow(w*sin(phi),2);
        counter[ind]++;
#endif // VELOCITY_STAT

#ifdef AUTOCORRELATION
        if (r > r_pdf - dr_pdf && r < r_pdf + dr_pdf)
        {   
            if(tmp_ind >= BUFFER_SIZE-1)
            {
                tmp_ind = 0;
                autocorr_counter++;
                calcCorr(W_BUFFER, w_autocorrelator);
                std::fill(W_BUFFER.begin(), W_BUFFER.end(), 0.0);
            }
            else
            {
                W_BUFFER[tmp_ind] = w * sin(phi);
                tmp_ind++;
            }

            if(phi_tmp_ind >= BUFFER_SIZE-1)
            {
                phi_tmp_ind = 0;
                phi_autocorr_counter++;
                calcCorr(PHI_BUFFER, phi_autocorrelator);
                std::fill(PHI_BUFFER.begin(), PHI_BUFFER.end(), 0.0);
            }
            else
            {
                PHI_BUFFER[phi_tmp_ind] = phi;
                phi_tmp_ind++;
            }
        }
#endif // AUTOCORRELATION
    }

#ifdef CONCENTRATION
    normalizePDF(concentration);
#endif // CONCENTRATION

#ifdef VELOCITY_STAT
    for(uint64_t i = 0; i < uint64_t(size); ++i)
    {
        velocityVariance[i] /= (counter[i]+1);
        velocityVariance[i] = sqrt(velocityVariance[i]);
    }
#endif // VELOCITY_STAT

#ifdef PDF_VELOCITY
    normalizePDF(pdf_vel);
    std::cout << "dispersion of w: " << std::sqrt(w_var/w_count) << '\n'; 
    std::cout << "dispersion of w_r: " << std::sqrt(w_r_var/w_count) << '\n'; 
#endif // PDF_VELOCITY

#ifdef AUTOCORRELATION
    normCorr(W_BUFFER, w_autocorrelator, autocorr_counter);
    normCorr(PHI_BUFFER, phi_autocorrelator, phi_autocorr_counter);
#endif // AUTOCORRELATION
}