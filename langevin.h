#ifndef LANGEVIN_H
#define LANGEVIN_H

#include "input_params.h"
#include <curand.h>
#include <curand_kernel.h>

#define CONCENTRATION
//#define DEBUG

__global__ void setup_kernel(curandState * state, unsigned long seed );

__global__ void numericalProcedure(double *d_concentration, double *d_velocityVariance,
double *d_pdf_vel, double *d_w_autocorrelator, double *d_phi_autocorrelator,
const input_params params, uint64_t size, uint64_t autocorr_size , curandState *state);

#endif /* LANGEVIN_H */