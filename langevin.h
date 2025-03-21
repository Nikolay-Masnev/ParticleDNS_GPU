#ifndef LANGEVIN_H
#define LANGEVIN_H

#include "input_params.h"
#include <curand.h>
#include <curand_kernel.h>

#define CONCENTRATION
//#define TRAJECTORY

__global__ void setup_kernel(curandState * state, unsigned long seed );

/*__global__ void numericalProcedure(unsigned long long int *d_concentration,
const input_params params, const unsigned long long int size, curandState *state);*/

__global__ void numericalProcedure(unsigned long long int *d_concentration,
const input_params params, const unsigned long long int size, curandState *state,
float *d_tr_x, float *d_tr_y, float *d_tr_wx, float *d_tr_wy, unsigned long long int tr_points);

#endif /* LANGEVIN_H */
