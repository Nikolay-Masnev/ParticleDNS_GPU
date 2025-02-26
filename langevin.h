#ifndef LANGEVIN_H
#define LANGEVIN_H

#include "input_params.h"
#include <curand.h>
#include <curand_kernel.h>

#define CONCENTRATION
//#define DEBUG

__global__ void setup_kernel(curandState * state, unsigned long seed );

__global__ void numericalProcedure(unsigned long long int *d_concentration,
const input_params params, const unsigned long long int size, curandState *state);

#endif /* LANGEVIN_H */