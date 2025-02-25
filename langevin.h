#ifndef LANGEVIN_H
#define LANGEVIN_H

#include "input_params.h"
#include <curand.h>
#include <curand_kernel.h>

#define CONCENTRATION
//#define DEBUG

__global__ void setup_kernel(curandState * state, unsigned long seed );

__global__ void numericalProcedure(uint64_t *d_concentration,
const input_params params, const uint64_t size, curandState *state);

#endif /* LANGEVIN_H */