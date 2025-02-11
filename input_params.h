#ifndef INPUT_PARAMS_H
#define INPUT_PARAMS_H

#include <cstdint>

struct input_params
{
    double BoxSize;
    double r_0;
    uint64_t numSteps;
    double dt;
    double a;
};

#endif  /* INPUT_PARAMS_H */