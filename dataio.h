#ifndef DATAIO_H
#define DATAIO_H

#include <string>
#include <vector>
#include "input_params.h"

std::vector<std::string> split_sentence(const std::string sen);
enum class ParamType {BoxSize, a, r0, numSteps, dt, unknown};
ParamType readToken(std::string &token);
void readParams(input_params & data, std::string paramsPath);
void printParams(const input_params &data);
void saveHist(double *concentration, const std::string & path, int size);

#endif /* DATAIO_H */