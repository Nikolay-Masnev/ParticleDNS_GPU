#include "dataio.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<std::string> split_sentence(const std::string sen);
ParamType readToken(std::string &token);
void readParams(input_params & data, std::string paramsPath);
void printParams(const input_params &params);

std::vector<std::string> split_sentence(const std::string sen) {
  
    std::vector<std::string> words;

    std::string word = "";

    for (char c : sen) {
        if (c == ' ') {
            words.push_back(word);
            word = "";
        }
        else {
            word += c;
        }
    }

    if (!word.empty()) {
        words.push_back(word);
    }

    return words;
}

ParamType readToken(std::string &token)
{
    ParamType type;

    if(token == "BoxSize")
        type = ParamType::BoxSize;
    else if (token == "r0")
        type = ParamType::r0;
    else if (token == "steps")
        type = ParamType::numSteps;
    else if (token == "dt")
        type = ParamType::dt;
    else if(token == "a")
        type = ParamType::a;
    else
    {
        std::cout << "CANNOT PARSE PARAMS FILE!\n";
        type = ParamType::unknown;
        throw std::invalid_argument( "received incorrect file\n" );
    }

    return type;
}

void readParams(input_params & data, std::string paramsPath)
{
    std::ifstream file(paramsPath);
    std::cout << paramsPath;
    std::string line;
    std::string token;
    ParamType tmpParam;
    double tmpValue;

    if(file.is_open())
    {
        while(std::getline(file, line))
        {
            token = line.substr(0, line.find(' '));
            tmpParam = readToken(token);
            tmpValue = std::stod(split_sentence(line)[2]);

            switch (tmpParam)
            {
            case ParamType::BoxSize:
                data.BoxSize = tmpValue;
                break;
            case ParamType::r0:
                data.r_0 = tmpValue;
                break;
            case ParamType::a:
                data.a = tmpValue;
                break;
            case ParamType::numSteps:
                data.numSteps = uint64_t(tmpValue);
                break;
            case ParamType::dt:
                data.dt = tmpValue;
                break;
            case ParamType::unknown:
            default:
                break;
            }
        }
    }
    else{
        std::cout << "CANNOT OPEN PARAMS FILE!\n";
    }

    printParams(data);
}

void printParams(const input_params &params)
{
    std::cout << "Box Size = " << params.BoxSize << '\n';
    std::cout << "r_0 = " << params.r_0 << '\n';
    std::cout << "steps = " << params.numSteps << '\n';
    std::cout << "dt = " << params.dt << '\n';
    std::cout << "a = " << params.a << '\n';
}

void saveHist(double *concentration, const std::string & path, int size)
{
    std::ofstream file (path);
    if (file.is_open())
    {
        for(int count = 0; count < size; count ++){
            file << concentration[count] << " " ;
        }

    file.close();
    }

    else std::cout << "Unable to open file";
}
