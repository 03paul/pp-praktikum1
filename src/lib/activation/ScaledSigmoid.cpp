#include <activation/ScaledSigmoid.h>
#include <cmath>

namespace dnn{

double ScaledSigmoid::activation (double x){

double result = 1.0 / (1.0 + std::exp(-x));
return min + (max - min) * result;
}

double ScaledSigmoid::activationBackwards (double x){
double result = 1.0 / (1.0 + std::exp(-x));
return (max - min) * result * (1 - result);
}

std::string ScaledSigmoid::getName() const{
return "scaled_sigmoid";
}

double ScaledSigmoid::getMax() const{
    return max;
}

double ScaledSigmoid::getMin() const{
    return min;
}

ScaledSigmoid::ScaledSigmoid(double min, double max)
    : min(min), max(max) {
}

}

