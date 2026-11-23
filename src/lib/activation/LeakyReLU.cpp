#include <activation/LeakyReLU.h>


namespace dnn{

double LeakyReLU::activation(double x) {
    if(x>0) return x;
    else return x * 0.01;
}

double LeakyReLU::activationBackwards(double x){
    if(x>0) return 1;
    else return 0.01;
}

std::string LeakyReLU::getName() const {
    return "leaky_relu";
}

}