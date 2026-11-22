#include <Layer.h>
#include<stdexcept>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}
//Task 5a
void Layer::forwardPropagation(
    const std::vector<double>& previousLayerOutput) {
    
    preActivationOutput.assign(layerOutputSize, 0.0);
    activationOutput.assign(layerOutputSize, 0.0);

for(std::size_t i = 0; i < layerOutputSize; ++i) {
        double weightedSum = 0.0;

        for(std::size_t j = 0; j < layerInputSize; ++j) {
            weightedSum =weightedSum+ weights[i][j] * previousLayerOutput[j];
        }
        preActivationOutput[i] = weightedSum + bias[i];
        activationOutput[i] = activationFunction->activation(preActivationOutput[i]);
    }
}

//5b
void Layer::backwardPropagation(
    const std::vector<double>& dCurrentLayerOutput,
    const std::vector<double>& previousLayerOutput) {
    
    if(activationFunction == nullptr) {
        throw std::logic_error("Layer has no activation function");
    }
    //gradient calcultaion before activation
    std::vector<double> dPreActivation(layerOutputSize, 0.0);

    for(std::size_t i = 0; i < layerOutputSize; ++i) {
        double slope = activationFunction->activationBackwards(preActivationOutput[i]);
        dPreActivation[i] = dCurrentLayerOutput[i] * slope;
    }

//gradient calculation for each weight 
   

dWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    for(std::size_t i = 0; i < layerOutputSize; ++i) {
        for(std::size_t j = 0; j < layerInputSize; ++j) {
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    
    }

    //store the bias gradient
    dBias= dPreActivation;

    //Add the samples gradient to the accumulated total
    

for(std::size_t i = 0; i < layerOutputSize; ++i) {
        for(std::size_t j = 0; j < layerInputSize; ++j) {
            accumulatedDWeights[i][j] = accumulatedDWeights[i][j] + dWeights[i][j];
        }
        accumulatedDBias[i] = accumulatedDBias[i] + dBias[i];
    }
    numProcessedSamples++;

    //Calculate the gradient for the previous layer's output
    dPreviousLayerOutput.assign(layerInputSize, 0.0);
    for(std::size_t j = 0; j < layerInputSize; ++j) {
        for(std::size_t i = 0; i < layerOutputSize; ++i) {
            dPreviousLayerOutput[j] += weights[i][j] * dPreActivation[i];
        }
    }

}



//5c
void Layer::updateParameters(
    const std::vector<std::vector<double>>& dWeights,
    const std::vector<double>& dBias,
    double learningRate) {

    for (std::size_t i = 0; i < layerOutputSize; i++) {

        //Update the weights of this neuron
        for (std::size_t j = 0; j < layerInputSize; j++) {
            weights[i][j] =
                weights[i][j] - learningRate * dWeights[i][j];
        }

        //Update the bias once per neuron
        bias[i] = bias[i] - learningRate * dBias[i];
    }
}

//5d
void Layer::commitToWeights(double learningRate) {

    //Stop if no samples have been processed
    if (numProcessedSamples == 0) {
        return;
    }

    double sampleCount = static_cast<double>(numProcessedSamples);

    //Calculate the average gradients
    for (std::size_t i = 0; i < layerOutputSize; i++) {
        for (std::size_t j = 0; j < layerInputSize; j++) {
            accumulatedDWeights[i][j] =
                accumulatedDWeights[i][j] / sampleCount;
        }

        accumulatedDBias[i] =
            accumulatedDBias[i] / sampleCount;
    }

    //Apply the update using the method from 5c
    updateParameters(
        accumulatedDWeights, accumulatedDBias, learningRate);

    //Reset the totals for the next batch
    for (std::size_t i = 0; i < layerOutputSize; i++) {
        for (std::size_t j = 0; j < layerInputSize; j++) {
            accumulatedDWeights[i][j] = 0.0;
        }

        accumulatedDBias[i] = 0.0;
    }

    numProcessedSamples = 0;
}




uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}

}
