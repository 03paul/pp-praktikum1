#include <Network.h>

#include <iostream>
#include <stdexcept>

#include <utility>

#include <loss/LossFunction.h>

namespace dnn {

Network::Network(
    std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters, 
    LossFunction* lossFunction
) : ownedLossFunction(lossFunction) {

    for(auto& parameters : layerParameters) {

        auto weights = std::move(std::get<0>(parameters)); // Gewichte
        auto bias = std::move(std::get<1>(parameters)); // Bias
        auto* activation = std::get<2>(parameters); // Aktivierungsfunktion

        uint64_t inputSize = weights.at(0).size();
        uint64_t outputSize = weights.size();

        Layer* layer = new Layer(
            layers.size(),
            nextSeed++,
            inputSize,
            outputSize,
            activation
        );

        layer->setParameters(
            std::move(weights),
            std::move(bias)
        );

        layers.push_back(layer);

    }

}

Network::~Network() {
    for(Layer* layer : layers) {
        delete layer;
    }

    delete ownedLossFunction;
}

Network& Network::addLayer(
    uint64_t inputDim,
    uint64_t outputDim,
    ActivationFunction* activationFunction
) {

    Layer* layer = new Layer(
        layers.size(),
        nextSeed,
        inputDim,
        outputDim,
        activationFunction
    );

    layer->init();

    layers.push_back(layer);

    ++nextSeed;

    return *this;

}

const std::vector<double>& Network::getLayerInput( // per & nur die Referenz zurückgeben
    uint64_t layerId) const {

        layers.at(layerId); // existiert Layer bereits? Sonst std::out_of_range-Exception.

        if(layerId == 0) return initialInput;

        return layers.at(layerId - 1)->getActivationOutput();

}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    initialInput = input;

    const std::vector<double>* currentInput = &initialInput;

    for(Layer* layer : layers) {
        layer->forwardPropagation(*currentInput);

        currentInput = &layer->getActivationOutput();
    }

    output = *currentInput;
}

void Network::fullBackwardPropagation(
    const std::vector<double>& predictedOutput,
    const std::vector<double>& actualOutput
) {
    if(ownedLossFunction == nullptr) {
        throw std::logic_error("");
    }

    // Initialer Gradient
    std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);

    // Alle Layer rückwärts durchlaufen
    for(std::size_t i = layers.size(); i > 0; --i) {
        Layer* layer = layers[i - 1];

        const std::vector<double>& layerInput = getLayerInput(i - 1);

        // Gradienten für aktuellen Layer berechnen
        layer->backwardPropagation(dACurr, layerInput);


        // Gradient für vorherigen Layer nehmen
        dACurr = layer->getDPreviousLayerOutput();
    }
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

void Network::trainBatch(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& actualOutputs,
    std::size_t batchSize,
    int epochs,
    double learningRate
) {
    // Eingaben validieren
    if(ownedLossFunction == nullptr) {
        throw std::logic_error("Keine Lossfunktion gesetzt!");
    }

    if(inputs.empty() || inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Ungültige Trainingsdaten");
    }

    if(batchSize == 0 || epochs == 0) {
        throw std::invalid_argument("Ungültige Batchgröße oder Epochenzahl");
    }

    if(layers.empty() || layers.front()->getWeights().empty()) {
        throw std::invalid_argument("Network hat keine initialisierten Schichten");
    }

    const std::size_t inputSize = layers.front()->getWeights().front().size();

    const std::size_t outputSize = layers.back()->getWeights().size();

    for(std::size_t i = 0; i < inputs.size(); ++i) {
        if(inputs[i].empty() || inputs[i].size() != inputSize || actualOutputs[i].empty() || actualOutputs[i].size() != outputSize) {
            throw std::invalid_argument("Inkompatible Sampledimensionen");
        }
    }

    // Cost- und Accuracy History leeren
    costHistory.clear();
    accuracyHistory.clear();

    // Training über mehrere Epochs
    for(int epoch = 0; epoch < epochs; ++epoch) {

        double epochCost = 0.0;
        double epochAccuracy = 0.0;

        for(std::size_t i = 0; i < inputs.size(); ++i) {

            fullForwardPropagation(inputs[i]);

            // Kosten und Acc akkumulieren
            epochCost += ownedLossFunction->getCostValue(output, actualOutputs[i]);

            epochAccuracy += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);

            // Backward Propagation
            fullBackwardPropagation(output, actualOutputs[i]);

            // nach jedem Batch aktualisieren
            if((i + 1) % batchSize == 0) {
                for(Layer* layer : layers) {
                    layer->commitToWeights(learningRate);
                }
            }

            // Durchschnittliche Epochenwerte speichern
            costHistory.push_back(epochCost / static_cast<double>(inputs.size()));

            accuracyHistory.push_back(epochAccuracy / static_cast<double>(inputs.size()));

        }
    };


}

double Network::computeAccuracy(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& actualOutputs
) {
    if(ownedLossFunction == nullptr) {
        throw std::logic_error("Keine Loss Function gesetzt");
    }

    if(inputs.empty() || inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Ungültige Evaluationsdaten");
    }

    if(layers.empty() || layers.front()->getWeights().empty()) throw std::invalid_argument("Keine initialisierten Layer");

    const std::size_t inputSize = layers.front()->getWeights().size();
    const std::size_t outputSize = layers.back()->getWeights().size();

    double totalCorrect = 0.0;

    for(std::size_t i = 0; i < inputs.size(); ++i) {
        if(inputs[i].empty() || inputs[i].size() != inputSize || actualOutputs[i].empty() || actualOutputs[i].size() != outputSize) {
            throw std::invalid_argument("Inkompatible Sampledimensionen");
        }

        fullForwardPropagation(inputs[i]);

        totalCorrect += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
    }

    return totalCorrect / static_cast<double>(inputs.size());
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}
