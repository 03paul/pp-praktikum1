#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dnn {

double BinaryCrossEntropy::getCostValue(
    const std::vector<double>& predictedOutput,
    const std::vector<double>& actualOutput
) const {

    const std::size_t n = predictedOutput.size();

    if(n == 0 || n != actualOutput.size()) {
        throw std::invalid_argument("Ungültige Outputdimensionen");
    }

    std::vector<double> P(n, 0.0);
    std::vector<double> Q(n, 0.0);
    std::vector<double> R(n, 0.0);

    for(std::size_t i = 0; i < n; ++i) {
        double prediction = clampProbability(predictedOutput[i]);

        P[i] = std::log(prediction);
        Q[i] = std::log(1 - prediction);
        R[i] = 1 - actualOutput[i];
    }

    double cost = -(
        dotProduct(actualOutput, P) +
        dotProduct(R, Q)
    ) / static_cast<double>(n);

    return cost;
    
}

double BinaryCrossEntropy::getAccuracyValue(
    const std::vector<double>& predictedOutput,
    const std::vector<double>& actualOutput
) const {
    const std::size_t n = predictedOutput.size();

    if(n == 0 || n != actualOutput.size()) {
        throw std::invalid_argument("Ungüueltige Ausgabedimensionen");
    }

    double correct = 0.0;

    for(std::size_t i = 0; i < n; ++i) {
        double prediction = predictedOutput[i] > 0.5 ? 1.0 : 0.0;

        if(prediction == actualOutput[i]) correct += 1.0;
    }

    return correct / static_cast<double>(n);
}

std::vector<double> BinaryCrossEntropy::getGradient(
    const std::vector<double>& predictedOutput,
    const std::vector<double>& actualOutput
) const {
    const std::size_t n = predictedOutput.size();

    if(n == 0 || n != actualOutput.size()) {
        throw std::invalid_argument("Ungültiuge Output Dimensions");
    }

    std::vector<double> gradient(n, 0.0);

    for(std::size_t i = 0; i < n; ++i) {
        double prediction = clampProbability(predictedOutput[i]);

        double actual = actualOutput[i];

        gradient[i] = -(
            actual / prediction
            - (1.0 - actual) / (1.0 - prediction)
        ) / static_cast<double>(n);
    }

    return gradient;
}

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

}
