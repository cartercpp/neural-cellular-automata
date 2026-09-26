//
// Created by cartercpp on 9/25/26.
//

#include "neural_network.hpp"
#include <initializer_list>
#include <vector>
#include <random>
#include <utility>
#include <cstddef>
#include <cmath>
#include "math_vector.hpp"
#include "matrix.hpp"

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) * vec[i];

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

double neural_network::TanhDerivative(double x)
{
    return 1 - x * x;
}

std::vector<math_vector<double>> neural_network::forward(const math_vector<double>& input) const
{
    std::vector<math_vector<double>> activations(m_neuronsPerLayer.size() - 1);
    activations[0] = input;

    for (std::size_t i = 1; i < activations.size(); ++i)
        activations[i] = Relu(m_weightMatrices[i - 1] * activations[i - 1] + m_biasVectors[i - 1]);

    return activations;
}

double neural_network::predict(const math_vector<double>& input) const
{
    return std::tanh((m_weightMatrices.back() * forward(input).back() + m_biasVectors.back())[0]);
}

math_vector<double> neural_network::backward(const std::vector<math_vector<double>>& activations, double delta)
{
    for (std::size_t i = 0; i < activations.size(); ++i)
        if (activations[i].size() != m_neuronsPerLayer[i])
            throw std::invalid_argument{"Size of `activations` doesn't match NN architecture"};

    const double prediction = std::tanh((m_weightMatrices.back() * activations.back() + m_biasVectors.back())[0]);
    math_vector<double> gradient(1, delta * TanhDerivative(prediction));

    for (std::size_t row = 0; row < m_weightDeltas.back().rows(); ++row)
        for (std::size_t column = 0; column < m_weightDeltas.back().columns(); ++column)
            m_weightDeltas.back()[row][column] += gradient[row] * activations.back()[column];
    m_biasDeltas.back() += gradient;
    gradient = m_weightMatrices[m_weightMatrices.size() - 1].transpose() * gradient;

    for (std::size_t i = 1; i < m_weightDeltas.size(); ++i)
    {
        gradient = gradient.multiply(ReluDerivative(activations[activations.size() - i]));

        matrix<double>& weightDeltaRef{m_weightDeltas[m_weightDeltas.size() - i - 1]};
        for (std::size_t row = 0; row < weightDeltaRef.rows(); ++row)
            for (std::size_t column = 0; column < weightDeltaRef.columns(); ++column)
                weightDeltaRef[row][column] += gradient[row] * activations[activations.size() - i - 1][column];
        m_biasDeltas[m_biasDeltas.size() - i - 1] += gradient;

        const matrix<double>& weightMatrix{m_weightMatrices[m_weightMatrices.size() - i - 1]};
        math_vector<double> newGradient(weightMatrix.columns(), 0);
        for (std::size_t column = 0; column < weightMatrix.columns(); ++column)
        {
            double value = 0;

            for (std::size_t row = 0; row < weightMatrix.rows(); ++row)
                value += weightMatrix[row][column] * gradient[row];

            newGradient[column] = value;
        }

        gradient = std::move(newGradient);
    }

    return gradient;
}

void neural_network::update_weights_and_biases()
{
    constexpr double beta1 = 0.9,
                     beta2 = 0.999,
                     epsilon = 1e-8;

    ++m_adamStep;

    const double beta1Correction = 1 - std::pow(beta1, m_adamStep),
                 beta2Correction = 1 - std::pow(beta2, m_adamStep);

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        matrix<double>& weightMatrixRef{m_weightMatrices[i]};
        math_vector<double>& biasVectorRef{m_biasVectors[i]};

        for (std::size_t row = 0; row < weightMatrixRef.rows(); ++row)
            for (std::size_t column = 0; column < weightMatrixRef.columns(); ++column)
            {
                const double gradient = m_weightDeltas[i][row][column];

                m_weightMomentum[i][row][column]
                    = beta1 * m_weightMomentum[i][row][column] + (1 - beta1) * gradient;
                m_weightVelocity[i][row][column]
                    = beta2 * m_weightVelocity[i][row][column] + (1 - beta2) * gradient * gradient;

                const double correctedMomentum = m_weightMomentum[i][row][column] / beta1Correction,
                             correctedVelocity = m_weightVelocity[i][row][column] / beta2Correction;

                m_weightMatrices[i][row][column] -= m_learningRate * correctedMomentum
                                                    / (std::sqrt(correctedVelocity) + epsilon);
            }
    }

    for (std::size_t i = 0; i < m_biasVectors.size(); ++i)
        for (std::size_t i2 = 0; i2 < m_biasVectors[i].size(); ++i2)
        {
            const double gradient = m_biasDeltas[i][i2];

            m_biasMomentum[i][i2] = beta1 * m_biasMomentum[i][i2] + (1 - beta1) * gradient;
            m_biasVelocity[i][i2] = beta2 * m_biasVelocity[i][i2] + (1 - beta2) * gradient * gradient;

            const double correctedMomentum = m_biasMomentum[i][i2] / beta1Correction,
                         correctedVelocity = m_biasVelocity[i][i2] / beta2Correction;

            m_biasVectors[i][i2] -= m_learningRate * correctedMomentum / (std::sqrt(correctedVelocity) + epsilon);
        }

    zero_deltas();
}

void neural_network::zero_deltas()
{
    for (std::size_t i = 0; i < m_weightDeltas.size(); ++i)
        for (std::size_t row = 0; row < m_weightDeltas[i].rows(); ++row)
        {
            for (std::size_t column = 0; column < m_weightDeltas[i].columns(); ++column)
                m_weightDeltas[i][row][column] = 0;

            m_biasDeltas[i][row] = 0;
        }
}

neural_network::neural_network(std::initializer_list<std::size_t> neuronsPerLayer, double learningRate)
    : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_biasVectors.reserve(m_neuronsPerLayer.size() - 1);

    std::mt19937 rng{std::random_device{}()};

    for (std::size_t layer = 1; layer < m_neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = m_neuronsPerLayer[layer],
                          prevLayerSize = m_neuronsPerLayer[layer - 1];

        std::normal_distribution<double> dist(0, std::sqrt(2.0 / static_cast<double>(prevLayerSize)));

        matrix<double> weights(layerSize, prevLayerSize, 0);
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rng);

        m_weightMatrices.emplace_back(std::move(weights));
        m_weightDeltas.emplace_back(layerSize, prevLayerSize, 0);
        m_weightMomentum.emplace_back(layerSize, prevLayerSize, 0);
        m_weightVelocity.emplace_back(layerSize, prevLayerSize, 0);
        m_biasVectors.emplace_back(layerSize, 0);
        m_biasDeltas.emplace_back(layerSize, 0);
        m_biasMomentum.emplace_back(layerSize, 0);
        m_biasVelocity.emplace_back(layerSize, 0);
    }
}

const std::vector<math_vector<double>>& neural_network::biases() const
{
    return m_biasVectors;
}

const std::vector<matrix<double>>& neural_network::weights() const
{
    return m_weightMatrices;
}