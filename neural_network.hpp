//
// Created by cartercpp on 9/25/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <cstddef>
#include "math_vector.hpp"
#include "matrix.hpp"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate
    );

    // METHODS

    [[nodiscard]] std::vector<math_vector<double>> forward(const math_vector<double>&) const;
    [[nodiscard]] double predict(const math_vector<double>& input) const;
    math_vector<double> backward(
        const std::vector<math_vector<double>>& activations,
        double delta
    );

    [[nodiscard]] const std::vector<math_vector<double>>& biases() const;
    [[nodiscard]] const std::vector<matrix<double>>& weights() const;

    void update_weights_and_biases();
    void zero_deltas();

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);
    static double TanhDerivative(double);

    std::vector<matrix<double>> m_weightMatrices,
                                m_weightDeltas,
                                m_weightMomentum,
                                m_weightVelocity;
    std::vector<math_vector<double>> m_biasVectors,
                                     m_biasDeltas,
                                     m_biasMomentum,
                                     m_biasVelocity;
    std::vector<std::size_t> m_neuronsPerLayer;
    std::size_t m_adamStep = 0;
    double m_learningRate;
};