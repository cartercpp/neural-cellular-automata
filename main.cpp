#include <iostream>
#include <print>
#include <random>
#include <numbers>
#include <algorithm>
#include <thread>
#include <stop_token>
#include "neural_network.hpp"

int main()
{
    // generate target cells:
    constexpr int rows = 15,
                  columns = 25,
                  steps = 10;
    constexpr double minX = -16,
                     maxX = 16,
                     minY = -17,
                     maxY = 12;

    double targetCells[rows][columns]{};
    for (double t = 0; t <= 2 * std::numbers::pi; t += 0.01)
    {
        const double x = 16 * std::pow(std::sin(t), 3),
                     y = 13 * std::cos(t) - 5 * std::cos(2 * t) - 2 * std::cos(3 * t) - std::cos(4 * t);

        const int row = rows - 1 - static_cast<int>((y - minY) / (maxY - minY) * (rows - 1)),
                  column = static_cast<int>((x - minX) / (maxX - minX) * (columns - 1));

        targetCells[row][column] = 1;
    }

    // generate random cells which will "evolve" into the target:
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0, 1);

    double cells[rows][columns]{};
    cells[rows / 2][columns / 2] = 1;

    // training stuff:
    neural_network nn({11, 16, 16, 1}, 0.003);

    std::cout << "\033[?25l";
    std::cout << "\033[2J";
    std::jthread thr{[&](std::stop_token st) {
        int epoch = 0;

        while (!st.stop_requested())
        {
            // generate `steps` grids:
            double generatedCells[steps][rows][columns];
            std::vector<math_vector<double>> activations[steps][rows][columns];

            for (int step = 0; step < steps; ++step)
            {
                const auto& source{(step == 0) ? cells : generatedCells[step - 1]};

                for (int row = 0; row < rows; ++row)
                    for (int column = 0; column < columns; ++column)
                    {
                        const math_vector<double> input{
                            ((row > 0) && (column > 0)) ? source[row - 1][column - 1] : 0,
                            (row > 0) ? source[row - 1][column] : 0,
                            ((row > 0) && (column + 1 < columns)) ? source[row - 1][column + 1] : 0,
                            (column > 0) ? source[row][column - 1] : 0,
                            source[row][column],
                            (column + 1 < columns) ? source[row][column + 1] : 0,
                            ((row + 1 < rows) && (column > 0)) ? source[row + 1][column - 1] : 0,
                            (row + 1 < rows) ? source[row + 1][column] : 0,
                            ((row + 1 < rows) && (column + 1 < columns)) ? source[row + 1][column + 1] : 0,
                            row / static_cast<double>(rows - 1),
                            column / static_cast<double>(columns - 1)
                        };

                        activations[step][row][column] = nn.forward(input);
                        const double change = std::tanh(
                            (nn.weights().back() * activations[step][row][column].back() + nn.biases().back())[0]
                        );
                        generatedCells[step][row][column] = source[row][column] + change;
                    }
            }

            // visualize:
            if (epoch % 250 == 0)
            {
                double averageLoss = 0;

                std::cout << "\033[0H";
                for (int row = 0; row < rows; ++row)
                {
                    for (int column = 0; column < columns; ++column)
                    {
                        std::cout << std::format(
                            "\033[38;2;{};{};{}m{}",
                            static_cast<int>(std::clamp(generatedCells[steps - 1][row][column], 0.0, 1.0) * 255),
                            0,
                            0,
                            (generatedCells[steps - 1][row][column] >= 0.5) ? 'o' : ' '
                        );

                        averageLoss += std::abs(generatedCells[steps - 1][row][column] - targetCells[row][column])
                                        / (rows * columns);
                    }
                    std::cout << '\n';
                }
                std::cout << std::format("\033[38;2;255;255;255mTraining Epoch: {}\nAverage Loss: {:.2f}",
                                            epoch, averageLoss) << std::flush;
            }

            // train:
            double cellGradients[steps][rows][columns]{};

            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < columns; ++column)
                {
                    const double error = generatedCells[steps - 1][row][column] - targetCells[row][column];
                    const double weight = ((targetCells[row][column] != 0) ? 8.0 : 1.0) / (rows * columns);
                    cellGradients[steps - 1][row][column] = weight * error;
                }

            for (int step = steps - 1; step >= 0; --step)
                for (int row = 0; row < rows; ++row)
                    for (int column = 0; column < columns; ++column)
                    {
                        const math_vector<double> gradient
                            = nn.backward(activations[step][row][column], cellGradients[step][row][column]);

                        if (step > 0)
                        {
                            if ((row > 0) && (column > 0))
                                cellGradients[step - 1][row - 1][column - 1] += gradient[0];

                            if (row > 0)
                                cellGradients[step - 1][row - 1][column] += gradient[1];

                            if ((row > 0) && (column + 1 < columns))
                                cellGradients[step - 1][row - 1][column + 1] += gradient[2];

                            if (column > 0)
                                cellGradients[step - 1][row][column - 1] += gradient[3];

                            cellGradients[step - 1][row][column] += gradient[4];
                            cellGradients[step - 1][row][column] += cellGradients[step][row][column];

                            if (column + 1 < columns)
                                cellGradients[step - 1][row][column + 1] += gradient[5];

                            if ((row + 1 < rows) && (column > 0))
                                cellGradients[step - 1][row + 1][column - 1] += gradient[6];

                            if (row + 1 < rows)
                                cellGradients[step - 1][row + 1][column] += gradient[7];

                            if ((row + 1 < rows) && (column + 1 < columns))
                                cellGradients[step - 1][row + 1][column + 1] += gradient[8];
                        }
                    }

            nn.update_weights_and_biases();
            ++epoch;
        }
    }};
    std::cin.get();
}