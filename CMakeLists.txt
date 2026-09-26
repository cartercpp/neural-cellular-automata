cmake_minimum_required(VERSION 4.3)
project(NeuralCellularAutomaton)

set(CMAKE_CXX_STANDARD 23)

add_executable(NeuralCellularAutomaton main.cpp
        neural_network.hpp
        neural_network.cpp
        math_vector.hpp
        matrix.hpp)
