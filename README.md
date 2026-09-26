# Neural Cellular Automata

A neural cellular automata experiment written from scratch in C++. A small neural network learns local update rules that evolve a single active cell into a heart-shaped target over multiple simulation steps.

Each cell uses its 3x3 neighborhood along with normalized row and column coordinates as input to the neural network. Training is visualized directly in the terminal.

## Build

Requires a C++23-compatible version of GCC.

```bash
g++ -std=c++23 -O3 -march=native -flto -DNDEBUG main.cpp neural_network.cpp -o main
```

## Run

```bash
./main
```

The program continuously trains and periodically displays the generated grid, training epoch, and average loss in the terminal. Press Enter to stop the program.

## Project Structure

- `main.cpp` — cellular automata simulation, training loop, and terminal visualization
- `neural_network.cpp` / `neural_network.hpp` — neural network implementation
- `matrix.hpp` — matrix operations
- `math_vector.hpp` — vector operations
