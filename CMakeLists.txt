cmake_minimum_required(VERSION 3.16)

project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(tensor_test
    src/tests/TensorTests.cpp
    src/Tensor/Tensor.cpp
)

target_include_directories(tensor_test PRIVATE
    src
)

add_executable(perceptron
    src/models/perceptron/perceptron.cpp
    src/Tensor/Tensor.cpp
)

target_include_directories(perceptron PRIVATE
    src
    src/Tensor
)
add_executable(layer_test
    src/layer_test.cpp
    src/models/Model.cpp
    src/Layer/Layer.cpp
    src/Optimizers/SDG.cpp
    src/Tensor/Tensor.cpp
)

target_include_directories(layer_test PRIVATE
    src
    src/Tensor
    src/models
    src/Layer
    src/Optimizers
    src/Tensor
)

