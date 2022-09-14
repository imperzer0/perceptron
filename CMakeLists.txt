cmake_minimum_required(VERSION 3.23)
project(perceptron)

set(CMAKE_CXX_STANDARD 20)

add_executable(perceptron main.cpp layer.cpp layer.hpp matrix.cpp matrix.hpp perceptron.hpp perceptron.cpp)
