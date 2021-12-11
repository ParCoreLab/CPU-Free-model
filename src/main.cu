#include <iostream>

#include "../include/single-gpu-naive.cuh"

int main(int argc, char* argv[]) {
    std::cout << "Running Single GPU Naive version" << std::endl;
    return init(argc, argv);
}
