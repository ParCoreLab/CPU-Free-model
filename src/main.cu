#include <iostream>

//#include "../include/single-gpu-naive.cuh"
#include "../include/multi-gpu-peer.cuh"

int main(int argc, char* argv[]) {
    std::cout << "Running Single GPU Naive version" << std::endl;
    return init(argc, argv);
}
