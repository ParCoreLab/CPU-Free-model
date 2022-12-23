#include <iostream>

#include "../include_nvshmem/baseline/discrete-pipelined-nvshmem.cuh"
#include "../include_nvshmem/baseline/discrete-standard-nvshmem.cuh"
#include "../include_nvshmem/single-stream/pipelined-nvshmem.cuh"

#include "../include/common.h"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Discrete Standard NVSHMEM", BaselineDiscreteStandardNVSHMEM::init),
        make_pair("Baseline Discrete Pipelined NVSHMEM", BaselineDiscretePipelinedNVSHMEM::init),
        make_pair("Single Stream Pipelined NVSHMEM", SingleStreamPipelinedNVSHMEM::init),
    };

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);
    const bool silent = get_arg(argv, argv + argc, "-s");

    auto &selected = versions[selection];

    if (!silent) {
        std::cout << "Versions (select with -v):"
                  << "\n";
        for (int i = 0; i < versions.size(); ++i) {
            auto &v = versions[i];
            std::cout << i << ":\t" << v.first << "\n";
        }
        std::cout << std::endl;

        std::cout << "Running " << selected.first << "\n" << std::endl;
    }

    return selected.second(argc, argv);
}
