#include <iostream>

#include "../include/baseline/discrete-pipelined.cuh"
#include "../include/baseline/discrete-standard.cuh"
#include "../include/baseline/persistent-standard.cuh"
#include "../include/profiling/discrete-pipelined.cuh"
#include "../include/profiling/discrete-standard.cuh"
#include "../include/single-stream/pipelined.cuh"

#include "../include/common.h"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Discrete Standard", BaselineDiscreteStandard::init),
        make_pair("Baseline Discrete Pipelined", BaselineDiscretePipelined::init),
        make_pair("Baseline Persistent Standard", BaselinePersistentStandard::init),
        make_pair("Single Stream Pipelined ", SingleStreamPipelined::init),
        make_pair("Profiling Discrete Standard", ProfilingDiscreteStandard::init),
        make_pair("Profiling Discrete Pipelined", ProfilingDiscretePipelined::init),
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
