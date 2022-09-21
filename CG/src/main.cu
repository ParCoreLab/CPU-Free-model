#include <iostream>

#include "../include/baseline/non-persistent-non-pipelined.cuh"
#include "../include/baseline/non-persistent-pipelined.cuh"
#include "../include/baseline/persistent-non-pipelined.cuh"
#include "../include/single-stream/pipelined.cuh"

#include "../include/common.h"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Non-Persistent Non-Pipelined", BaselineNonPersistentNonPipelined::init),
        make_pair("Baseline Non-Persistent Pipelined", BaselineNonPersistentPipelined::init),
        make_pair("Baseline Persistent Non-Pipelined (with Prefetching)",
                  BaselinePersistentNonPipelined::init),
        make_pair("Single Stream Pipelined ", SingleStreamPipelined::init),
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
