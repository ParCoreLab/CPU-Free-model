#include <iostream>

#include "../include/baseline/non-persistent-unified-memory-pipelined.cuh"
#include "../include/baseline/persistent-unified-memory-gather-vector.cuh"
#include "../include/baseline/persistent-unified-memory-stale-device-vector.cuh"
#include "../include/baseline/persistent-unified-memory.cuh"

#include "../include/common.h"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Persistent Kernel with Unified Memory",
                  BaselinePersistentUnifiedMemory::init),
        make_pair(
            "Baseline Persistent Kernel with Unified Memory (Input vector gathered before SpMV)",
            BaselinePersistentUnifiedMemoryGatherVector::init),
        make_pair(
            "Baseline Persistent Kernel with Unified Memory (Input vector is on device but stale)",
            BaselinePersistentUnifiedMemoryStaleDeviceVector::init),
        make_pair("Baseline Pipelined Non Persistent Kernel with Unified Memory ",
                  BaselineNonPersistentUnifiedMemoryPipelined::init),
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
