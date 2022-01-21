#include <iostream>

#include "../include/common.h"
#include "../include/multi-gpu-peer.cuh"
#include "../include/single-gpu-naive.cuh"
#include "../include/single-stream/multi-threaded.cuh"
#include "../include/single-stream/single-threaded.cuh"

int main(int argc, char* argv[]) {
    const std::array<std::pair<std::string, initfunc_t>, 4> versions{
        std::make_pair("Single stream multi threaded (default)", SSMultiThreaded::init),
        std::make_pair("Single stream single threaded", SSSingleThreaded::init),
        std::make_pair("Double stream multi threaded", MultiGPUPeer::init),
        std::make_pair("Single GPU Persistent Naive", SingleGPUNaive::init)};

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);

    std::cout << "Versions (select with -v):"
              << "\n";
    for (int i = 0; i < versions.size(); ++i) {
        auto& v = versions[i];
        std::cout << i << ": " << v.first << "\n";
    }
    std::cout << std::endl;

    auto& selected = versions[selection];

    std::cout << "Running " << selected.first << "\n" << std::endl;
    return selected.second(argc, argv);
}
