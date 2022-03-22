#include <iostream>

#include "../include/baseline/multi-threaded-copy-overlap.cuh"
#include "../include/baseline/multi-threaded-copy.cuh"
#include "../include/baseline/multi-threaded-p2p.cuh"
#include "../include/baseline/single-threaded-copy.cuh"

#include "../include/common.h"
#include "../include/multi-gpu-peer.cuh"
#include "../include/single-stream/multi-threaded-two-block-comm.cuh"
#include "../include/single-stream/multi-threaded.cuh"
#include "../include/single-stream/single-threaded-two-block-comm.cuh"
#include "../include/single-stream/single-threaded.cuh"

int main(int argc, char *argv[]) {
    const std::array<std::pair<std::string, initfunc_t>, 9> versions{
            std::make_pair("Single stream multi threaded (default)", SSMultiThreaded::init),
            std::make_pair("Single stream multi threaded (two thread blocks communicate)",
                           SSMultiThreadedTwoBlockComm::init),
            std::make_pair("Single stream single threaded", SSSingleThreaded::init),
            std::make_pair("Single stream single threaded (two thread blocks communicate)",
                           SSSingleThreadedTwoBlockComm::init),
            std::make_pair("Double stream multi threaded", MultiGPUPeer::init),
            std::make_pair("Baseline Multi Threaded Copy", BaselineMultiThreadedCopy::init),
            std::make_pair("Baseline Multi Threaded Copy Overlap",
                           BaselineMultiThreadedCopyOverlap::init),
            std::make_pair("Baseline Multi Threaded P2P", BaselineMultiThreadedP2P::init),
            std::make_pair("Baseline Single Threaded Copy", BaselineSingleThreadedCopy::init)};

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);
    const bool silent = get_arg(argv, argv + argc, "-s");

    auto &selected = versions[selection];

    if (!silent) {
        std::cout << "Versions (select with -v):"
                  << "\n";
        for (int i = 0; i < versions.size(); ++i) {
            auto &v = versions[i];
            std::cout << i << ": " << v.first << "\n";
        }
        std::cout << std::endl;

        std::cout << "Running " << selected.first << "\n" << std::endl;
    }

    return selected.second(argc, argv);
}
