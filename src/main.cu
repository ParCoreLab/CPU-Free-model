#include <iostream>

#include "../include/baseline/multi-threaded-copy-overlap.cuh"
#include "../include/baseline/multi-threaded-copy.cuh"
#include "../include/baseline/multi-threaded-p2p.cuh"
#include "../include/baseline/single-threaded-copy.cuh"

#include "../include/common.h"
#include "../include/multi-stream/multi-gpu-peer-tiling-half.cuh"
#include "../include/multi-stream/multi-gpu-peer-tiling.cuh"
#include "../include/multi-stream/multi-gpu-peer.cuh"
#include "../include/single-stream/multi-threaded-one-block-comm.cuh"
#include "../include/single-stream/multi-threaded-two-block-comm.cuh"
#include "../include/single-stream/single-threaded-two-block-comm.cuh"
#include "../include/single-stream/single-threaded.cuh"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Single stream multi threaded (one thread block communicates)",
                  SSMultiThreadedOneBlockComm::init),
        make_pair("Single stream multi threaded (two thread blocks communicate)",
                  SSMultiThreadedTwoBlockComm::init),
        make_pair("Single stream single threaded", SSSingleThreaded::init),
        make_pair("Single stream single threaded (two thread blocks communicate)",
                  SSSingleThreadedTwoBlockComm::init),
        make_pair("Double stream multi threaded", MultiGPUPeer::init),
        make_pair("Baseline Multi Threaded Copy", BaselineMultiThreadedCopy::init),
        make_pair("Baseline Multi Threaded Copy Overlap", BaselineMultiThreadedCopyOverlap::init),
        make_pair("Baseline Multi Threaded P2P", BaselineMultiThreadedP2P::init),
        make_pair("Baseline Single Threaded Copy", BaselineSingleThreadedCopy::init),
        make_pair("Double stream multi threaded with Tiling", MultiGPUPeerTiling::init),
        make_pair("Double stream multi threaded with Tiling but one kernel is not cooperative",
                  MultiGPUPeerTilingHalf::init)};

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
