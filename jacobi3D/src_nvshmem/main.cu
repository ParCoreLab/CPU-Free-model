#include <array>
#include <iostream>

#include "../include_nvshmem/PERKS-nvshmem/multi-stream-perks-nvshmem-block.h"
#include "../include_nvshmem/PERKS-nvshmem/multi-stream-perks-nvshmem.h"
#include "../include_nvshmem/baseline/multi-threaded-nvshmem-opt.cuh"
#include "../include_nvshmem/baseline/multi-threaded-nvshmem.cuh"
#include "../include_nvshmem/multi-stream/multi-gpu-multi-block-tiling.cuh"
#include "../include_nvshmem/multi-stream/multi-gpu-peer-tiling.cuh"
#include "../include_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-multi-block-comm-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-opt-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-one-block-comm-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-two-block-comm-no-compute.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-multi-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-one-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-two-block-comm.cuh"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline NVSHMEM", BaselineMultiThreadedNvshmemOpt::init),

        make_pair("Design 1 (NVSHMEM)", MultiGPUPeerTilingNvshmem::init),
        make_pair("Design 2 (NVSHMEM)", SSMultiThreadedTwoBlockCommNvshmem::init),

        make_pair("Design 1 Partitioned (NVSHMEM)", MultiGPUMultiBlockPeerTilingNvshmem::init),
        make_pair("Design 2 Partitioned (NVSHMEM)", SSMultiThreadedMultiBlockCommNvshmem::init),

        make_pair("Baseline NVSHMEM (No Computation)",
                  BaselineMultiThreadedNvshmemOptNoCompute::init),
        make_pair("Design 1 NVSHMEM (No Compute)", MultiGPUPeerTilingNvshmemNoCompute::init),

        make_pair("PERKS NVSHMEM", MultiStreamPERKSNvshmem::init),
        make_pair("PERKS NVSHMEM Partitioned", MultiStreamPERKSNvshmemBlock::init),

        //        make_pair("NVSHMEM Baseline Multi Threaded", BaselineMultiThreadedNvshmem::init),
        //        make_pair("NVSHMEM Single stream multi threaded (one thread block communicates)",
        //                  SSMultiThreadedOneBlockCommNvshmem::init),
        //        make_pair("NVSHMEM Baseline Multi Threaded (No Computation)",
        //        BaselineMultiThreadedNvshmemNoCompute::init), make_pair(
        //            "NVSHMEM Single stream multi threaded (one thread block communicates; no
        //            computation)", SSMultiThreadedOneBlockCommNvshmemNoCompute::init),
        //        make_pair(
        //                    "NVSHMEM Single stream multi threaded (two thread blocks communicate;
        //                    no computation)", SSMultiThreadedTwoBlockCommNvshmemNoCompute::init),
        //            make_pair("Design 2 NVSHMEM (No Compute)",
        //            SSMultiThreadedMultiBlockCommNvshmemNoCompute::init),
    };

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);
    const bool silent = get_arg(argv, argv + argc, "-s");

    auto &selected = versions[selection];

    if (!silent) {
        std::cout << "Versions (select with -v):"
                  << "\n";
        for (size_t i = 0; i < versions.size(); ++i) {
            auto &v = versions[i];
            std::cout << i << ":\t" << v.first << "\n";
        }
        std::cout << std::endl;

        std::cout << "Running " << selected.first << "\n" << std::endl;
    }

    return selected.second(argc, argv);
}
