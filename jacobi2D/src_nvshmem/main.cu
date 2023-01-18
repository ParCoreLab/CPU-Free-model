
#include "../include_nvshmem/baseline/multi-threaded-nvshmem-opt.cuh"
#include "../include_nvshmem/baseline/multi-threaded-nvshmem.cuh"
#include "../include_nvshmem/multi-stream/multi-gpu-multi-block-tiling.cuh"
#include "../include_nvshmem/multi-stream/multi-gpu-peer-tiling.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-multi-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-one-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-two-block-comm.cuh"

#include "../include_nvshmem/no-comm/multi-gpu-peer-tiling-no-comm.cuh"
#include "../include_nvshmem/no-comm/multi-threaded-nvshmem-no-comm.cuh"
#include "../include_nvshmem/no-comm/multi-threaded-nvshmem-opt-no-comm.cuh"
#include "../include_nvshmem/no-comm/multi-threaded-one-block-comm-no-comm.cuh"
#include "../include_nvshmem/no-comm/multi-threaded-two-block-comm-no-comm.cuh"

#include "../include_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-opt-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-one-block-comm-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-two-block-comm-no-compute.cuh"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("NVSHMEM Baseline", BaselineMultiThreadedNvshmem::init),
        make_pair("NVSHMEM Baseline Optimized",
                  BaselineMultiThreadedNvshmemOpt::init),
        make_pair("NVSHMEM Single stream multi threaded (one thread block communicates)",
                  SSMultiThreadedOneBlockCommNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded (two thread blocks communicate)",
                  SSMultiThreadedTwoBlockCommNvshmem::init),
        make_pair("NVSHMEM Double stream multi threaded", MultiGPUPeerTilingNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded Partitioned",
                  SSMultiThreadedMultiBlockCommNvshmem::init),
        make_pair("NVSHMEM Double stream multi threaded Partitioned",
                  MultiGPUMultiBlockPeerTilingNvshmem::init),

        make_pair("NVSHMEM Baseline Multi Threaded (No Computation)",
                  BaselineMultiThreadedNvshmemNoCompute::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized (No Computation)",
                  BaselineMultiThreadedNvshmemOptNoCompute::init),
        make_pair(
            "NVSHMEM Single stream multi threaded (one thread block communicates; No Computation)",
            SSMultiThreadedOneBlockCommNvshmemNoCompute::init),
        make_pair(
            "NVSHMEM Single stream multi threaded (two thread blocks communicate; No Computation)",
            SSMultiThreadedTwoBlockCommNvshmemNoCompute::init),
        make_pair("NVSHMEM Double stream multi threaded (No Computation)",
                  MultiGPUPeerTilingNvshmemNoCompute::init),

        make_pair("NVSHMEM Baseline Multi Threaded (No Communication)",
                  BaselineMultiThreadedNvshmemNoComm::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized (No Communication)",
                  BaselineMultiThreadedNvshmemOptNoComm::init),
        make_pair(
            "NVSHMEM Single stream multi threaded (one thread block communicates; No Communication)",
            SSMultiThreadedOneBlockCommNvshmemNoComm::init),
        make_pair(
            "NVSHMEM Single stream multi threaded (two thread blocks communicate; No Communication)",
            SSMultiThreadedTwoBlockCommNvshmemNoComm::init),
        make_pair("NVSHMEM Double stream multi threaded (No Communication)",
                  MultiGPUPeerTilingNvshmemNoComm::init),
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
