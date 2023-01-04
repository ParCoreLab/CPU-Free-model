#include "../include_nvshmem/baseline/multi-threaded-nvshmem.cuh"
#include "../include_nvshmem/baseline/multi-threaded-nvshmem-opt.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-one-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-two-block-comm.cuh"
#include "../include_nvshmem/single-stream/multi-threaded-multi-block-comm.cuh"
#include "../include_nvshmem/multi-stream/multi-gpu-peer-tiling.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-nvshmem-opt-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-one-block-comm-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-two-block-comm-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include_nvshmem/no-compute/multi-threaded-multi-block-comm-no-compute.cuh"

using std::make_pair;

int main(int argc, char *argv[])
{
    const std::array versions{
        make_pair("NVSHMEM Baseline Multi Threaded", BaselineMultiThreadedNvshmem::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized", BaselineMultiThreadedNvshmemOpt::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Put (one thread block communicates)", SSMultiThreadedOneBlockCommNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded (two thread blocks communicate)", SSMultiThreadedTwoBlockCommNvshmem::init),
        make_pair("NVSHMEM Double stream multi threaded", MultiGPUPeerTilingNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded Partitioned,All SM Comm", SSMultiThreadedMultiBlockCommNvshmem::init),
        make_pair("NVSHMEM Baseline Multi Threaded (No Computation)", BaselineMultiThreadedNvshmemNoCompute::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized (No Computation)", BaselineMultiThreadedNvshmemOptNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Put (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded (two thread blocks communicate; no computation)", SSMultiThreadedTwoBlockCommNvshmemNoCompute::init),
        make_pair("NVSHMEM Double stream multi threaded with Tiling (No Computation)", MultiGPUPeerTilingNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded Partitioned,All SM Comm (No Computation)", SSMultiThreadedMultiBlockCommNvshmemNoCompute::init),
    };

    const int selection = get_argval<int>(argv, argv + argc, "-v", 0);
    const bool silent = get_arg(argv, argv + argc, "-s");

    auto &selected = versions[selection];

    if (!silent)
    {
        std::cout << "Versions (select with -v):"
                  << "\n";
        for (int i = 0; i < versions.size(); ++i)
        {
            auto &v = versions[i];
            std::cout << i << ":\t" << v.first << "\n";
        }
        std::cout << std::endl;

        std::cout << "Running " << selected.first << "\n"
                  << std::endl;
    }

    return selected.second(argc, argv);
}
