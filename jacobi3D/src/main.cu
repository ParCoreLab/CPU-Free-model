#include <iostream>
#include "../include/baseline/multi-threaded-copy.cuh"
#include "../include/baseline/multi-threaded-copy-overlap.cuh"
#include "../include/baseline/multi-threaded-p2p.cuh"
#include "../include/baseline/single-threaded-copy.cuh"

#include "../include/baseline_nvshmem/multi-threaded-nvshmem.cuh"
#include "../include/baseline_nvshmem/multi-threaded-nvshmem-opt.cuh"

#include "../include/single-stream/multi-threaded-one-block-comm.cuh"
#include "../include/single-stream/multi-threaded-one-block-comm-layer.cuh"
#include "../include/single-stream/multi-threaded-two-block-comm.cuh"


#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-contiguous.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-bulk.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-bulk-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-thread-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-thread-put.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-original-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-original-put.cuh"


#include "../include/multi-stream/multi-gpu-peer-tiling.cuh"
//#include "../include/multi-stream_nvshmem/multi-gpu-peer-tiling.cuh"


#include "../include/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-overlap-no-compute.cuh"
#include "../include/no-compute/multi-threaded-one-block-comm-no-compute.cuh"
#include "../include/no-compute/multi-threaded-one-block-warp-comm-no-compute.cuh"
#include "../include/no-compute/multi-threaded-p2p-no-compute.cuh"
#include "../include/no-compute/multi-threaded-two-block-comm-no-compute.cuh"

#include "../include/no-compute_nvshmem/multi-threaded-nvshmem-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-nvshmem-opt-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-contiguous-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-bulk-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-bulk-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-thread-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-thread-put-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-original-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-original-put-no-compute.cuh"

using std::make_pair;

int main(int argc, char *argv[]) {
    const std::array versions{
        make_pair("Baseline Multi Threaded Copy", BaselineMultiThreadedCopy::init),
        make_pair("Baseline Multi Threaded Copy Overlap", BaselineMultiThreadedCopyOverlap::init),
        make_pair("Baseline Multi Threaded P2P", BaselineMultiThreadedP2P::init),
        make_pair("Baseline Single Threaded Copy", BaselineSingleThreadedCopy::init),

        make_pair("Baseline Multi Threaded NVSHMEM", BaselineMultiThreadedNvshmem::init),
        make_pair("Baseline Single Threaded NVSHMEM Optimized", BaselineMultiThreadedNvshmemOpt::init),

        make_pair("Naive Single stream multi threaded Tile-by-Tile (one thread block communicates)",
                  SSMultiThreadedOneBlockComm::init),
        make_pair("Naive Single stream multi threaded Plane-by-Plane (one thread block communicates)",
                  SSMultiThreadedOneBlockCommLayer::init),
        make_pair("Naive Single stream multi threaded (two thread blocks communicate)",
                  SSMultiThreadedTwoBlockComm::init),
        make_pair("Naive Double stream multi threaded with Tiling", MultiGPUPeerTiling::init),

        make_pair("NVSHMEM Single stream multi threaded bulk (one thread block communicates)",
                  SSMultiThreadedOneBlockCommBulkNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded bulk get (one thread block communicates)",
                  SSMultiThreadedOneBlockCommBulkGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded contiguous (one thread block communicates)",
                  SSMultiThreadedOneBlockCommContiguousNvshmem::init),
                  
        make_pair("NVSHMEM Single stream multi threaded thread get (one thread block communicates)",
                  SSMultiThreadedOneBlockCommThreadGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded thread put (one thread block communicates)",
                  SSMultiThreadedOneBlockCommThreadPutNvshmem::init),

        make_pair("NVSHMEM Single stream multi threaded original (one thread block communicates)",
                  SSMultiThreadedOneBlockCommOriginalGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded original put (one thread block communicates)",
                  SSMultiThreadedOneBlockCommOriginalPutNvshmem::init),

        make_pair("Baseline Multi Threaded Copy (No computation)",
                  BaselineMultiThreadedCopyNoCompute::init),
        make_pair("Baseline Multi Threaded Copy Overlap (No Computation)",
                  BaselineMultiThreadedCopyOverlapNoCompute::init),
        make_pair("Baseline Multi Threaded P2P (No Computation)",
                  BaselineMultiThreadedP2PNoCompute::init),

        make_pair("Baseline Multi Threaded NVSHMEM (No Computation)", BaselineMultiThreadedNvshmemNoCompute::init),
        make_pair("Baseline Single Threaded NVSHMEM Optimized (No Computation)", BaselineMultiThreadedNvshmemOptNoCompute::init),

        make_pair("Single stream multi threaded (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommNoCompute::init),
        make_pair("Single stream multi threaded warp (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockWarpCommNoCompute::init),
        make_pair("Single stream multi threaded (two thread blocks communicate; no computation)",
                  SSMultiThreadedTwoBlockCommNoCompute::init),
        make_pair("Double stream multi threaded with Tiling (no computation)",
                  MultiGPUPeerTilingNoCompute::init),

        make_pair("NVSHMEM Single stream multi threaded bulk (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommBulkNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded bulk get (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommBulkGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded contiguous (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommContiguousNvshmemNoCompute::init),
                  
        make_pair("NVSHMEM Single stream multi threaded thread get (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommThreadGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded thread put (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommThreadPutNvshmemNoCompute::init),

        make_pair("NVSHMEM Single stream multi threaded original (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommOriginalGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded original put (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommOriginalPutNvshmemNoCompute::init)

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
