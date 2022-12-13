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
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-layer-put.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-layer-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-layer-get-overlap.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-thread-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-thread-put.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-tile-get.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-one-block-comm-tile-put.cuh"
#include "../include/single-stream_nvshmem/multi-threaded-two-block-comm.cuh"
#include "../include/multi-stream/multi-gpu-peer-tiling.cuh"
#include "../include/multi-stream_nvshmem/multi-gpu-peer-tiling.cuh"

#include "../include/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-overlap-no-compute.cuh"
#include "../include/no-compute/multi-threaded-one-block-comm-no-compute.cuh"
#include "../include/no-compute/multi-threaded-one-block-comm-layer-no-compute.cuh"
#include "../include/no-compute/multi-threaded-p2p-no-compute.cuh"
#include "../include/no-compute/multi-threaded-two-block-comm-no-compute.cuh"

#include "../include/no-compute_nvshmem/multi-threaded-nvshmem-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-nvshmem-opt-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-contiguous-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-layer-put-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-layer-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-layer-get-overlap-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-two-block-comm-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-gpu-peer-tiling-no-compute.cuh"

#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-thread-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-thread-put-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-tile-get-no-compute.cuh"
#include "../include/no-compute_nvshmem/multi-threaded-one-block-comm-tile-put-no-compute.cuh"

using std::make_pair;

int main(int argc, char *argv[])
{
    const std::array versions{
        make_pair("Baseline Multi Threaded Copy", BaselineMultiThreadedCopy::init),
        make_pair("Baseline Multi Threaded Copy Overlap", BaselineMultiThreadedCopyOverlap::init),
        make_pair("Baseline Multi Threaded P2P", BaselineMultiThreadedP2P::init),
        make_pair("Baseline Single Threaded Copy", BaselineSingleThreadedCopy::init),
        make_pair("Naive Single stream multi threaded Tile-by-Tile (one thread block communicates)", SSMultiThreadedOneBlockComm::init),
        make_pair("Naive Single stream multi threaded Plane-by-Plane (one thread block communicates)", SSMultiThreadedOneBlockCommLayer::init),
        make_pair("Naive Single stream multi threaded (two thread blocks communicate)", SSMultiThreadedTwoBlockComm::init),
        make_pair("Naive Double stream multi threaded with Tiling", MultiGPUPeerTiling::init),
        make_pair("Baseline Multi Threaded Copy (No computation)", BaselineMultiThreadedCopyNoCompute::init),
        make_pair("Baseline Multi Threaded Copy Overlap (No Computation)", BaselineMultiThreadedCopyOverlapNoCompute::init),
        make_pair("Baseline Multi Threaded P2P (No Computation)", BaselineMultiThreadedP2PNoCompute::init),
        make_pair("Single stream multi threaded (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommNoCompute::init),
        make_pair("Single stream multi threaded layer (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommLayerNoCompute::init),
        make_pair("Single stream multi threaded (two thread blocks communicate; no computation)", SSMultiThreadedTwoBlockCommNoCompute::init),
        make_pair("Double stream multi threaded with Tiling (no computation)", MultiGPUPeerTilingNoCompute::init),
        make_pair("NVSHMEM Baseline Multi Threaded", BaselineMultiThreadedNvshmem::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized", BaselineMultiThreadedNvshmemOpt::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Put (one thread block communicates)", SSMultiThreadedOneBlockCommLayerPutNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Get (one thread block communicates)", SSMultiThreadedOneBlockCommLayerGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Get Overlap (one thread block communicates)", SSMultiThreadedOneBlockCommLayerGetOverlapNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded thread get (one thread block communicates)", SSMultiThreadedOneBlockCommThreadGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded thread put (one thread block communicates)", SSMultiThreadedOneBlockCommThreadPutNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded tile get (one thread block communicates)", SSMultiThreadedOneBlockCommTileGetNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded tile put (one thread block communicates)", SSMultiThreadedOneBlockCommTilePutNvshmem::init),
        make_pair("NVSHMEM Single stream multi threaded (two thread blocks communicate)", SSMultiThreadedTwoBlockCommNvshmem::init),
        make_pair("NVSHMEM Double stream multi threaded with Tiling", MultiGPUPeerTilingNvshmem::init),
        make_pair("NVSHMEM Baseline Multi Threaded (No Computation)", BaselineMultiThreadedNvshmemNoCompute::init),
        make_pair("NVSHMEM Baseline Multi Threaded Optimized (No Computation)", BaselineMultiThreadedNvshmemOptNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Put (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommLayerPutNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Get (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommLayerGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded Layer Get Overlap (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommLayerGetOverlapNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded thread get (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommThreadGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded thread put (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommThreadPutNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded tile get (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommTileGetNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded tile put (one thread block communicates; no computation)", SSMultiThreadedOneBlockCommTilePutNvshmemNoCompute::init),
        make_pair("NVSHMEM Single stream multi threaded (two thread blocks communicate; no computation)", SSMultiThreadedTwoBlockCommNvshmem::init),
        make_pair("NVSHMEM Double stream multi threaded with Tiling (No Computation)", MultiGPUPeerTilingNvshmemNoCompute::init),

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
