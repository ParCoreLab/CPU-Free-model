#include "../include/baseline/multi-threaded-copy-overlap.cuh"
#include "../include/baseline/multi-threaded-copy.cuh"
#include "../include/baseline/multi-threaded-p2p.cuh"
#include "../include/baseline/single-threaded-copy.cuh"

#include "../include/single-stream/multi-threaded-one-block-comm-layer.cuh"
#include "../include/single-stream/multi-threaded-two-block-comm.cuh"

#include "../include/multi-stream/multi-gpu-peer-tiling.cuh"

#include "../include/no-compute/multi-gpu-peer-tiling-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-no-compute.cuh"
#include "../include/no-compute/multi-threaded-copy-overlap-no-compute.cuh"

#include "../include/no-compute/multi-threaded-one-block-comm-layer-no-compute.cuh"
#include "../include/no-compute/multi-threaded-p2p-no-compute.cuh"
#include "../include/no-compute/multi-threaded-two-block-comm-no-compute.cuh"


using std::make_pair;

int main(int argc, char *argv[])
{
    const std::array versions{
        make_pair("Baseline Multi Threaded Copy", BaselineMultiThreadedCopy::init),
        make_pair("Baseline Multi Threaded Copy Overlap", BaselineMultiThreadedCopyOverlap::init),
        make_pair("Baseline Multi Threaded P2P", BaselineMultiThreadedP2P::init),
        make_pair("Baseline Single Threaded Copy", BaselineSingleThreadedCopy::init),
        
        make_pair("Single stream multi threaded Plane-by-Plane (one thread block communicates)",
                  SSMultiThreadedOneBlockCommLayer::init),
        make_pair("Single stream multi threaded (two thread blocks communicate)",
                  SSMultiThreadedTwoBlockComm::init),

        make_pair("Double stream multi threaded with Tiling", MultiGPUPeerTiling::init),

        make_pair("Baseline Multi Threaded Copy (No computation)",
                  BaselineMultiThreadedCopyNoCompute::init),
        make_pair("Baseline Multi Threaded Copy Overlap (No Computation)",
                  BaselineMultiThreadedCopyOverlapNoCompute::init),
        make_pair("Baseline Multi Threaded P2P (No Computation)",
                  BaselineMultiThreadedP2PNoCompute::init),

        make_pair("Single stream multi threaded Plane-by-Plane (one thread block communicates; no computation)",
                  SSMultiThreadedOneBlockCommLayerNoCompute::init),
        make_pair("Single stream multi threaded (two thread blocks communicate; no computation)",
                  SSMultiThreadedTwoBlockCommNoCompute::init),
        make_pair("Double stream multi threaded with Tiling (no computation)",
                  MultiGPUPeerTilingNoCompute::init),
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
