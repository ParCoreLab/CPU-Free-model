#ifndef INC_2D_CG_COMMON_H
#define INC_2D_CG_COMMON_H

#include <algorithm>
#include <sstream>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

typedef double real;

constexpr int THREADS_PER_BLOCK = 512;
constexpr real tol = 1e-5f;

namespace cg = cooperative_groups;

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m, int *n, int *nnz,
                       T_ELEM **aVal, int **aRowInd, int **aColInd, int extendSymMatrix);

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg, const T default_val) {
    T argval = default_val;
    char **itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

void report_results(const int num_rows, real *x_ref, real *x, const int num_devices,
                    const double single_gpu_runtime, const double start, const double stop,
                    const bool compare_to_single_gpu);

// Data filled on CPU needed for MultiGPU operations.
struct MultiDeviceData {
    unsigned char *hostMemoryArrivedList;
    unsigned int numDevices;
    unsigned int deviceRank;
};

// Class used for coordination of multiple devices.
class PeerGroup {
    const MultiDeviceData &data;
    const cg::grid_group &grid;

    __device__ unsigned char load_arrived(unsigned char *arrived) const {
#if __CUDA_ARCH__ < 700
        return *(volatile unsigned char *)arrived;
#else
        unsigned int result;
        asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(result) : "l"(arrived) : "memory");
        return result;
#endif
    }

    __device__ void store_arrived(unsigned char *arrived, unsigned char val) const {
#if __CUDA_ARCH__ < 700
        *(volatile unsigned char *)arrived = val;
#else
        unsigned int reg_val = val;
        asm volatile("st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived) : "memory");

        // Avoids compiler warnings from unused variable val.
        (void)(reg_val = reg_val);
#endif
    }

   public:
    __device__ PeerGroup(const MultiDeviceData &data, const cg::grid_group &grid)
        : data(data), grid(grid){};

    __device__ unsigned int size() const { return data.numDevices * grid.size(); }

    __device__ unsigned int thread_rank() const {
        return data.deviceRank * grid.size() + grid.thread_rank();
    }

    __device__ void sync() const {
        grid.sync();

        // One thread from each grid participates in the sync.
        if (grid.thread_rank() == 0) {
            if (data.deviceRank == 0) {
                // Leader grid waits for others to join and then releases them.
                // Other GPUs can arrive in any order, so the leader have to wait for
                // all others.
                for (int i = 0; i < data.numDevices - 1; i++) {
                    while (load_arrived(&data.hostMemoryArrivedList[i]) == 0)
                        ;
                }
                for (int i = 0; i < data.numDevices - 1; i++) {
                    store_arrived(&data.hostMemoryArrivedList[i], 0);
                }
                __threadfence_system();
            } else {
                // Other grids note their arrival and wait to be released.
                store_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1], 1);
                while (load_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1]) == 1)
                    ;
            }
        }

        grid.sync();
    }

    // Calculate size of a grid with `num_allocated_tbs` thread blocks
    __device__ unsigned int calc_subgrid_size(const int num_allocated_tbs) const {
        return num_allocated_tbs * THREADS_PER_BLOCK;
    }

    // Calculate rank of a thread in grid with `num_allocated_tbs thread blocks
    __device__ unsigned int calc_subgrid_thread_rank(const int num_allocated_tbs) const {
        int thread_rank = grid.thread_rank();
        int subgrid_size = calc_subgrid_size(num_allocated_tbs);

        return (subgrid_size > thread_rank) ? thread_rank : thread_rank - subgrid_size;
    }
};

// Single GPU kernels
namespace SingleGPU {
__global__ void initVectors(real *r, real *x, int num_rows);

__global__ void gpuCopyVector(real *srcA, real *destB, int num_rows);

__global__ void gpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY);

__global__ void gpuSaxpy(real *x, real *y, real a, int num_rows);

__global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int num_rows);

}  // namespace SingleGPU

// Multi GPU kernels
namespace MultiGPU {
__global__ void initVectors(real *r, real *x, int num_rows, const int device_rank,
                            const int num_devices);

__global__ void gpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha,
                        real *inputVecX, real *outputVecY, const int device_rank,
                        const int num_devices);

__global__ void gpuSaxpy(real *x, real *y, real a, int num_rows, const int device_rank,
                         const int num_devices);

__global__ void gpuCopyVector(real *srcA, real *destB, int num_rows, const int device_rank,
                              const int num_devices);

__global__ void gpuScaleVectorAndSaxpy(real *x, real *y, real a, real scale, int num_rows,
                                       const int device_rank, const int num_devices);

__device__ unsigned char load_arrived(unsigned char *arrived);

__device__ void store_arrived(unsigned char *arrived, unsigned char val);

__global__ void syncPeers(const int device_rank, const int num_devices,
                          unsigned char *hostMemoryArrivedList);
}  // namespace MultiGPU

__global__ void a_minus(real a, real *na);

__global__ void r1_div_x(real r1, real r0, real *b);

__global__ void update_a_k(real dot_delta_1, real dot_gamma_1, real b, real *a);

__global__ void update_b_k(real dot_delta_1, real dot_delta_0, real *b);

__global__ void init_a_k(real dot_delta_1, real dot_gamma_1, real *a);

__global__ void init_b_k(real *b);

// Multi-GPU Sync Kernel

namespace SingleGPUPipelinedDiscrete {
__global__ void gpuDotProductsMerged(real *vecA_delta, real *vecB_delta, real *vecA_gamma,
                                     real *vecB_gamma, int num_rows, const int sMemSize);

__global__ void addLocalDotContributions(double *dot_result_delta, double *dot_result_gamma);

__global__ void resetLocalDotProducts(double *dot_result_delta, double *dot_result_gamma);

double run_single_gpu(const int iter_max, int *um_I, int *um_J, real *um_val, real *host_val,
                      int num_rows, int nnz);
}  // namespace SingleGPUPipelinedDiscrete

namespace SingleGPUStandardDiscrete {

__global__ void gpuDotProduct(real *vecA, real *vecB, int num_rows);

__global__ void addLocalDotContribution(double *dot_result);

__global__ void resetLocalDotProduct(double *dot_result);

double run_single_gpu(const int iter_max, int *um_I, int *um_J, real *um_val, real *host_val,
                      int num_rows, int nnz);
}  // namespace SingleGPUStandardDiscrete

namespace CPU {
void cpuSpMV(int *I, int *J, real *val, int nnz, int num_rows, real alpha, real *inputVecX,
             real *outputVecY);

real dotProduct(real *vecA, real *vecB, int size);

void scaleVector(real *vec, real alpha, int size);

void saxpy(real *x, real *y, real a, int size);

void cpuConjugateGrad(const int iter_max, int *I, int *J, real *val, real *x, real *Ax, real *p,
                      real *r, int nnz, int num_rows, real tol);
}  // namespace CPU

bool get_arg(char **begin, char **end, const std::string &arg);

void genTridiag(int *I, int *J, real *val, int N, int nz);

#define noop

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }                                                                                       \
    noop

#define CURAND_CALL(x)                                      \
    do {                                                    \
        if ((x) != CURAND_STATUS_SUCCESS) {                 \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

#endif  // INC_CG_COMMON_H