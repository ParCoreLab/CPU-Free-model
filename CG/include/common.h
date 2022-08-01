#ifndef INC_2D_CG_COMMON_H
#define INC_2D_CG_COMMON_H

#include <algorithm>
#include <sstream>
#include <string>

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

bool get_arg(char **begin, char **end, const std::string &arg);

void genTridiag(int *I, int *J, float *val, int N, int nz);

float dotProduct(float *vecA, float *vecB, int size);

void scaleVector(float *vec, float alpha, int size);

void saxpy(float *x, float *y, float a, int size);

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p, float *r, int nnz,
                      int N, float tol);

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

#endif  // INC_CG_COMMON_H