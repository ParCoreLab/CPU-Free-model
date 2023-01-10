//// #include "./common/common.hpp"
//// #include <cooperative_groups.h>
//// #include <cuda.h>
//// #include "stdio.h"
//// #include "./common/cuda_computation.cuh"
//// #include "./common/cuda_common.cuh"
//// #include "./common/types.hpp"
//#ifdef GEN
//#include "./genconfig.cuh"
//#endif
//
//#ifdef _TIMER_
//#include "cuda_profiler_api.h"
//#endif
//#include "config.cuh"
//#include <cooperative_groups.h>
//#include <cuda.h>
//#include "./common/cuda_common.cuh"
//#include "./common/cuda_computation.cuh"
//#include "./common/jacobi_cuda.cuh"
//#include "./common/types.hpp"
//#include "assert.h"
//#include "stdio.h"
//
//// #define TILE_X 256
//// #define NAIVE
//#if defined(NAIVE) || defined(BASELINE) || defined(BASELINE_CM)
//#define TRADITIONLAUNCH
//#endif
//#if defined(GEN) || defined(PERSISTENT) || defined(GENWR)
//#define PERSISTENTLAUNCH
//#endif
//#if defined PERSISTENTLAUNCH || defined(BASELINE_CM)
//#define PERSISTENTTHREAD
//#endif
//#if defined(BASELINE) || defined(BASELINE_CM) || defined(GEN) || defined(GENWR) || \
//    defined(PERSISTENT)
//#define USEMAXSM
//#endif
//
//#define ITEM_PER_THREAD (8)
//
//#include "./perksconfig.cuh"
//
//#define cudaCheckError()                                                                     \
//    {                                                                                        \
//        cudaError_t e = cudaGetLastError();                                                  \
//        if (e != cudaSuccess) {                                                              \
//            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
//        }                                                                                    \
//    }
//
//__global__ void printptx(int* result) {
//    // printf("code is run in %d\n",PERKS_ARCH);
//    result[0] = PERKS_ARCH;
//}
// void host_printptx(int& result) {
//    int* d_r;
//    cudaMalloc((void**)&d_r, sizeof(int));
//    printptx<<<1, 1>>>(d_r);
//    cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//}
//
// template <class REAL>
// int getMinWidthY(int width_x, int width_y, int global_bdimx, bool isDoubleTile) {
//    int minwidthy1 = j3d_iterative<REAL>(nullptr, 100000, width_y, width_x, nullptr, global_bdimx,
//                                         1, 1, false, false, 0, isDoubleTile, true);
//
//    int minwidthy2 = j3d_iterative<REAL>(nullptr, 100000, width_y, width_x, nullptr, global_bdimx,
//                                         2, 1, false, false, 0, isDoubleTile, true);
//    int minwidthy3 = j3d_iterative<REAL>(nullptr, 100000, width_y, width_x, nullptr, global_bdimx,
//                                         1, 1, true, false, 0, isDoubleTile, true);
//    int minwidthy4 = j3d_iterative<REAL>(nullptr, 100000, width_y, width_x, nullptr, global_bdimx,
//                                         2, 1, true, false, 0, isDoubleTile, true);
//
//    int result = max(minwidthy1, minwidthy2);
//    result = max(result, minwidthy3);
//    result = max(result, minwidthy4);
//    return result;
//}
//
// template int getMinWidthY<float>(int, int, int, bool);
// template int getMinWidthY<double>(int, int, int, bool);
//
// template <class REAL>
// int j3d_iterative(REAL* h_input, int height, int width_y, int width_x, REAL* __var_0__,
//                  int global_bdimx, int blkpsm, int iteration, bool useSM, bool usewarmup,
//                  int warmupiteration, bool isDoubleTile, bool getminHeight) {
//    const int LOCAL_ITEM_PER_THREAD = isDoubleTile ? ITEM_PER_THREAD * 2 : ITEM_PER_THREAD;
//    global_bdimx = global_bdimx == 128 ? 128 : 256;
//    if (isDoubleTile) {
//        if (global_bdimx == 256) blkpsm = 1;
//        if (global_bdimx == 128) blkpsm = min(blkpsm, 2);
//    }
//    int TILE_Y = LOCAL_ITEM_PER_THREAD * global_bdimx / TILE_X;
//    if (blkpsm <= 0) blkpsm = 100;
//    // int iteration=4;
//    /* Host allocation Begin */
//    int sm_count;
//    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
//#ifndef __PRINT__
//    printf("sm_count is %d\n", sm_count);
//#endif
//    // printf("sm_count is %d\n",sm_count);
//    int ptx;
////    host_printptx(ptx);
//#ifndef __PRINT__
//    // printf("code is run in %d\n",ptx);
//#endif
//#ifdef NAIVE
//    auto execute_kernel = kernel3d_restrict<REAL, HALO>;
//#endif
//#if defined(BASELINE) || defined(BASELINE_CM)
//    auto execute_kernel =
//        isDoubleTile
//            ? (global_bdimx == 128
//                   ? kernel3d_baseline<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, 128>
//                   : kernel3d_baseline<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, 256>)
//            : (global_bdimx == 128 ? kernel3d_baseline<REAL, HALO, ITEM_PER_THREAD, TILE_X, 128>
//                                   : kernel3d_baseline<REAL, HALO, ITEM_PER_THREAD, TILE_X, 256>);
//#endif
//#ifdef PERSISTENT
//    auto execute_kernel =
//        isDoubleTile
//            ? (global_bdimx == 128
//                   ? kernel3d_persistent<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, 128>
//                   : kernel3d_persistent<REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, 256>)
//            : (global_bdimx == 128 ? kernel3d_persistent<REAL, HALO, ITEM_PER_THREAD, TILE_X, 128>
//                                   : kernel3d_persistent<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                   256>);
//#endif
//#ifdef GEN
//    auto execute_kernel =
//        useSM
//            ? (global_bdimx == 128
//                   ? (blkpsm >= 4
//                          ? (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, 2 * ITEM_PER_THREAD>::val, true, 128>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, ITEM_PER_THREAD>::val, true, 128>)
//                          : (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, 2 * ITEM_PER_THREAD>::val, true, 128>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, ITEM_PER_THREAD>::val, true, 128>))
//                   : (blkpsm >= 2
//                          ? (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, 2 * ITEM_PER_THREAD>::val, true, 256>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, ITEM_PER_THREAD>::val, true, 256>)
//                          : (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, 2 * ITEM_PER_THREAD>::val, true, 256>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, ITEM_PER_THREAD>::val, true, 256>)))
//            : (global_bdimx == 128
//                   ? (blkpsm >= 4
//                          ? (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, 2 * ITEM_PER_THREAD>::val, false,
//                                       128>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, ITEM_PER_THREAD>::val, false, 128>)
//                          : (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, 2 * ITEM_PER_THREAD>::val, false,
//                                       128>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, ITEM_PER_THREAD>::val, false, 128>))
//                   : (blkpsm >= 2
//                          ? (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, 2 * ITEM_PER_THREAD>::val, false,
//                                       256>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 2, ITEM_PER_THREAD>::val, false, 256>)
//                          : (isDoubleTile
//                                 ? kernel3d_general<
//                                       REAL, HALO, 2 * ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, 2 * ITEM_PER_THREAD>::val, false,
//                                       256>
//                                 : kernel3d_general<
//                                       REAL, HALO, ITEM_PER_THREAD, TILE_X, REG_FOLDER_Z,
//                                       getminblocks<REAL, 1, ITEM_PER_THREAD>::val, false,
//                                       256>)));
//    int lreg_folder_z = 0;
//    // if(isDoubleTile)
//    bool ifspill = false;
//    {
//        if (global_bdimx == 128) {
//            if (blkpsm >= 4) {
//                if (ptx == 800) {
//                    lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                                      800, true, REAL>::val
//                                          : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                                      800, false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
//                                                false, REAL>::spill;
//                }
//                if (ptx == 700) {
//                    lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                                      700, true, REAL>::val
//                                          : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                                      700, false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
//                                                false, REAL>::spill;
//                }
//            } else {
//                if (isDoubleTile) {
//                    if (ptx == 800) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 *
//                        ITEM_PER_THREAD,
//                                                          256, 800, true, REAL>::val
//                                              : regfolder<HALO, curshape, 128, 2 *
//                                              ITEM_PER_THREAD,
//                                                          256, 800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    800, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    800, false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 *
//                        ITEM_PER_THREAD,
//                                                          256, 700, true, REAL>::val
//                                              : regfolder<HALO, curshape, 128, 2 *
//                                              ITEM_PER_THREAD,
//                                                          256, 700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    700, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    700, false, REAL>::spill;
//                    }
//                } else {
//                    if (ptx == 800) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                        256,
//                                                          800, true, REAL>::val
//                                              : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                                              256,
//                                                          800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                        800,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                                        800,
//                                                    false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                        256,
//                                                          700, true, REAL>::val
//                                              : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                                              256,
//                                                          700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                        700,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                                        700,
//                                                    false, REAL>::spill;
//                    }
//                }
//            }
//        } else {
//            if (blkpsm >= 2) {
//                if (ptx == 800) {
//                    lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                                      800, true, REAL>::val
//                                          : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                                      800, false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
//                                                false, REAL>::spill;
//                }
//                if (ptx == 700) {
//                    lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                                      700, true, REAL>::val
//                                          : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                                      700, false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
//                                                false, REAL>::spill;
//                }
//            } else {
//                if (isDoubleTile) {
//                    if (ptx == 800) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 *
//                        ITEM_PER_THREAD,
//                                                          256, 800, true, REAL>::val
//                                              : regfolder<HALO, curshape, 256, 2 *
//                                              ITEM_PER_THREAD,
//                                                          256, 800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    800, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    800, false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 *
//                        ITEM_PER_THREAD,
//                                                          256, 700, true, REAL>::val
//                                              : regfolder<HALO, curshape, 256, 2 *
//                                              ITEM_PER_THREAD,
//                                                          256, 700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    700, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    700, false, REAL>::spill;
//                    }
//                } else {
//                    if (ptx == 800) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                        256,
//                                                          800, true, REAL>::val
//                                              : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                                              256,
//                                                          800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                        800,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                                        800,
//                                                    false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        lreg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                        256,
//                                                          700, true, REAL>::val
//                                              : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                                              256,
//                                                          700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                        700,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                                        700,
//                                                    false, REAL>::spill;
//                    }
//                }
//            }
//        }
//    }
//    if (lreg_folder_z < REG_FOLDER_Z || ifspill) return -5;
//#endif
//
//#ifdef GENWR
//
//    auto execute_kernel =
//        isDoubleTile
//            ?
//
//            (global_bdimx == 128
//                 ? (blkpsm >= 4
//                        ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                        TILE_X,
//                                                            256, true, 128, curshape>
//                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                                 TILE_X,
//                                                            256, false, 128, curshape>)
//                        : (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                        TILE_X,
//                                                            256, true, 128, curshape>
//                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                                 TILE_X,
//                                                            256, false, 128, curshape>))
//                 : (blkpsm >= 2
//                        ? (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                        TILE_X,
//                                                            256, true, 256, curshape>
//                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                                 TILE_X,
//                                                            256, false, 256, curshape>)
//                        : (useSM ? kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                        TILE_X,
//                                                            256, true, 256, curshape>
//                                 : kernel3d_general_wrapper<REAL, HALO, 2 * ITEM_PER_THREAD,
//                                 TILE_X,
//                                                            256, false, 256, curshape>)))
//            : (global_bdimx == 128
//                   ? (blkpsm >= 4
//                          ? (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              128, true, 128, curshape>
//                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              128, false, 128, curshape>)
//                          : (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              256, true, 128, curshape>
//                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              256, false, 128, curshape>))
//                   : (blkpsm >= 2
//                          ? (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              128, true, 256, curshape>
//                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              128, false, 256, curshape>)
//                          : (useSM ? kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              256, true, 256, curshape>
//                                   : kernel3d_general_wrapper<REAL, HALO, ITEM_PER_THREAD, TILE_X,
//                                                              256, false, 256, curshape>)));
//    int reg_folder_z = 0;
//    // if(isDoubleTile)
//    bool ifspill = false;
//    {
//        if (global_bdimx == 128) {
//            if (blkpsm >= 4) {
//                if (ptx == 800) {
//                    reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                    800,
//                                                     true, REAL>::val
//                                         : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                         800,
//                                                     false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 800,
//                                                false, REAL>::spill;
//                }
//                if (ptx == 700) {
//                    reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                    700,
//                                                     true, REAL>::val
//                                         : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128,
//                                         700,
//                                                     false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 128, 700,
//                                                false, REAL>::spill;
//                }
//            } else {
//                if (isDoubleTile) {
//                    if (ptx == 800) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
//                                                         256, 800, true, REAL>::val
//                                             : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
//                                                         256, 800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    800, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    800, false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
//                                                         256, 700, true, REAL>::val
//                                             : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD,
//                                                         256, 700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    700, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, 2 * ITEM_PER_THREAD, 256,
//                                                    700, false, REAL>::spill;
//                    }
//                } else {
//                    if (ptx == 800) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                        256,
//                                                         800, true, REAL>::val
//                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                                             256,
//                                                         800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                        800,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                                        800,
//                                                    false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                        256,
//                                                         700, true, REAL>::val
//                                             : regfolder<HALO, curshape, 128, ITEM_PER_THREAD,
//                                             256,
//                                                         700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                        700,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 128, ITEM_PER_THREAD, 256,
//                                        700,
//                                                    false, REAL>::spill;
//                    }
//                }
//            }
//        } else {
//            if (blkpsm >= 2) {
//                if (ptx == 800) {
//                    reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                    800,
//                                                     true, REAL>::val
//                                         : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                         800,
//                                                     false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 800,
//                                                false, REAL>::spill;
//                }
//                if (ptx == 700) {
//                    reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                    700,
//                                                     true, REAL>::val
//                                         : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128,
//                                         700,
//                                                     false, REAL>::val;
//                    ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
//                                                true, REAL>::spill
//                                    : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 128, 700,
//                                                false, REAL>::spill;
//                }
//            } else {
//                if (isDoubleTile) {
//                    if (ptx == 800) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
//                                                         256, 800, true, REAL>::val
//                                             : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
//                                                         256, 800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    800, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    800, false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
//                                                         256, 700, true, REAL>::val
//                                             : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD,
//                                                         256, 700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    700, true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, 2 * ITEM_PER_THREAD, 256,
//                                                    700, false, REAL>::spill;
//                    }
//                } else {
//                    if (ptx == 800) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                        256,
//                                                         800, true, REAL>::val
//                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                                             256,
//                                                         800, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                        800,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                                        800,
//                                                    false, REAL>::spill;
//                    }
//                    if (ptx == 700) {
//                        reg_folder_z = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                        256,
//                                                         700, true, REAL>::val
//                                             : regfolder<HALO, curshape, 256, ITEM_PER_THREAD,
//                                             256,
//                                                         700, false, REAL>::val;
//                        ifspill = useSM ? regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                        700,
//                                                    true, REAL>::spill
//                                        : regfolder<HALO, curshape, 256, ITEM_PER_THREAD, 256,
//                                        700,
//                                                    false, REAL>::spill;
//                    }
//                }
//            }
//        }
//    }
//    if (ifspill) return -5;
//
//#endif
//    // shared memory related
//    size_t executeSM = 0;
//#ifndef NAIVE
//    int basic_sm_space =
//        ((TILE_Y + 2 * HALO) * (TILE_X + HALO + isBOX) * (1 + HALO * 2) + 1) * sizeof(REAL);
//    executeSM = basic_sm_space;
//#endif
//    // printf("sm is %ld\n",executeSM);
//    // #if defined(GEN) || defined(MIX)
//    // int sharememory1 = basic_sm_space+2*BD_STEP_XY*FOLDER_Z*sizeof(REAL);
//    // int sharememory2 = sharememory1 + sizeof(REAL) * (SFOLDER_Z)*(TILE_Y*2-1)*TILE_X;
//    // #endif
//
//    // return 0;
//// shared memory related
//#ifdef USEMAXSM
//    int maxSharedMemory;
//    cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
//    int SharedMemoryUsed = maxSharedMemory - 2048;
//    cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
//                         SharedMemoryUsed);
//#endif
//
//#ifdef NAIVE
//    dim3 block_dim_1(global_bdimx, 4, 1);
//    dim3 grid_dim_1(width_x / global_bdimx, width_y / 4, height);
//
//    dim3 executeBlockDim = block_dim_1;
//    dim3 executeGridDim = grid_dim_1;
//
//#endif
//#ifdef BASELINE
//
//    dim3 block_dim_2(global_bdimx, 1, 1);
//    dim3 grid_dim_2(width_x / TILE_X, width_y / TILE_Y,
//                    max(1, min(height / (2 * HALO + 1),
//                               max(2, (sm_count * 8) * TILE_X * TILE_Y / width_x / width_y))));
//    // printf("<<%d,%d,%d>>\n",grid_dim_2.x,grid_dim_2.y,grid_dim_2.z);
//    // if(executeGridDim.z*(2*HALO+1)>=height)return -4;
//
//    // dim3 block_dim3(TILE_X, 1, 1);
//    // dim3 grid_dim3(MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current), 1,
//    //
//    sm_count*numBlocksPerSm_current/MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current));
//
//    dim3 executeBlockDim = block_dim_2;
//    dim3 executeGridDim = grid_dim_2;
//#endif
//    // #ifdef BASELINE_MEMWARP
//    //   dim3 block_dim_2(bdimx+2*TILE_X, 1, 1);
//    //   dim3 grid_dim_2(width_x/TILE_X,
//    //   width_y/TILE_Y,max(2,(sm_count*8)*TILE_X*TILE_Y/width_x/width_y));
//    //   // dim3 block_dim3(TILE_X, 1, 1);
//    //   // dim3 grid_dim3(MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current), 1,
//    //
//    sm_count*numBlocksPerSm_current/MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current));
//
//    //   dim3 executeBlockDim=block_dim_2;
//    //   dim3 executeGridDim=grid_dim_2;
//    // #endif
//
//#ifdef PERSISTENTLAUNCH
//    int max_sm_flder = 0;
//#endif
//
//    // printf("asdfjalskdfjaskldjfals;");
//
//#if defined(PERSISTENTTHREAD)
//    int numBlocksPerSm_current = 100;
//
//#if defined(GEN)
//    int reg_folder_z = REG_FOLDER_Z;
//#endif
//
//#if defined(GEN) || defined(GENWR)
//
//    executeSM += reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);
//#endif
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_current, execute_kernel,
//                                                  global_bdimx, executeSM);
//    cudaCheckError();
//    if (blkpsm <= 0) blkpsm = numBlocksPerSm_current;
//    numBlocksPerSm_current = min(blkpsm, numBlocksPerSm_current);
//    // numBlocksPerSm_current=1;
//    dim3 block_dim_3(global_bdimx, 1, 1);
//    dim3 grid_dim_3(width_x / TILE_X, width_y / TILE_Y,
//                    MIN(height, MAX(1, sm_count * numBlocksPerSm_current /
//                                           (width_x * width_y / TILE_X / TILE_Y))));
//    dim3 executeBlockDim = block_dim_3;
//    dim3 executeGridDim = grid_dim_3;
//
//    if (numBlocksPerSm_current == 0) return -3;
//        // printf("plckpersm is %d\n", numBlocksPerSm_current);
//        // printf("<%d,%d,%d>\n",width_x/TILE_X, width_y/TILE_Y, MIN(height,
//        // MAX(1,sm_count*numBlocksPerSm_current/(width_x*width_y/TILE_X/TILE_Y))));
//        // printf("plckpersm is %f\n", (double)executeSM);
//#endif
//    // printf("<%d,%d>\n",TILE_X,TILE_Y);
//    // return 0;
//    int minHeight = 0;
//#if defined(GEN) || defined(GENWR)
//
//    //
//    // printf("%d\n",numBlocksPerSm_current);
//    int perSMUsable = SharedMemoryUsed / numBlocksPerSm_current;
//    int perSMValsRemaind = (perSMUsable - basic_sm_space) / sizeof(REAL);
//    int reg_boundary = reg_folder_z * 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX);
//    max_sm_flder = (perSMValsRemaind - reg_boundary) /
//                   (2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) + TILE_X * TILE_Y);
//    // printf(">>%ld,%ld\n",perSMUsable/1024,perSMValsRemaind/1024);
//    //
//    printf(">>%ld,%ld\n",(perSMValsRemaind-reg_boundary),(2*HALO*(TILE_Y+TILE_X*2*isBOX)+TILE_X*TILE_Y));
//    //
//    printf(">>%ld\n",(max_sm_flder*(2*HALO*(TILE_Y+TILE_X*2*isBOX)+TILE_X*TILE_Y))*sizeof(REAL)/1024);
//    // return 0;
//    if (!useSM) max_sm_flder = 0;
//    if (useSM && max_sm_flder == 0) return -1;
//
//    int sharememory1 = 2 * HALO * (TILE_Y + TILE_X + 2 * isBOX) * (max_sm_flder + reg_folder_z) *
//                       sizeof(REAL);  // boundary
//    int sharememory2 = sharememory1 + sizeof(REAL) * (max_sm_flder) * (TILE_Y)*TILE_X;
//    executeSM = sharememory2 + basic_sm_space;
//
//    minHeight = (max_sm_flder + reg_folder_z + 2 * NOCACHE_Z) * executeGridDim.z;
//
//    // printf("numblkpsm is %ld\n",numBlocksPerSm_current);
//    // printf("smfolder is %ld\n",max_sm_flder);
//    // printf("SM is %ld/%ld KB\n",executeSM/1024,SharedMemoryUsed/1024);
//#endif
//
//    // if(executeGridDim.x*executeGridDim.y*executeGridDim.z<sm_count)return -2;
//    if (executeGridDim.z * (2 * HALO + 1) > height) return -4;
//    // if(max_sm_flder+reg_folder_z<2*halo)return -4;
//
//    // printf("<%d,%d,%d>",executeGridDim.x,executeGridDim.y,executeGridDim.z);
//
//    if (getminHeight) return (minHeight);
//
//    REAL* input;
//    cudaMalloc(&input, sizeof(REAL) * (height * width_x * width_y));
//    Check_CUDA_Error("Allocation Error!! : input\n");
//
//    cudaGetLastError();
////    cudaMemcpy(input, h_input, sizeof(REAL) * (height * width_x * width_y),
/// cudaMemcpyHostToDevice);
//    REAL* __var_1__;
//    cudaMalloc(&__var_1__, sizeof(REAL) * (height * width_x * width_y));
//    Check_CUDA_Error("Allocation Error!! : __var_1__\n");
//    REAL* __var_2__;
//    cudaMalloc(&__var_2__, sizeof(REAL) * (height * width_x * width_y));
//    Check_CUDA_Error("Allocation Error!! : __var_2__\n");
//
//    size_t L2_utage = width_y * height * sizeof(REAL) * HALO * (width_x / TILE_X) * 2 +
//                      width_x * height * sizeof(REAL) * HALO * (width_y / TILE_Y) * 2;
//
//    REAL* l2_cache1;
//    REAL* l2_cache2;
//    cudaMalloc(&l2_cache1, L2_utage);
//    cudaMalloc(&l2_cache2, L2_utage);
//#ifndef __PRINT__
//    printf("l2 cache used is %ld KB : 4096 KB \n", L2_utage / 1024);
//#endif
//
//    int l_warmupiteration = warmupiteration > 0 ? warmupiteration : 1000;
//
//#ifdef PERSISTENTLAUNCH
//    int l_iteration = iteration;
//    void* KernelArgs[] = {(void**)&input,     (void*)&__var_2__,   (void**)&height,
//                          (void**)&width_y,   (void*)&width_x,     (void**)&l2_cache1,
//                          (void**)&l2_cache2, (void*)&l_iteration, (void*)&max_sm_flder};
//    // #ifdef __PRINT__
//    void* KernelArgsNULL[] = {(void**)&__var_2__, (void*)&__var_1__,         (void**)&height,
//                              (void**)&width_y,   (void*)&width_x,           (void**)&l2_cache1,
//                              (void**)&l2_cache2, (void*)&l_warmupiteration,
//                              (void*)&max_sm_flder};
//    // #endif
//#endif
//    cudaCheckError();
//    // bool warmup=false;
//    if (usewarmup) {
//        cudaEvent_t warstart, warmstop;
//        cudaEventCreate(&warstart);
//        cudaEventCreate(&warmstop);
//#ifdef TRADITIONLAUNCH
//        {
//            cudaEventRecord(warstart, 0);
//            // cudaCheckError();
//            for (int i = 0; i < l_warmupiteration; i++) {
//                // execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
//                //       (__var_2__, width_y, width_x , __var_1__);
//                execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>(
//                    __var_2__, __var_1__, height, width_y, width_x);
//                REAL* tmp = __var_2__;
//                __var_2__ = __var_1__;
//                __var_1__ = tmp;
//            }
//            cudaEventRecord(warmstop, 0);
//            cudaEventSynchronize(warmstop);
//            cudaCheckError();
//            float warmelapsedTime;
//            cudaEventElapsedTime(&warmelapsedTime, warstart, warmstop);
//            float nowwarmup = (warmelapsedTime);
//            // nowwarmup = max()
//            int nowiter = (350 + nowwarmup - 1) / nowwarmup;
//
//            for (int out = 0; out < nowiter; out++) {
//                for (int i = 0; i < l_warmupiteration; i++) {
//                    // execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
//                    // (__var_2__, width_y, width_x , __var_1__);
//                    execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>(
//                        __var_2__, __var_1__, height, width_y, width_x);
//                    REAL* tmp = __var_2__;
//                    __var_2__ = __var_1__;
//                    __var_1__ = tmp;
//                }
//            }
//        }
//#endif
//
//#ifdef PERSISTENTLAUNCH
//        {
//            // double accumulate=0;
//            cudaEventRecord(warstart, 0);
//            cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim,
//                                        KernelArgsNULL, executeSM, 0);
//            cudaEventRecord(warmstop, 0);
//            cudaEventSynchronize(warmstop);
//            cudaCheckError();
//            float warmelapsedTime;
//            cudaEventElapsedTime(&warmelapsedTime, warstart, warmstop);
//            int nowwarmup = warmelapsedTime;
//            int nowiter = (350 + nowwarmup - 1) / nowwarmup;
//            for (int i = 0; i < nowiter; i++) {
//                cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim,
//                executeBlockDim,
//                                            KernelArgsNULL, executeSM, 0);
//            }
//        }
//#endif
//    }
//
//#ifdef _TIMER_
//    cudaEvent_t _forma_timer_start_, _forma_timer_stop_;
//    cudaEventCreate(&_forma_timer_start_);
//    cudaEventCreate(&_forma_timer_stop_);
//    cudaEventRecord(_forma_timer_start_, 0);
//#endif
//
//#ifdef TRADITIONLAUNCH
//    execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>(input, __var_2__, height,
//                                                                   width_y, width_x);
//
//    for (int i = 1; i < iteration; i++) {
//        execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>(__var_2__, __var_1__,
//        height,
//                                                                       width_y, width_x);
//        REAL* tmp = __var_2__;
//        __var_2__ = __var_1__;
//        __var_1__ = tmp;
//    }
//#endif
//#ifdef PERSISTENTLAUNCH
//    printf("executeGridDim: %d %d %d\n", executeGridDim.x, executeGridDim.y, executeGridDim.z);
//    printf("executeBlockDim: %d %d %d\n", executeBlockDim.x, executeBlockDim.y,
//    executeBlockDim.z);
//
//    cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim,
//    KernelArgs,
//                                executeSM, 0);
//#endif
//    cudaDeviceSynchronize();
//    cudaCheckError();
//#ifdef _TIMER_
//    cudaEventRecord(_forma_timer_stop_, 0);
//    cudaEventSynchronize(_forma_timer_stop_);
//    float elapsedTime;
//    cudaEventElapsedTime(&elapsedTime, _forma_timer_start_, _forma_timer_stop_);
//#ifndef __PRINT__
//    printf("[FORMA] SIZE : %d,%d,%d\n", height, width_y, width_x);
//    printf("[FORMA] Computation Time(ms) : %lf\n", elapsedTime);
//    printf("[FORMA] Speed(GCells/s) : %lf\n",
//           (REAL)iteration * height * width_x * width_y / elapsedTime / 1000 / 1000);
//    printf("[FORMA] Computation(GFLOPS/s) : %lf\n", (REAL)iteration * height * width_x * width_y *
//                                                        (HALO * 2 + 1) * (HALO * 2 + 1) /
//                                                        elapsedTime / 1000 / 1000);
//    printf("[FORMA] Bandwidth(GB/s) : %lf\n", (REAL)iteration * height * width_x * width_y *
//                                                  sizeof(REAL) * 2 / elapsedTime / 1000 / 1000);
//#if defined(GEN) || defined(GENWR)
//    printf("[FORMA] rfder : %d\n", reg_folder_z);
//#endif
//#ifdef PERSISTENTLAUNCH
//    printf("[FORMA] sfder : %d\n", max_sm_flder);
//    // printf("[FORMA] sm : %f\n",executeSM/1024);
//#endif
//#else
//    // h y x iter TILEX thready=1 gridx gridy latency speed
//    printf("%d\t%d\t", ptx, sizeof(REAL) / 4);
//    printf("%d\t%d\t%d\t%d\t", height, width_y, width_x, iteration);
//    printf("%d\t%d\t<%d,%d,%d>\t%d\t%d\t", executeBlockDim.x, LOCAL_ITEM_PER_THREAD,
//           executeGridDim.x, executeGridDim.y, executeGridDim.z, sm_count,
//           (executeGridDim.x) * (executeGridDim.y) * (executeGridDim.z) / sm_count);
//#ifndef NAIVE
//    printf("%f\t", (double)basic_sm_space / 1024);
//#endif
//    printf("%f\t%lf\t", elapsedTime,
//           (REAL)iteration * height * width_x * width_y / elapsedTime / 1000 / 1000);
//#if defined(GEN) || defined(GENWR)
//    printf("%d\t%d\t%d\t", reg_folder_z, max_sm_flder, executeSM / 1024);
//#endif
//    printf("\n");
//#endif
//    cudaEventDestroy(_forma_timer_start_);
//    cudaEventDestroy(_forma_timer_stop_);
//#endif
//    cudaDeviceSynchronize();
//    cudaCheckError();
//
//#if defined(PERSISTENTLAUNCH)
//    // || defined(PERSISTENT)
//    if (iteration % 2 == 1) {
//        cudaMemcpy(__var_0__, __var_2__, sizeof(REAL) * height * width_x * width_y,
//                   cudaMemcpyDeviceToHost);
//    } else {
//        cudaMemcpy(__var_0__, input, sizeof(REAL) * height * width_x * width_y,
//                   cudaMemcpyDeviceToHost);
//    }
//#else
//    cudaMemcpy(__var_0__, __var_2__, sizeof(REAL) * height * width_x * width_y,
//               cudaMemcpyDeviceToHost);
//#endif
//    cudaDeviceSynchronize();
//    cudaCheckError();
//
//    cudaFree(input);
//    cudaFree(__var_1__);
//    cudaFree(__var_2__);
//    cudaFree(l2_cache1);
//    cudaFree(l2_cache2);
//    return 0;
//}
//
// PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_ITERATIVE);
//
//// template void j3d_iterative<float>(float * h_input, int height, int width_y, int width_x, float
///* / __var_0__, int iteration); template void j3d_iterative<double>(float * h_input, int height,
/// int / width_y, int width_x, float * __var_0__, int iteration);
