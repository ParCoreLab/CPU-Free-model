#include "./config.cuh"
// #include "./genconfig.cuh"
#include "./j3d-general-kernels.cuh"
#include "./perksconfig.cuh"

#include <math.h>
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/types.hpp"

#include <cooperative_groups.h>

#ifdef USESM
#define isUseSM (true)
#else
#define isUseSM (false)
#endif

// #define NOCACHE_Z (HALO)

#define MAXTHREAD (256)

template <class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int registeramount,
          bool UseSMCache, int BLOCKDIM, int shape, int minblocks>
__launch_bounds__(MAXTHREAD, minblocks) __global__
    void kernel3d_general_wrapper(REAL* __restrict__ input, REAL* __restrict__ output, int width_z,
                                  int width_y, int width_x, REAL* l2_cache_i, REAL* l2_cache_o,
                                  int iteration, volatile int* iteration_done, int max_sm_flder) {
    kernel3d_general_inner<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X,
                           // reg_folder_z,
                           regfolder<halo, shape, BLOCKDIM, LOCAL_ITEM_PER_THREAD, registeramount,
                                     PERKS_ARCH, UseSMCache, REAL>::val,
                           UseSMCache, BLOCKDIM>(input, output, width_z, width_y, width_x,
                                                 l2_cache_i, l2_cache_o, iteration, iteration_done,
                                                 max_sm_flder);

    // inner_general<REAL, LOCAL_TILE_Y, halo,
    // regfolder<halo,isstar,registeramount,PERKS_ARCH,UseSMCache,REAL,LOCAL_TILE_Y>::val,
    // // 1,
    // UseSMCache>( input,  width_y,  width_x,
    //   __var_4__,
    //   l2_cache_o, l2_cache_i,
    //   iteration,
    //   max_sm_flder);
}

#ifndef BOX
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 128, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 128, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 128, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 128, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 128, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 128, star_shape);

PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 256, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 256, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 256, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 256, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 256, star_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 256, star_shape);

#else
#ifndef TYPE0
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 128, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 128, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 128, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 128, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 128, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 128, box_shape);

PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 256, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 256, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 256, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 256, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 256, box_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 256, box_shape);
#else
#ifdef POISSON
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 128, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 128, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 128, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 128, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 128, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 128, poisson_shape);

PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 256, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 256, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 256, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 256, poisson_shape);

PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 256, poisson_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 256, poisson_shape);
#else
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 128, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 128, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 128, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 128, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 128, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 128, type0_shape);

PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               true, 256, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 128,
                               false, 256, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               true, 256, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 16, TILE_X, 256,
                               false, 256, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               true, 256, type0_shape);
PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER, HALO, 8, TILE_X, 256,
                               false, 256, type0_shape);
#endif
#endif
#endif
// PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,HALO,ITEM_PER_THREAD,TILE_X,128,true);
// PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,HALO,ITEM_PER_THREAD,TILE_X,128,false);
// PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,HALO,ITEM_PER_THREAD,TILE_X,256,true);
// PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,HALO,ITEM_PER_THREAD,TILE_X,256,false);

// PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,1HALO,128,true);
// PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,1HALO,128,false);
// PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,1HALO,256,true);
// PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,1HALO,256,false);
