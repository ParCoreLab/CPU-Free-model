#include "./config.cuh"
// #include "./genconfig.cuh"
#include "./perksconfig.cuh"
#include "./jacobi-general-kernel.cuh"

// #ifdef SMASYNC
//   #include <cooperative_groups/memcpy_async.h>
//   #include <cuda_pipeline.h>
// #endif
#include "./common/cuda_common.cuh"

#define MAXTHREAD (256)
// #define MINBLOCK (1)
template<class REAL, int LOCAL_TILE_Y, int halo, 
          int registeramount, bool UseSMCache, bool isstar,
          int minblocks>
__launch_bounds__(MAXTHREAD, minblocks)
__global__ void kernel_general_wrapper(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__, 
  REAL * __restrict__ l2_cache_o,REAL * __restrict__ l2_cache_i,
  int iteration,
  int max_sm_flder)
{
  inner_general<REAL, LOCAL_TILE_Y, halo, 
  regfolder<halo,isstar,registeramount,PERKS_ARCH,UseSMCache,REAL,LOCAL_TILE_Y>::val, 
  // 1, 
  UseSMCache>( input,  width_y,  width_x, 
    __var_4__, 
    l2_cache_o, l2_cache_i,
    iteration,
    max_sm_flder);
}

PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,8,HALO,128,true);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,8,HALO,128,false);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,8,HALO,256,true);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,8,HALO,256,false);

PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,16,HALO,128,true);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,16,HALO,128,false);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,16,HALO,256,true);
PERKS_INITIALIZE_ALL_TYPE_4ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER,16,HALO,256,false);
// #if PERKS_ARCH==800
// #elif PERKS_ARCH==700
// #elif PERKS_ARCH==600
// #error "should not be 600"
// #elif PERKS_ARCH==000
// #error "undefined"
// #else
// #error "wrong architecture"
// #endif
// template<> 
// __global__ void kernel_general_wrapper<float,RTILE_Y,HALO,256,true>
// ( float * __restrict__ input, int width_y, int width_x, 
//   float * __restrict__ __var_4__, 
//   float * __restrict__ l2_cache_o,float * __restrict__ l2_cache_i,
//   int iteration,
//   int max_sm_flder);