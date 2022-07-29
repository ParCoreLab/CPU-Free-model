
#define ISINITI (true)
#define NOTINITIAL (false)
#define SYNC (true)
#define NOSYNC (false)

//#undef TILE_Y
// #define USESM

// #ifdef USESM
//   #define USESMSET (true)
// #else
//   #define USESMSET (false)
// #endif



#ifndef BOX
#define stencilParaT \
  const REAL west[6]={12.0/118,9.0/118,3.0/118,2.0/118,5.0/118,6.0/118};\
  const REAL east[6]={12.0/118,9.0/118,3.0/118,3.0/118,4.0/118,6.0/118};\
  const REAL north[6]={5.0/118,7.0/118,5.0/118,4.0/118,3.0/118,2.0/118};\
  const REAL south[6]={5.0/118,7.0/118,5.0/118,1.0/118,6.0/118,2.0/118};\
  const REAL center=15.0/118;
;
  #define stencilParaList const REAL west[6],const REAL east[6],const REAL north[6],const REAL south[6],const REAL center
  #define stencilParaInput  west,east,north,south,center
  #define R_PTR r_ptr[INPUTREG_SIZE]
  #define isBOX (0)
  #define isStar (true)
#else
  #if HALO==1
    #define stencilParaT \
    const REAL filter[3][3] = {\
        {7.0/118, 5.0/118, 9.0/118},\
        {12.0/118,15.0/118,12.0/118},\
        {9.0/118, 5.0/118, 7.0/118}\
    };
  #endif
  #if HALO==2
  #define stencilParaT \
  const REAL filter[5][5] = {\
    {1.0/118, 2.0/118, 3.0/118, 4.0/118, 5.0/118},\
    {7.0/118, 7.0/118, 5.0/118, 7.0/118, 6.0/118},\
    {8.0/118,12.0/118,15.0/118,12.0/118,12.0/118},\
    {9.0/118, 9.0/118, 5.0/118, 7.0/118, 15.0/118},\
    {10.0/118, 11.0/118, 12.0/118, 13.0/118, 14.0/118}\
  };
  #endif
  #define stencilParaList const REAL filter[halo*2+1][halo*2+1]
  #define stencilParaInput  filter
  #define R_PTR r_ptr[2*halo+1][INPUTREG_SIZE]
  #define isBOX (halo)
  #define isStar (false)
#endif


template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE+2*halo)>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE], 
                                            REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
                                            REAL R_PTR,
                                            int reg_base, 
                                            stencilParaList
                                            // const REAL west[6],const REAL east[6], 
                                            // const REAL north[6],const REAL south[6],
                                            // const REAL center 
                                          )
{
#ifndef BOX
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind-1-hl]*west[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y-1-hl]*south[hl];
    }
  }
  _Pragma("unroll")
  for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
  {
    result[l_y]+=r_ptr[reg_base+l_y]*center;
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y+1+hl]*north[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind+1+hl]*east[hl];
    }
  }
#else
  _Pragma("unroll")\
  for(int hl_y=-halo; hl_y<=halo; hl_y++)
  {
    _Pragma("unroll")
    for(int hl_x=-halo; hl_x<=halo; hl_x++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {
        result[l_y]+=filter[hl_y+halo][hl_x+halo]*r_ptr[hl_x+halo][hl_y+halo+l_y];
      }
    }
  }
#endif
}
//special for box version
#ifdef BOX
template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE+2*halo), int CACHESIZE>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE], 
                                            REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
                                            REAL R_PTR,
                                            REAL r_space[CACHESIZE],
                                            int reg_base, 
                                            stencilParaList
                                            // const REAL west[6],const REAL east[6], 
                                            // const REAL north[6],const REAL south[6],
                                            // const REAL center 
                                          )
{

  _Pragma("unroll")\
  for(int hl_y=-halo; hl_y<=halo; hl_y++)
  {
    _Pragma("unroll")
    for(int hl_x=-halo; hl_x<0; hl_x++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {
        result[l_y]+=filter[hl_y+halo][hl_x+halo]*r_ptr[hl_x+halo][hl_y+halo+l_y];
      }
    }
    // _Pragma("unroll")
    // for(int hl_x=-halo; hl_x<=halo; hl_x++)
    // {
    //   _Pragma("unroll")
    //   for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    //   {
    //     result[l_y]+=filter[hl_y+halo][0+halo]*r_ptr[0+halo][hl_y+halo+l_y];
    //   }
    // }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=filter[hl_y+halo][0+halo]*r_space[hl_y+reg_base+l_y];
    }
    _Pragma("unroll")
    for(int hl_x=1; hl_x<=halo; hl_x++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {
        result[l_y]+=filter[hl_y+halo][hl_x+halo]*r_ptr[hl_x+halo][hl_y+halo+l_y];
      }
    }
  }
}
#endif
template<class REAL, int halo>
__global__ void
kernel2d_restrict (REAL* input, int width_y, int width_x, REAL* output);

template<class REAL, int halo>
__global__ void
kernel2d_restrict_box (REAL* input, int width_y, int width_x, REAL* output);


#define PERKS_DECLARE_INITIONIZATION_REFERENCE(_type,halo) \
    __global__ void kernel2d_restrict<_type,halo>(_type*,int,int,_type*);

#define PERKS_DECLARE_INITIONIZATION_REFERENCE_BOX(_type,halo) \
    __global__ void kernel2d_restrict_box<_type,halo>(_type*,int,int,_type*);


template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void
kernel_baseline (REAL*__restrict__ input, int width_y, int width_x, REAL*__restrict__ output);

template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void
kernel_baseline_box (REAL* __restrict__ input, int width_y, int width_x, REAL* __restrict__ output);

template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void
kernel_baseline_async (REAL*__restrict__ input, int width_y, int width_x, REAL*__restrict__ output);

template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void
kernel_baseline_box_async (REAL* __restrict__ input, int width_y, int width_x, REAL* __restrict__ output);


#define PERKS_DECLARE_INITIONIZATION_BASELINE(_type,tile,halo) \
    __global__ void kernel_baseline<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

#define PERKS_DECLARE_INITIONIZATION_BASELINE_BOX(_type,tile,halo) \
    __global__ void kernel_baseline_box<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

#define PERKS_DECLARE_INITIONIZATION_BASELINE_ASYNC(_type,tile,halo) \
    __global__ void kernel_baseline_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

#define PERKS_DECLARE_INITIONIZATION_BASELINE_BOX_ASYNC(_type,tile,halo) \
    __global__ void kernel_baseline_box_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);



template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void kernel_persistent_baseline(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration);

  template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void kernel_persistent_baseline_box( REAL * __restrict__ input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration);

template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void kernel_persistent_baseline_async(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration);

template<class REAL, int LOCAL_TILE_Y, int halo>
__global__ void kernel_persistent_baseline_box_async( REAL * __restrict__ input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration);

#define PERKS_DECLARE_INITIONIZATION_PBASELINE(_type,tile,halo) \
    __global__ void kernel_persistent_baseline<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

#define PERKS_DECLARE_INITIONIZATION_PBASELINE_BOX(_type,tile,halo) \
    __global__ void kernel_persistent_baseline_box<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

#define PERKS_DECLARE_INITIONIZATION_PBASELINE_ASYNC(_type,tile,halo) \
    __global__ void kernel_persistent_baseline_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

#define PERKS_DECLARE_INITIONIZATION_PBASELINE_BOX_ASYNC(_type,tile,halo) \
    __global__ void kernel_persistent_baseline_box_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, int minblocks, bool UseSMCache>
__global__ void kernel_general(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration, int max_sm_flder);

template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, int minblocks, bool UseSMCache>
__global__ void kernel_general_async(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration, int max_sm_flder);

template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, int minblocks, bool UseSMCache>
__global__ void kernel_general_box( REAL * __restrict__ input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration, int max_sm_flder);

template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, int minblocks, bool UseSMCache>
__global__ void kernel_general_box_async( REAL * __restrict__ input, int width_y, int width_x, 
  REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
  int iteration, int max_sm_flder);


#define PERKS_DECLARE_INITIONIZATION_GENERAL(_type,tile,halo,rf,minblocks,usesm) \
    __global__ void kernel_general<_type,tile,halo,rf,minblocks,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);

#define PERKS_DECLARE_INITIONIZATION_GENERALBOX(_type,tile,halo,rf,minblocks,usesm) \
    __global__ void kernel_general_box<_type,tile,halo,rf,minblocks,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);

#define PERKS_DECLARE_INITIONIZATION_GENERAL_ASYNC(_type,tile,halo,rf,minblocks,usesm) \
    __global__ void kernel_general_async<_type,tile,halo,rf,minblocks,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);

#define PERKS_DECLARE_INITIONIZATION_GENERALBOX_ASYNC(_type,tile,halo,rf,minblocks,usesm) \
    __global__ void kernel_general_box_async<_type,tile,halo,rf,minblocks,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);


#include "../perksconfig.cuh"
#include "./cuda_common.cuh"
template<class REAL, int LOCAL_TILE_Y, int halo, 
          int registeramount, bool UseSMCache, bool isstar=(isBOX==0),
          int minblocks=256/registeramount>
__global__ void  kernel_general_wrapper
(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__, 
  REAL * __restrict__ l2_cache_o,REAL * __restrict__ l2_cache_i,
  int iteration,
  int max_sm_flder);

#define PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER(_type,tile,halo,ramount,usesm) \
    __global__ void kernel_general_wrapper<_type,tile,halo,ramount,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);









