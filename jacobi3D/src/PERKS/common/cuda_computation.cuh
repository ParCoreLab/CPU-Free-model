#pragma once
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



// #define bdimx (256)
// #define bdimx (128)

// #define ITEM_PER_THREAD (8) 
// #ifndef ITEM_PER_THREAD
//   #define ITEM_PER_THREAD (8) 
// #endif
// (TILE_Y/(bdimx/TILE_X))
#ifndef TILE_X
  #define TILE_X (64)
#endif
// #define TILE_Y (ITEM_PER_THREAD*bdimx/TILE_X)

// #define NOCACHE_Y (HALO)

#ifndef NOCACHE_Z
#define NOCACHE_Z (HALO)
#endif
// (32)

// #define RTILE_Z (1)

// #ifndef X_FACTOR
// #define X_FACTOR (16)
// #endif
// #ifndef RFOLDER_Z
// #define RFOLDER_Z (3)
// #endif
// #ifndef SFOLDER_Z
// #define SFOLDER_Z (0)
// #endif 
// #define FOLDER_Z (RFOLDER_Z+SFOLDER_Z)
// #define SKIP (1)
// #define SM_X (TILE_X  + 2 * halo)
// #define SM_Y (TILE_Y  + 2 * halo)
// #define SM_H (RTILE_Z + 2 * halo)
// #ifndef RFOLDER_Z
// #define RFOLDER_Z (3)
// #endif


#ifndef BOX
  #define curshape (star_shape)
  #if HALO==1
    #define stencilParaT \
      const REAL center=-1.67f;\
      const REAL west[1]={0.162f};\
      const REAL east[1]={0.161f};\
      const REAL north[1]={0.163f};\
      const REAL south[1]={0.164f};\
      const REAL bottom[1]={0.166f};\
      const REAL top[1]={0.165f};
    #endif
    #if HALO==2
      #define stencilParaT \
        const REAL center=-0.996f;\
        const REAL west[2]={0.083f,0.083f};\
        const REAL east[2]={0.083f,0.083f};\
        const REAL north[2]={0.083f,0.083f};\
        const REAL south[2]={0.083f,0.083f};\
        const REAL bottom[2]={0.083f,0.083f};\
        const REAL top[2]={0.083f,0.083f};
    #endif
    #define isBOX (0)
    #define stencilParaList const REAL west[HALO],const REAL east[HALO],const REAL north[HALO],const REAL south[HALO],const REAL top[HALO], const REAL bottom[HALO], const REAL center
    #define stencilParaInput  west,east,north,south,top,bottom,center
    #define REG_Y_SIZE_MOD (LOCAL_ITEM_PER_THREAD)
#else
  #ifndef TYPE0
    #define curshape (box_shape)
    #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0.5/159, 0.7/159, 0.90/159},\
          {1.2/159, 1.5/159, 1.2/159},\
          {0.9/159, 0.7/159, 0.50/159}\
        },\
        { {0.51/159, 0.71/159, 0.91/159},\
          {1.21/159, 1.51/159, 1.21/159},\
          {0.91/159, 0.71/159, 0.51/159}\
        },\
        { {0.52/159, 0.72/159, 0.920/159},\
          {1.22/159, 1.52/159, 1.22/159},\
          {0.92/159, 0.72/159, 0.520/159}\
        }\
      };
  #else
    #ifdef POISSON
      #define curshape (poisson_shape)
      #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        },\
        { {-0.0833f,        -0.166f,   -0.0833f},\
          {-0.166f, 2.666f,     -0.166f},\
          {-0.0833f,       -0.166f,    -0.0833f}\
        },\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        }\
      };
    #else
      #define curshape (type0_shape)
      #define stencilParaT \
        const REAL filter[3][3][3] = {\
          { {0.50/159,  0.0,  0.50/159},\
            {0.0,   0.0,  0.0},\
            {0.50/159,  0.0,  0.50/159}\
          },\
          { {0.51/159,  0.71/159, 0.91/159},\
            {1.21/159,  1.51/159, 1.21/159},\
            {0.91/159,  0.71/159, 0.51/159}\
          },\
          { {0.52/159,  0.0,  0.52/159},\
            {0.0,   0.0,  0.0},\
            {0.52/159,  0.0,  0.52/159}\
          }\
        };
    #endif
  #endif
  
  #define stencilParaList const REAL filter[halo*2+1][halo*2+1][halo*2+1]
  #define stencilParaInput  filter
  #define isBOX (HALO)
  #define REG_Y_SIZE_MOD (1+2*halo)
#endif


// template<class REAL, int RESULT_SIZE, int halo, int SMZ_SIZE=halo+1+halo, int REGZ_SIZE=2*halo+1, int REGY_SIZE=REG_Y_SIZE_MOD, int REGX_SIZE=2*halo+1, int REG_BASE=halo>
template<class REAL, int RESULT_SIZE, int halo, int REGY_SIZE,  int REGZ_SIZE=2*halo+1, int REGX_SIZE=2*halo+1, int REG_BASE=halo, int SMZ_SIZE=halo+1+halo>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE],
                                            REAL* sm_ptr[SMZ_SIZE], 
                                            int sm_y_base, int sm_width, int sm_x_ind,
#ifndef BOX
                                            REAL reg_ptr[REGZ_SIZE][RESULT_SIZE],
#else
                                            REAL reg_ptr[REGZ_SIZE][REGY_SIZE][REGX_SIZE],
#endif                                  
                                            stencilParaList)
{
  #ifndef BOX
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=west[hl]*
              sm_ptr[0][sm_width*(l_y+sm_y_base) + sm_x_ind-1-hl];
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=east[hl]*
          sm_ptr[0][sm_width*(l_y+sm_y_base) + sm_x_ind+1+hl];
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=north[hl]*
          sm_ptr[0][sm_width*(l_y+sm_y_base+1+hl) + sm_x_ind];
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=south[hl]*
          sm_ptr[0][sm_width*(l_y+sm_y_base-1-hl) + sm_x_ind];
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=bottom[hl]*reg_ptr[REG_BASE-1-hl][l_y];
        result[l_y]+=top[hl]*reg_ptr[REG_BASE+1+hl][l_y];
      }
    }
  }
  _Pragma("unroll")
  for(int l_y=0; l_y<RESULT_SIZE; l_y++)
  {
    result[l_y]+=center*reg_ptr[REG_BASE][l_y];
  }
  #else
    // register REAL reg_ptr[REGZ_SIZE][REGY_SIZE][REGX_SIZE];
   _Pragma("undroll")
    for(int l_y=0; l_y<2*halo; l_y++)
    {
      for(int l_z=0; l_z<2*halo+1; l_z++)
      {
        for(int l_x=0; l_x<2*halo+1; l_x++)
        {
          reg_ptr[l_z][l_y][l_x]
            = 
            sm_ptr[(l_z)][sm_width*(l_y+sm_y_base-halo) + sm_x_ind-halo+l_x];
            
        }
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      {
        int h_y=2*halo;
        _Pragma("unroll")
        for(int h_z=0; h_z<2*halo+1; h_z++)
        {
          _Pragma("unroll")
          for(int h_x=0; h_x<2*halo+1; h_x++)
          {
            reg_ptr[h_z][h_y][h_x]
              = 
              sm_ptr[(h_z)][sm_width*(l_y + h_y+ sm_y_base-halo) + sm_x_ind-halo+h_x];
              // smbuffer_buffer_ptr[(h_z+halo)][tile_x_with_halo*(l_y + h_y-halo+ps_y+index_y) + tid_x+ps_x+h_x];
          }
        }
      }
      
      _Pragma("unroll")
      for(int hl_z=-halo; hl_z<halo+1; hl_z++)
      {
        _Pragma("unroll")
        for(int hl_y=-halo; hl_y<halo+1; hl_y++)
        {
          _Pragma("unroll")
          for(int hl_x=-halo; hl_x<halo+1; hl_x++)
          {
            result[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              // sm_ptr[hl_z+halo][sm_width*(l_y+halo + hl_y+ sm_y_base-halo) + sm_x_ind-halo+hl_x+halo];
              reg_ptr[hl_z+halo][hl_y+halo][hl_x+halo];
          }
        }
      }
      for(int l_y=0; l_y<2*halo; l_y++)
      {
        for(int l_z=0; l_z<2*halo+1; l_z++)
        {
          for(int l_x=0; l_x<2*halo+1; l_x++)
          {
            reg_ptr[l_z][l_y][l_x]=reg_ptr[l_z][l_y+1][l_x];
          }
        }
      }
    }


  #endif
}


template<class REAL, int halo>
__global__ void kernel3d_restrict(REAL* input, REAL* output,
                                  int height, int width_y, int width_x); 
#define PERKS_DECLARE_INITIONIZATION_NAIVE(_type,halo) \
    __global__ void kernel3d_restrict<_type,halo>(_type*,_type*,int,int,int);

template<class REAL, int halo , int ipt, int tilex, int blockdim=256>
__global__ void kernel3d_baseline(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x); 
#define PERKS_DECLARE_INITIONIZATION_BASELINE(_type,halo,ipt,tilex,blockdim) \
    __global__ void kernel3d_baseline<_type,halo,ipt,tilex,blockdim>(_type*__restrict__,_type*__restrict__,int,int,int);


// template<class REAL, int halo , int ipt, int tilex, int tiley>
// __global__ void kernel3d_baseline_memwarp(REAL* __restrict__ input, REAL*__restrict__ output,
//                                   int height, int width_y, int width_x); 
// #define PERKS_DECLARE_INITIONIZATION_BASELINE_MEMWARP(_type,halo,ipt,tilex,tiley) \
//     __global__ void kernel3d_baseline_memwarp<_type,halo,ipt,tilex,tiley>(_type*__restrict__,_type*__restrict__,int,int,int);


template<class REAL, int halo, int ipt, int tilex, int blockdim=256>
__global__ void kernel3d_persistent(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x, 
                                  REAL * l2_cache_i, REAL * l2_cache_o, 
                                  int iteration); 
#define PERKS_DECLARE_INITIONIZATION_PERSISTENT(_type,halo,ipt,tilex,blockdim) \
    __global__ void kernel3d_persistent<_type,halo,ipt,tilex,blockdim>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, int);


// minblocks:
// 256 bdim
// float: 
//   ipt=8 -> 1 or 2
//   ipt=16 -> 1
// double:
//   -> 1
//  128 bdim:
// ( 256 bdim )*2
template<class REAL, int blocktype, int itemperblock>
struct getminblocks
{
  static int const val=1;
};
template<>
struct getminblocks<float,2,8>
//only in float precision and itb==8 we need a situation of 
{
  static int const val=2;  
};
template<>
struct getminblocks<double,2,8>
//only in float precision and itb==8 we need a situation of 
{
  static int const val=2;  
};


template<class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int tilex, 
    int reg_folder_z=0,int blocktype,  bool UseSMCache=false, 
    int BLOCKDIM=256, 
    int minblocks=getminblocks<REAL,blocktype,LOCAL_ITEM_PER_THREAD>::val>
__global__ void kernel3d_general(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x, 
                                  REAL * l2_cache_i, REAL * l2_cache_o, 
                                  int iteration, int max_sm_flder=0); 

#define PERKS_DECLARE_INITIONIZATION_GENERAL(_type,halo,ipt,tilex,regf,minblocks, usesm,blckdim) \
    __global__ void kernel3d_general<_type,halo,ipt,tilex,regf,minblocks,usesm,blckdim>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, int, int);



#include "../perksconfig.cuh"
#include "./cuda_common.cuh"
template<class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X,
          int registeramount, bool UseSMCache, int BLOCKDIM, int shape=star_shape,
          int minblocks=256/registeramount>
__global__ void  kernel3d_general_wrapper
(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x, 
                                  REAL * l2_cache_i, REAL * l2_cache_o, 
                                  int iteration, int max_sm_flder=0);

#define PERKS_DECLARE_INITIONIZATION_GENERAL_WRAPPER(_type,halo,ttile,tilex,ramount,usesm,blockdim,shape) \
    __global__ void kernel3d_general_wrapper<_type,halo,ttile,tilex,ramount,usesm,blockdim,shape>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, int, int);


