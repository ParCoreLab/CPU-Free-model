// #include "cuda.h
#pragma once
#include "stdio.h"

#ifndef CUDACOMMON
#define CUDACOMMON

#ifndef __CUDA_ARCH__
    #define PERKS_ARCH 000
#else
    #if __CUDA_ARCH__==800
        #define PERKS_ARCH 800
    #elif __CUDA_ARCH__==700
        #define PERKS_ARCH 700
    #elif __CUDA_ARCH__==600
        #define PERKS_ARCH 600
    #else
        #error "unsupport"
    #endif
#endif

#ifdef ASYNCSM
  // #if PERKS_ARCH<800 
  //   #error "unsupport architecture" ${PERKS_ARCH}
  // #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif



extern __host__ __device__ __forceinline__ int MAX(int a, int b) { return a > b ? a : b; }
extern __host__ __device__ __forceinline__ int MIN(int a, int b) { return a < b ? a : b; }
extern __host__ __device__ __forceinline__ int CEIL(int a, int b) { return ( (a) % (b) == 0 ? (a) / (b) :  ( (a) / (b) + 1 ) ); }

// void Check_CUDA_Error(const char* message){
//   cudaError_t error = cudaGetLastError();
//   if( error != cudaSuccess ){
//     printf("CUDA-ERROR:%s, %s\n",message,cudaGetErrorString(error) ); 
//     exit(-1);
//   }
// }
#define Check_CUDA_Error(message) \
  do{\
    cudaError_t error = cudaGetLastError();\
    if( error != cudaSuccess ){\
      printf("CUDA-ERROR:%s, %s\n",message,cudaGetErrorString(error) ); \
      exit(-1);\
    }\
  }while(0)
//#ifndef PERKS_CUDA_HEADER
//#define PERKS_CUDA_HEADER
//template<class REAL>

//this is where the aimed implementation located
//template<class REAL>
//void jacobi_iterative(REAL*, int, int, REAL*, int );

//#define PERKS_DECLARE_INITIONIZATION_ITERATIVE(_type) \
    void jacobi_iterative(_type*,int,int,_type*, int);

//#endif

// init register array of ARRAY

template<class REAL, int SIZE>
__device__ void __forceinline__ init_reg_array(REAL reg_array[SIZE], int val)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    reg_array[l_y]=val;
  }
}



template<class REAL, int SIZE_REG, int SIZE=SIZE_REG, int REG_BASE=0>
__device__ void __forceinline__ reg2global3d(
            REAL reg_array[SIZE_REG], REAL*dst,
            int global_z, int width_z,
            int global_y, int width_y,
            int global_x, int width_x,
            int tid_x)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    dst[global_z*width_x*width_y+(global_y+l_y)*width_x+global_x+tid_x]=reg_array[l_y+REG_BASE];
  }
}



template<class REAL,int REG_SIZE_Z, int REG_SIZE_Y, 
                        int BASE_Z=0,  int BASE_Y=0, 
                        int SIZE_Z=REG_SIZE_Z, int SIZE_Y=REG_SIZE_Y>
__device__ void __forceinline__ global2regs3d(
  REAL*src, REAL reg_array[REG_SIZE_Z][REG_SIZE_Y],
  int global_z, int width_z,
  int global_y, int width_y,
  int global_x, int width_x,
  int tid_x)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE_Y; l_y++)
  {
    int l_global_y = (MIN(global_y+l_y+BASE_Y,width_y-1));
    l_global_y = (MAX(l_global_y,0));
    _Pragma("unroll")
    for(int l_z=0; l_z<SIZE_Z ; l_z++)
    {
      int l_global_z = (MIN(global_z+l_z+BASE_Z,width_z-1));
        l_global_z = (MAX(l_global_z,0));
      reg_array[l_z+BASE_Z][l_y+BASE_Y] = src[l_global_z*width_x*width_y+l_global_y*width_x+
            ((global_x+tid_x))];
    }
  }
}

template<class REAL, int START, int END, int halo>
__device__ void __forceinline__ ptrselfcp(REAL *ptr, 
                                      int ps_y, int y_step, 
                                      int local_x, int x_width)
{
  _Pragma("unroll")
  for(int l_y=START; l_y<END; l_y++)
  {
    int dst_ind=(l_y+ps_y)*(x_width);
    int src_ind=(l_y+ps_y+y_step)*(x_width);
    ptr[dst_ind+local_x]=ptr[src_ind+local_x];
    if(threadIdx.x<halo*2)
        ptr[dst_ind+local_x+blockDim.x]=ptr[src_ind+local_x+blockDim.x];

  }
}

// template<class REAL, int SRC_SIZE, int DST_SIZE, int SIZE,int halo=0>
// __device__ void __forceinline__ reg2reg(REAL src_reg[SRC_SIZE], REAL dst_reg[DST_SIZE],
//                                         int src_basic, int dst_basic)
// {
//   _Pragma("unroll")
//   for(int l_y=0; l_y<SIZE; l_y++)
//   {
//     dst_reg[l_y+dst_basic]=src_reg[l_y+src_basic];
//   }
// }

// template<class REAL, int SRC_SIZE, int DST_SIZE, int SIZE, int halo>
// __device__ void __forceinline__ regs2regs(REAL src_reg[2*halo+1][SRC_SIZE], REAL dst_reg[2*halo+1][DST_SIZE],
//                                         int src_basic, int dst_basic)
// {
//   _Pragma("unroll")
//   for(int l_x=0; l_x<halo*2+1; l_x++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<SIZE; l_y++)
//     {
//       dst_reg[l_x][l_y+dst_basic]=src_reg[l_x][l_y+src_basic];
//     }
//   }
// }

template<class REAL, int SIZE_Z, int SIZE_Y>
__device__ void __forceinline__ regsself3d(
  REAL reg_array[SIZE_Z][SIZE_Y])
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE_Y; l_y++)
  {
    _Pragma("unroll")
    for(int l_z=0; l_z<SIZE_Z ; l_z++)
    { 
      reg_array[l_z][l_y] = reg_array[l_z+1][l_y];
    }
  }
}

template<class REAL, int SIZE_Z, int SIZE_Y, int SIZE_X>
__device__ void __forceinline__ regsself3d(
  REAL reg_array[SIZE_Z][SIZE_Y][SIZE_X])
{
  _Pragma("unroll")
  for(int l_z=0; l_z<SIZE_Z-1 ; l_z++)
  { 
    // int l_z=0; 
    _Pragma("unroll")
    for(int l_y=0; l_y<SIZE_Y; l_y++)
    {
      _Pragma("unroll")
      for(int l_x=0; l_x<SIZE_X ; l_x++)
      { 
        reg_array[l_z][l_y][l_x] = reg_array[l_z+1][l_y][l_x];
      }
    }
  }
}


template<class REAL, int halo, bool isInit=false, bool sync=true>
__device__ void __forceinline__ global2sm(REAL* src, REAL* sm_buffer, 
                                              int size, 
                                              int global_y_base, int global_y_size,
                                              int global_x_base, int global_x_size,
                                              int sm_y_base, int sm_x_base, int sm_width,
                                              int tid)
{
  //fill shared memory buffer
  _Pragma("unroll")
  for(int l_y=0; l_y<size; l_y++)
  {
    int l_global_y;
    if(isInit)
    {
      l_global_y=(MAX(global_y_base+l_y,0));
    }
    else
    {
      l_global_y=(MIN(global_y_base+l_y,global_y_size-1));
      l_global_y=(MAX(l_global_y,0));
    }
  
    #define  dst_ind (l_y+sm_y_base)*sm_width
    
    #ifndef ASYNCSM
      sm_buffer[dst_ind-halo+tid+sm_x_base]=src[l_global_y * global_x_size + MAX(global_x_base-halo+tid,0)];
      if(halo>0)
      {
        if(tid<halo*2)
        {  
          sm_buffer[dst_ind-halo+tid+blockDim.x+sm_x_base]=src[(l_global_y) * global_x_size + MIN(-halo+tid+blockDim.x+global_x_base, global_x_size-1)];
        }
      }
    #else
      #if PERKS_ARCH>=800 
        __pipeline_memcpy_async(sm_buffer+dst_ind-halo+tid+sm_x_base, 
              src + (l_global_y) * global_x_size + MAX(global_x_base-halo+tid,0)
                , sizeof(REAL));
        if(halo>0)
        {
          if(tid<halo*2)
          {
            __pipeline_memcpy_async(sm_buffer+dst_ind-halo+tid+blockDim.x+sm_x_base, 
                    src + (l_global_y) * global_x_size + MIN(-halo+tid+blockDim.x+global_x_base,global_x_size-1)
                      , sizeof(REAL));
          }
        }
        __pipeline_commit();
      #endif
    #endif
  }
  if(sync==true)
  {  
    #ifdef ASYNCSM
      #if PERKS_ARCH>=800 
        __pipeline_wait_prior(0);
      #endif
    #endif
    __syncthreads();
  }
  
  #undef dst_ind
}

template<class REAL, int halo, int BASE_Z, int SIZE_Z, int SMSIZE, int SM_BASE_Z=BASE_Z, bool isInit=false, bool sync=true>
__device__ void __forceinline__ global2sm(REAL *src, REAL* smbuffer_buffer_ptr[SMSIZE],
                                          int gbase_x, int gbase_y, int gbase_z,
                                          int width_x, int width_y, int width_z,
                                          int sm_width_x, int sm_base_x,
                                          int y_start, int y_end , int y_step,int sm_base_y,
                                          int tile_x, int tid_x)
{
  _Pragma("unroll")
  for(int l_z=0; l_z<SIZE_Z; l_z++)
  {
    int l_global_z = (MAX(gbase_z+l_z+BASE_Z,0));
    if(!isInit||halo>=2)
    {
        l_global_z = (MIN(l_global_z,width_z-1));
    }
    // _Pragma("unroll")
    for(int l_y=y_start; l_y<y_end; l_y+=y_step)
    {
      int l_global_y = (MIN(gbase_y+l_y,width_y-1));
        l_global_y = (MAX(l_global_y,0));
        smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
            src[l_global_z*width_x*width_y+l_global_y*width_x+
            MAX((gbase_x+tid_x-halo),0)];
      if(tid_x<halo*2)
      {
          smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + tile_x-halo+sm_base_x]=
              src[l_global_z*width_x*width_y+l_global_y*width_x+
                MIN(gbase_x+tid_x-halo+tile_x,width_x-1)];
      }
    }
  }
  if(sync)
  {
    __syncthreads();
  }
}

template<class REAL, int halo, int BASE_Z, int SIZE_Z, int SMSIZE, int SM_BASE_Z=BASE_Z,  int LOCAL_ITEM_PER_THREAD, int gdim_y, bool isInit=false, bool sync=true>
__device__ void __forceinline__ global2sm(REAL *src, REAL* smbuffer_buffer_ptr[SMSIZE],
                                          int gbase_x, int gbase_y, int gbase_z,
                                          int width_x, int width_y, int width_z,
                                          int sm_width_x, int sm_base_x,
                                          int sm_base_y,
                                          int tile_x, int tid_x, int tid_y)
{
  _Pragma("unroll")
  for(int l_z=0; l_z<SIZE_Z; l_z++)
  {
    int l_global_z;
    if(!isInit)
    {
        l_global_z = (MIN(gbase_z+l_z+BASE_Z,width_z-1));
    }
    else
    {
       l_global_z = (MAX(gbase_z+l_z+BASE_Z,0));
    }

    #pragma unroll
    for(int i=0; i<LOCAL_ITEM_PER_THREAD; i++)
    {
      int l_y=i+tid_y*LOCAL_ITEM_PER_THREAD-halo;
      int l_global_y = (MIN(gbase_y+l_y,width_y-1));
        l_global_y = (MAX(l_global_y,0));
        smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
            src[l_global_z*width_x*width_y+l_global_y*width_x+
            MAX((gbase_x+tid_x-halo),0)];
      if(tid_x<halo*2)
      {
          smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + tile_x-halo+sm_base_x]=
              src[l_global_z*width_x*width_y+l_global_y*width_x+
                MIN(gbase_x+tid_x-halo+tile_x,width_x-1)];
      }
    }
    if(gdim_y>=2*halo)
    {
      // if(tid_y==0)
      {
        // #pragma unroll
        // for(int i=0; i<2*halo; i++)
        if(tid_y<2*halo)
        {
          int l_y=tid_y+gdim_y*LOCAL_ITEM_PER_THREAD-halo;
          int l_global_y = (MIN(gbase_y+l_y,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
                src[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((gbase_x+tid_x-halo),0)];
          if(tid_x<halo*2)
          {
              smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + tile_x-halo+sm_base_x]=
                  src[l_global_z*width_x*width_y+l_global_y*width_x+
                    MIN(gbase_x+tid_x-halo+tile_x,width_x-1)];
          }
        }
      }
    }
    else if(gdim_y>=2)
    {
      if(tid_y<2)
      {
        #pragma unroll
        for(int i=0; i<halo; i++)
        {
          int l_y=i+2*tid_y+gdim_y*LOCAL_ITEM_PER_THREAD-halo;
          int l_global_y = (MIN(gbase_y+l_y,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
                src[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((gbase_x+tid_x-halo),0)];
          if(tid_x<halo*2)
          {
              smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + tile_x-halo+sm_base_x]=
                  src[l_global_z*width_x*width_y+l_global_y*width_x+
                    MIN(gbase_x+tid_x-halo+tile_x,width_x-1)];
          }
        }
      }
    }
    else
    {
      if(tid_y==0)
      {
        #pragma unroll
        for(int i=0; i<2*halo; i++)
        {
          int l_y=i+gdim_y*LOCAL_ITEM_PER_THREAD-halo;
          int l_global_y = (MIN(gbase_y+l_y,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
                src[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((gbase_x+tid_x-halo),0)];
          if(tid_x<halo*2)
          {
              smbuffer_buffer_ptr[l_z+SM_BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + tile_x-halo+sm_base_x]=
                  src[l_global_z*width_x*width_y+l_global_y*width_x+
                    MIN(gbase_x+tid_x-halo+tile_x,width_x-1)];
          }
        }
      }
    }
  }
  if(sync)
  {
    __syncthreads();
  }
}

__device__ void __forceinline__ pipesync()
{
  #ifdef ASYNCSM
    {
      #if PERKS_ARCH>=800 
      __pipeline_commit();
      __pipeline_wait_prior(0);
      #endif
    }
  #else
    __syncthreads();
  #endif
}

//template<class REAL, int SIZE, bool considerbound=true>
template<class REAL, bool considerbound=true>
__device__ void __forceinline__ sm2global(REAL *sm_src, REAL* dst,
                                          int size, 
                                          int global_y_base, int global_y_size,
                                          int global_x_base, int global_x_size,
                                          int sm_y_base, int sm_x_base, int sm_width,
                                          int tid)
{

 // _Pragma("unroll")
  for(int l_y=0; l_y<size; l_y++)
  {
    int global_y=l_y+global_y_base;
    int global_x=tid+global_x_base;
    if(considerbound)
    {
      if(global_y>=global_y_size||global_x>=global_x_size)break;
    }
    dst[(global_y) * global_x_size + global_x] = sm_src[(sm_y_base + l_y) * sm_width + tid + sm_x_base];
  }
}

// template<class REAL, int REG_SIZE, int SIZE, int halo=0>
// __device__ void __forceinline__ sm2reg(REAL* sm_src, REAL reg_dst[SIZE],
//                                       int y_base, 
//                                       int x_base, int x_id,
//                                       int sm_width, 
//                                       int reg_base=0)
// {
//   _Pragma("unroll")
//   for(int l_y=0; l_y<SIZE ; l_y++)
//   {
//     reg_dst[l_y+reg_base] = sm_src[(l_y+y_base)*sm_width+x_base+x_id];//input[(global_y) * width_x + global_x];
//   }
// }

// template<class REAL, int REG_SIZE, int SIZE, int halo>
// __device__ void __forceinline__ sm2regs(REAL* sm_src, REAL reg_dst[2*halo+1][REG_SIZE],
//                                       int y_base, 
//                                       int x_base, int x_id,
//                                       int sm_width, 
//                                       int reg_base=0)
// {
//   _Pragma("unroll")
//   for(int l_x=0; l_x<halo*2+1; l_x++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<SIZE ; l_y++)
//     {
//       reg_dst[l_x][l_y+reg_base] = sm_src[(l_y+y_base)*sm_width+x_base+x_id+l_x-halo];//input[(global_y) * width_x + global_x];
//     }
//   }
// }



template<class REAL, int REG_SIZE_Y, int REG_SIZE_Z, 
            int SM_SIZE_Z, int SM_BASE_Z,
            int BASE_Y=0, int BASE_Z=0, 
            int SIZE_Y=REG_SIZE_Y, int SIZE_Z=REG_SIZE_Z>
__device__ void __forceinline__ sm2regs(REAL* sm_src[SM_SIZE_Z], REAL reg_dst[REG_SIZE_Z][REG_SIZE_Y],
                                      int y_base, 
                                      int x_base, 
                                      int sm_width,
                                      int tid_x)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE_Y; l_y++)
  {
    _Pragma("unroll")
    for(int l_z=0; l_z<SIZE_Z ; l_z++)
    {
      reg_dst[l_z+BASE_Z][l_y+BASE_Y] = sm_src[l_z+SM_BASE_Z][(l_y+y_base)*sm_width+x_base+tid_x];//input[(global_y) * width_x + global_x];
    }
  }


}

template<class REAL, int REG_SIZE, int SIZE>
__device__ void __forceinline__ reg2sm( REAL reg_src[REG_SIZE], REAL* sm_dst,
                                      int sm_y_base, 
                                      int sm_x_base, int tid,
                                      int sm_width,
                                      int reg_base)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE ; l_y++)
  {
    sm_dst[(l_y+sm_y_base)*sm_width + sm_x_base + tid]=reg_src[l_y+reg_base];
  }
}
#endif