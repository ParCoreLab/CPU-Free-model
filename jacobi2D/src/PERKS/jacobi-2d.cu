#ifdef _TIMER_
#include "cuda_profiler_api.h"
#endif
#include <cuda.h>
#include "stdio.h"
#include <cooperative_groups.h>
#include "stdio.h"
#include "assert.h"
#include "config.cuh" 
#include "./common/jacobi_cuda.cuh"
#include "./common/types.hpp"
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"



// #include "./genconfig.cuh"

#define GENWR

//#ifndef REAL
//#define REAL float
//#endif
//configuration
#if defined(GEN)|| defined(GENWR) ||defined(PERSISTENT)
  #define PERSISTENTLAUNCH
#endif
#if defined PERSISTENTLAUNCH||defined(BASELINE_CM)
  #define PERSISTENTTHREAD
#endif
#if defined(BASELINE)||defined(BASELINE_CM)||defined(GEN)||defined(GENWR)||defined(PERSISTENT)
  #define USEMAXSM
#endif


#ifndef RUNS
  #define RUNS (1)
#endif
namespace cg = cooperative_groups;


#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}


__global__ void printptx(int *result)
{
  // printf("code is run in %d\n",PERKS_ARCH);
  result[0]=PERKS_ARCH;
}
void host_printptx(int&result)
{
  int*d_r;
  cudaMalloc((void**)&d_r, sizeof(int));
  printptx<<<1,1>>>(d_r);
  cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_r);
}

#ifndef RTILE_Y
#define RTILE_Y (8)
#endif

template<int halo, bool isstar, int registeramount, int arch, bool useSM, class REAL, int LOCAL_RTILE_Y=RTILE_Y>
int getMinWidthY(int width_x, int bdimx, int blkpsm)
{
  //firstly be able  to get the register info
  int ptx;
  host_printptx(ptx);
  bdimx=((bdimx==128)?128:256);

  int smcount=0;
  if(ptx==700)
  {
    smcount=80;
  }
  if(ptx==800)
  {
    smcount=108;
  }
  int registerfoler=0;
  int max_sm_flder=0;
  // int total_flder=0;
  // int blkpsm=256/registeramount;
  {
    {
      
      if(useSM)
      {
        registerfoler=regfolder<halo,isstar,registeramount,arch,true,REAL,LOCAL_RTILE_Y>::val;

        int maxSharedMemory;
        cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
        int SharedMemoryUsed = maxSharedMemory-1024;

        int basic_sm_space=(LOCAL_RTILE_Y+2*HALO)*(bdimx+2*HALO)+1;
        // printf("%d\n",SharedMemoryUsed/sizeof(REAL)/blkpsm);
        // printf("%d\n",2*HALO*isBOX);
        // printf("%d\n",basic_sm_space);
        // printf("%d\n",2*HALO*(registerfoler)*LOCAL_RTILE_Y);
        // printf("%d\n",2*HALO*(bdimx+2*HALO));
        int tmp0=SharedMemoryUsed/sizeof(REAL)/blkpsm;
        int tmp1=2*HALO*isBOX;
        int tmp2=basic_sm_space;
        int tmp3=2*HALO*(registerfoler)*LOCAL_RTILE_Y;
        int tmp4=2*HALO*(bdimx+2*HALO);
        tmp0=tmp0-tmp1-tmp2-tmp3-tmp4;
        tmp0=tmp0>0?tmp0:0;
        max_sm_flder=(tmp0)/(bdimx+4*HALO)/LOCAL_RTILE_Y;
        // printf("%d,%d\n",registerfoler,max_sm_flder);
      }
      else
      {
        registerfoler=regfolder<halo,isstar,registeramount,arch,false,REAL,LOCAL_RTILE_Y>::val;
      }
    }
  }
  int dimy=smcount*blkpsm/(width_x/bdimx);
  // printf("<%d,%d>\n",registerfoler,dimy);
  return (registerfoler+max_sm_flder)*LOCAL_RTILE_Y*dimy;
}
template<class REAL>
int getMinWidthY(int width_x, int bdimx, int registers, bool useSM, int blkpsm, bool isDoubleTile)
{
  const int  LOCAL_RTILE_Y=isDoubleTile?RTILE_Y*2: RTILE_Y;

  int ptx;
  host_printptx(ptx);
  bdimx=((bdimx==128)?128:256);

#ifdef GEN
  registers=0;
#endif
  if(blkpsm!=1||registers==0)
  {
    #if defined(PERSISTENT)

      #ifndef BOX
      auto execute_kernel = isDoubleTile?kernel_persistent_baseline<REAL,RTILE_Y*2,HALO>
                  :kernel_persistent_baseline<REAL,RTILE_Y,HALO>;
      #else
      auto execute_kernel = isDoubleTile?kernel_persistent_baseline_box<REAL,RTILE_Y*2,HALO>
                  :kernel_persistent_baseline_box<REAL,RTILE_Y,HALO>;
      #endif
      return 0;
    #endif
    #if defined(BASELINE_CM)||defined(BASELINE)
      #ifndef BOX
        auto execute_kernel = isDoubleTile?kernel_baseline<REAL,RTILE_Y*2,HALO>
                  :kernel_baseline<REAL,RTILE_Y,HALO>;
      #else
        auto execute_kernel = isDoubleTile?kernel_baseline_box<REAL,RTILE_Y*2,HALO>
                  :kernel_baseline_box<REAL,RTILE_Y,HALO>;
      #endif
      return 0;
    #endif
    #if defined(NAIVE)||defined(NAIVENVCC)
      #ifndef BOX
        auto execute_kernel = kernel2d_restrict<REAL,HALO>;
      #else
        auto execute_kernel = kernel2d_restrict_box<REAL,HALO>;
      #endif
      return 0;
    #endif 
    #ifdef GEN
      #ifndef BOX
        auto execute_kernel = isDoubleTile?
              (useSM?kernel_general<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
                kernel_general<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,false>)
              :(useSM?kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
                kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,false>);
      #else
        auto execute_kernel = isDoubleTile?
              (useSM?kernel_general_box<REAL,RTILE_Y*2,HALO,REG_FOLDER_Y,1,true>:
                kernel_general_box<REAL,RTILE_Y*2,HALO,REG_FOLDER_Y,1,false>)
              :(useSM?kernel_general_box<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
                kernel_general_box<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,false>)
              ;
      #endif
    #endif

    #ifdef GENWR
      auto execute_kernel = isDoubleTile?
                    (blkpsm*bdimx>=2*256?
                    (useSM?kernel_general_wrapper<REAL,2*RTILE_Y,HALO,128,true>:
                        kernel_general_wrapper<REAL,2*RTILE_Y,HALO,128,false>)
                    :
                    (useSM?kernel_general_wrapper<REAL,2*RTILE_Y,HALO,256,true>:
                        kernel_general_wrapper<REAL,2*RTILE_Y,HALO,256,false>))
                    :(blkpsm*bdimx>=2*256?
                    (useSM?kernel_general_wrapper<REAL,RTILE_Y,HALO,128,true>:
                        kernel_general_wrapper<REAL,RTILE_Y,HALO,128,false>)
                    :
                    (useSM?kernel_general_wrapper<REAL,RTILE_Y,HALO,256,true>:
                        kernel_general_wrapper<REAL,RTILE_Y,HALO,256,false>))
                    ; 

      // auto execute_kernel = kernel_general_wrapper<float,RTILE_Y,HALO,256,true>;
    #endif
    int basic_sm_space=(LOCAL_RTILE_Y+2*HALO)*(bdimx+2*HALO)+1;
    size_t sharememory_basic=(basic_sm_space)*sizeof(REAL);
    size_t executeSM = sharememory_basic;
    int getblkpsm=100;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &getblkpsm, execute_kernel, bdimx, executeSM);
    if(blkpsm<=0)blkpsm=getblkpsm;
    else blkpsm=min(blkpsm,getblkpsm);
    // blkpsm=1;
  }

  if(ptx==800)
  {
    if(registers==0)
    {
      if(useSM)
      {
        return isDoubleTile?getMinWidthY<HALO,isStar,0,800,true,REAL,RTILE_Y*2>(width_x,bdimx,blkpsm)
         :getMinWidthY<HALO,isStar,0,800,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return 0;
        // return getMinWidthY<HALO,isStar,0,800,false,REAL>(width_x,bdimx,blkpsm);
      }
    }
    if(registers==128)
    {
      if(useSM)
      {
        return isDoubleTile?getMinWidthY<HALO,isStar,128,800,true,REAL,RTILE_Y*2>(width_x,bdimx,blkpsm)
                    :getMinWidthY<HALO,isStar,128,800,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return isDoubleTile?getMinWidthY<HALO,isStar,128,800,false,REAL,RTILE_Y*2>(width_x,bdimx,blkpsm):
        getMinWidthY<HALO,isStar,128,800,false,REAL>(width_x,bdimx,blkpsm);
      }
    }
    if(registers==256)
    {
      if(useSM)
      {
        return isDoubleTile? getMinWidthY<HALO,isStar,256,800,true,REAL,RTILE_Y*2>(width_x,bdimx,blkpsm)
        :getMinWidthY<HALO,isStar,256,800,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return isDoubleTile? getMinWidthY<HALO,isStar,256,800,false,REAL,RTILE_Y*2>(width_x,bdimx,blkpsm)
        :getMinWidthY<HALO,isStar,256,800,false,REAL>(width_x,bdimx,blkpsm);;
      }
    }
  }

  if(ptx==700)
  {
    if(registers==0)
    {
      if(useSM)
      {
        return isDoubleTile?getMinWidthY<HALO,isStar,0,700,true,REAL,2*RTILE_Y>(width_x,bdimx,blkpsm)
        : getMinWidthY<HALO,isStar,0,700,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return 0;
        // return getMinWidthY<HALO,isStar,0,700,false,REAL>(width_x,bdimx,blkpsm);
      }
    }
    if(registers==128)
    {
      if(useSM)
      {
        return isDoubleTile? getMinWidthY<HALO,isStar,128,700,true,REAL,2*RTILE_Y>(width_x,bdimx,blkpsm)
          :getMinWidthY<HALO,isStar,128,700,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return isDoubleTile?getMinWidthY<HALO,isStar,128,700,false,REAL,2*RTILE_Y>(width_x,bdimx,blkpsm)
            : getMinWidthY<HALO,isStar,128,700,false,REAL>(width_x,bdimx,blkpsm);
      }
    }
    if(registers==256)
    {
      if(useSM)
      {
        return isDoubleTile? getMinWidthY<HALO,isStar,256,700,true,REAL,2*RTILE_Y>(width_x,bdimx,blkpsm)
          :getMinWidthY<HALO,isStar,256,700,true,REAL>(width_x,bdimx,blkpsm);
      }
      else
      {
        return isDoubleTile? getMinWidthY<HALO,isStar,256,700,false,REAL,2*RTILE_Y>(width_x,bdimx,blkpsm)
          :getMinWidthY<HALO,isStar,256,700,false,REAL>(width_x,bdimx,blkpsm);
      }
    }
  }
  return 0;
}


template<int halo, bool isstar, int arch, class REAL>
int getMinWidthY(int width_x, int bdimx, bool isDoubleTile)
{
  if(isDoubleTile)
  {
    int minwidthy1 = getMinWidthY<halo,isstar,128,arch,false,REAL,RTILE_Y*2>(width_x,bdimx,2);
    int minwidthy2 = getMinWidthY<halo,isstar,256,arch,false,REAL,RTILE_Y*2>(width_x,bdimx,1);
    int minwidthy3 = getMinWidthY<halo,isstar,128,arch,true,REAL,RTILE_Y*2>(width_x,bdimx,2);
    int minwidthy4 = getMinWidthY<halo,isstar,256,arch,true,REAL,RTILE_Y*2>(width_x,bdimx,1);

    int result = max(minwidthy1,minwidthy2);
    result = max(result,minwidthy3);
    result = max(result,minwidthy4);
    return result;
  }
  else
  {
    int minwidthy1 = getMinWidthY<halo,isstar,128,arch,false,REAL>(width_x,bdimx,2);
    int minwidthy2 = getMinWidthY<halo,isstar,256,arch,false,REAL>(width_x,bdimx,1);
    int minwidthy3 = getMinWidthY<halo,isstar,128,arch,true,REAL>(width_x,bdimx,2);
    int minwidthy4 = getMinWidthY<halo,isstar,256,arch,true,REAL>(width_x,bdimx,1);

    int result = max(minwidthy1,minwidthy2);
    result = max(result,minwidthy3);
    result = max(result,minwidthy4);
    return result;
  }

}

template<class REAL>
int getMinWidthY(int width_x, int bdimx,bool isDoubleTile)
{
  int ptx;
  host_printptx(ptx);
  if(ptx==800)
  {
    return getMinWidthY<HALO,isStar,800, REAL>(width_x,bdimx,isDoubleTile);
  }
  if(ptx==700)
  {
    return getMinWidthY<HALO,isStar,700, REAL>(width_x,bdimx,isDoubleTile);
  }  
  return 0;
}

template int getMinWidthY<float>  (int, int, int, bool,int, bool);
template int getMinWidthY<double> (int, int, int, bool,int, bool);
template int getMinWidthY<float>  (int, int,bool);
template int getMinWidthY<double> (int, int,bool);




template<class REAL>
// void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration, bool async=false){
int jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, 
  int bdimx, int blkpsm, int iteration, bool async, bool useSM, 
  bool usewarmup, int warmupiteration,
  bool isDoubleTile){
// extern "C" void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration){
/* Host allocation Begin */
 // #define LOCAL_RTILE_Y (RTILE_Y)
  bdimx=((bdimx==128)?128:256);
#ifndef NAIVE
  const int LOCAL_RTILE_Y = isDoubleTile?RTILE_Y*2:RTILE_Y;
#endif
  int ptx;
  host_printptx(ptx);
  if(blkpsm<=0)blkpsm=100;
#ifdef GENWR
int REG_FOLDER_Y=0;
if(blkpsm*bdimx>=2*256)
{
  if(useSM)
  {
    if(ptx==800)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,128,800,true,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,128,800,true,REAL>::val);
    if(ptx==700)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,128,700,true,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,128,700,true,REAL>::val);
  }
  else
  {
    if(ptx==800)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,128,800,false,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,128,800,false,REAL>::val);
    if(ptx==700)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,128,700,false,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,128,700,false,REAL>::val);
  }
}
else
{
  if(useSM)
  {
    if(ptx==800)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,256,800,true,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,256,800,true,REAL>::val);
    if(ptx==700)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,256,700,true,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,256,700,true,REAL>::val);
  }
  else
  {
    if(ptx==800)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,256,800,false,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,256,800,false,REAL>::val);
    if(ptx==700)REG_FOLDER_Y=isDoubleTile?(regfolder<HALO,isStar,256,700,false,REAL,2*RTILE_Y>::val):(regfolder<HALO,isStar,256,700,false,REAL>::val);
  }
}

#endif



  #ifndef __PRINT__
    printf("code is run in %d\n",ptx);
  #endif
/*************************************/
  // if(ptx<800&&async==true)
  // {
  //   printf("error async not support\n");//lower ptw not support 
  //   return -1;
  // }
//initialization
#if defined(PERSISTENT)

  #ifndef BOX
  // auto execute_kernel = async?kernel_persistent_baseline_async<REAL,LOCAL_RTILE_Y,HALO>: kernel_persistent_baseline<REAL,LOCAL_RTILE_Y,HALO>;
  auto execute_kernel = isDoubleTile? kernel_persistent_baseline<REAL,2*RTILE_Y,HALO>
                                :kernel_persistent_baseline<REAL,RTILE_Y,HALO>;
  #else
  // auto execute_kernel = async?kernel_persistent_baseline_box_async<REAL,LOCAL_RTILE_Y,HALO>:kernel_persistent_baseline_box<REAL,LOCAL_RTILE_Y,HALO>;
  auto execute_kernel = isDoubleTile? kernel_persistent_baseline_box<REAL,RTILE_Y*2,HALO>
                                : kernel_persistent_baseline_box<REAL,RTILE_Y,HALO>;
  #endif
#endif
#if defined(BASELINE_CM)||defined(BASELINE)
  #ifndef BOX
    // auto execute_kernel = async?kernel_baseline_async<REAL,LOCAL_RTILE_Y,HALO>:kernel_baseline<REAL,LOCAL_RTILE_Y,HALO>;
    auto execute_kernel = isDoubleTile?kernel_baseline<REAL,2*RTILE_Y,HALO>
                                : kernel_baseline<REAL,RTILE_Y,HALO>;
  #else
    // auto execute_kernel = async?kernel_baseline_box_async<REAL,LOCAL_RTILE_Y,HALO>:kernel_baseline_box<REAL,LOCAL_RTILE_Y,HALO>;
    auto execute_kernel = isDoubleTile? kernel_baseline_box<REAL,2*RTILE_Y,HALO>
                                :kernel_baseline_box<REAL,RTILE_Y,HALO>;
  #endif
#endif
#ifdef NAIVE
  #ifndef BOX
    auto execute_kernel = kernel2d_restrict<REAL,HALO>;
  #else
    auto execute_kernel = kernel2d_restrict_box<REAL,HALO>;
  #endif
#endif 
#ifdef GEN
  #ifndef BOX

    auto execute_kernel = isDoubleTile? 
            (useSM?kernel_general<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
            kernel_general<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,false>)
            :(useSM?kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
            kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,false>);
  #else

    auto execute_kernel = isDoubleTile?
            (useSM?kernel_general_box<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
              kernel_general_box<REAL,2*RTILE_Y,HALO,REG_FOLDER_Y,1,false>)
            :(useSM?kernel_general_box<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,true>:
              kernel_general_box<REAL,RTILE_Y,HALO,REG_FOLDER_Y,1,false>)
          ;
  #endif
#endif

#ifdef GENWR
  auto execute_kernel = isDoubleTile?
                (blkpsm*bdimx>=2*256?
                (useSM?kernel_general_wrapper<REAL,2*RTILE_Y,HALO,128,true>:
                    kernel_general_wrapper<REAL,2*RTILE_Y,HALO,128,false>)
                :
                (useSM?kernel_general_wrapper<REAL,2*RTILE_Y,HALO,256,true>:
                    kernel_general_wrapper<REAL,2*RTILE_Y,HALO,256,false>))
                :(blkpsm*bdimx>=2*256?
                (useSM?kernel_general_wrapper<REAL,RTILE_Y,HALO,128,true>:
                    kernel_general_wrapper<REAL,RTILE_Y,HALO,128,false>)
                :
                (useSM?kernel_general_wrapper<REAL,RTILE_Y,HALO,256,true>:
                    kernel_general_wrapper<REAL,RTILE_Y,HALO,256,false>))
                ; 
          // auto execute_kernel = kernel_general_wrapper<float,RTILE_Y,HALO,256,true>;

#endif

  //auto execute_kernel = kernel_general<REAL,LOCAL_RTILE_Y,HALO,REG_FOLDER_Y,UseSMCache>;

  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );

  //initialization input and output space
  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : input\n");
  cudaMemcpy(input,h_input,sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyHostToDevice);
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

  //initialize tmp space for halo region
#if defined(GEN) || defined(GENWR)|| defined(PERSISTENT)
  REAL * L2_cache3;
  REAL * L2_cache4;
  size_t L2_utage_2 = sizeof(REAL)*(width_y)*2*(width_x/bdimx)*HALO;
#ifndef __PRINT__
  printf("l2 cache used is %ld KB : 4096 KB \n",L2_utage_2*2/1024);
#endif
  cudaMalloc(&L2_cache3,L2_utage_2*2);
  L2_cache4=L2_cache3+(width_y)*2*(width_x/bdimx)*HALO;
#endif

  //initialize shared memory
  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  //could not use all share memory in a100. so set it in default.

#if defined(USEMAXSM)
  int SharedMemoryUsed = maxSharedMemory-1024;
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif

size_t executeSM = 0;
#ifndef NAIVE
  //shared memory used for compuation
  int basic_sm_space=(LOCAL_RTILE_Y+2*HALO)*(bdimx+2*HALO)+1;
  size_t sharememory_basic=(basic_sm_space)*sizeof(REAL);
  executeSM = sharememory_basic;
  {
    #if defined(GEN) || defined(GENWR)  
    #define halo HALO
    executeSM +=  (HALO*2*(( REG_FOLDER_Y)*LOCAL_RTILE_Y+isBOX))*sizeof(REAL);
    #undef halo
    #endif
  }
#endif


#ifdef PERSISTENTTHREAD
  int numBlocksPerSm_current=1000;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdimx, executeSM);
  cudaDeviceSynchronize();
  cudaCheckError();
  #ifndef __PRINT__
    // printf("");
    printf("blk per sm is %d/%d %d in sm %f\n", numBlocksPerSm_current,blkpsm,bdimx,(double)executeSM/1024);
  #endif
  // int smbound=SharedMemoryUsed/executeSM;
  // printf("%d,%d,%d\n",numBlocksPerSm_current,blkpsm,smbound);
  if(blkpsm!=0)
  {
    
    numBlocksPerSm_current=min(numBlocksPerSm_current,blkpsm);
  }
  dim3 block_dim(bdimx);
  dim3 grid_dim(width_x/bdimx,sm_count*numBlocksPerSm_current/(width_x/bdimx));
  
  dim3 executeBlockDim=block_dim;
  dim3 executeGridDim=grid_dim;

  #ifndef __PRINT__
    printf("blk per sm is %d/%d\n", numBlocksPerSm_current,blkpsm);
    printf("grid is (%d,%d)\n", grid_dim.x, grid_dim.y);
  #endif

#endif 

  #ifdef PERSISTENT
    size_t max_sm_flder=0;
  #endif

  #define halo HALO
  #if defined(GEN) || defined(GENWR)  

  size_t max_sm_flder=0;
  int tmp0=SharedMemoryUsed/sizeof(REAL)/numBlocksPerSm_current;
  int tmp1=2*HALO*isBOX;
  int tmp2=basic_sm_space;
  int tmp3=2*HALO*(REG_FOLDER_Y)*LOCAL_RTILE_Y;
  int tmp4=2*HALO*(bdimx+2*HALO);
  tmp0=tmp0-tmp1-tmp2-tmp3-tmp4;
  tmp0=tmp0>0?tmp0:0;
  max_sm_flder=(tmp0)/(bdimx+4*HALO)/LOCAL_RTILE_Y;
  // printf("smflder is %d\n",max_sm_flder);
  if(!useSM)max_sm_flder=0;
  if(useSM&&max_sm_flder==0)return 1;

  size_t sm_cache_size =max_sm_flder==0?0: (max_sm_flder*LOCAL_RTILE_Y+2*HALO)*(bdimx+2*HALO)*sizeof(REAL);
  size_t y_axle_halo = (HALO*2*((max_sm_flder + REG_FOLDER_Y)*LOCAL_RTILE_Y+isBOX))*sizeof(REAL);
  executeSM=sharememory_basic+y_axle_halo;
  executeSM+=sm_cache_size;

  #undef halo

  #ifndef __PRINT__
    printf("the max flder is %ld and the total sm size is %ld/block\n", max_sm_flder, executeSM);
  #endif

  #endif


#ifdef NAIVE
  dim3 block_dim_1(MIN(width_x,bdimx),1);
  dim3 grid_dim_1(width_x/MIN(width_x,bdimx),width_y/1);

  dim3 executeBlockDim=block_dim_1;
  dim3 executeGridDim=grid_dim_1;
#endif
#ifdef BASELINE
  dim3 block_dim2(bdimx);
  dim3 grid_dim2(width_x/bdimx,MIN((sm_count*8*1024/bdimx)/(width_x/bdimx),width_y/LOCAL_RTILE_Y));
//  printf("<%d,%d,%d>",); 
  dim3 executeBlockDim=block_dim2;
  dim3 executeGridDim=grid_dim2;

#endif
//in order to get a better performance, warmup run is necessary.

// #ifdef WARMUPRUN

  int l_warmupiteration=warmupiteration>0?warmupiteration:1000;
  // printf("%d\n",l_warmupiteration);
// #endif

#if defined(GEN)|| defined(GENWR) || defined(PERSISTENT)
    void* KernelArgs_NULL[] ={(void**)&__var_2__,(void**)&width_y,
      (void*)&width_x,(void*)&__var_1__,(void*)&L2_cache3,(void*)&L2_cache4,
      (void*)&l_warmupiteration, (void *)&max_sm_flder};
    
  // #endif
#endif

if(usewarmup){
  // cudaCheckError();

  // cudaEventRecord(warstart,0);
  cudaEvent_t warstart,warmstop;
  cudaEventCreate(&warstart);
  cudaEventCreate(&warmstop);
  #ifdef TRADITIONLAUNCH
  {


      cudaEventRecord(warstart,0);
      for(int i=0; i<l_warmupiteration; i++)
      {
        execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
              (__var_2__, width_y, width_x , __var_1__);
        REAL* tmp = __var_2__;
        __var_2__=__var_1__;
        __var_1__= tmp;
      } 
      cudaEventRecord(warmstop,0);
      cudaEventSynchronize(warmstop);
      float warmelapsedTime;
      cudaEventElapsedTime(&warmelapsedTime,warstart,warmstop);
      int nowwarmup=warmelapsedTime;
      int nowiter=(350+nowwarmup-1)/nowwarmup;

      for(int out=0; out<nowiter; out++)
      {
        for(int i=0; i<l_warmupiteration; i++)
        {
          execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
                (__var_2__, width_y, width_x , __var_1__);
          REAL* tmp = __var_2__;
          __var_2__=__var_1__;
          __var_1__= tmp;
        }       
      }
  }
  #endif 
  
  #ifdef PERSISTENTLAUNCH
  {
      // double accumulate=0;
      cudaEventRecord(warstart,0);
      cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgs_NULL, executeSM,0);
      cudaEventRecord(warmstop,0);
      cudaEventSynchronize(warmstop);
      float warmelapsedTime;
      cudaEventElapsedTime(&warmelapsedTime,warstart,warmstop);
      int nowwarmup=warmelapsedTime;
      int nowiter=(350+nowwarmup-1)/nowwarmup;
      for(int i=0; i<nowiter; i++)
      {
        cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgs_NULL, executeSM,0);
      }
  }
  #endif
  // cudaEventRecord(warmstop,0);

  // float warmelapsedTime;
  // cudaEventElapsedTime(&warmelapsedTime,warstart,warmstop);
  // printf("run times %f\n",warmelapsedTime);

}
// printf("herer");


//@NEO: l2 cache setting not show performance difference. 
#if defined(GEN) && defined(L2PER)
    // REAL l2perused;
    // size_t inner_window_size = 30*1024*1024;
    // cudaStreamAttrValue stream_attribute;
    // stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(L2_cache3);                  // Global Memory data pointer
    // stream_attribute.accessPolicyWindow.num_bytes = min(inner_window_size,L2_utage_2*2);                                   // Number of bytes for persistence access
    // stream_attribute.accessPolicyWindow.hitRatio  = 1;                                             // Hint for cache hit ratio
    // stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;                  // Persistence Property
    // stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

    // cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
    // cudaCtxResetPersistingL2Cache();
    // cudaStreamSynchronize(0);
#endif

#if defined(GEN)|| defined(GENWR) || defined(PERSISTENT)
  int l_iteration=iteration;
  void* ExecuteKernelArgs[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache3,(void*)&L2_cache4,
    (void*)&l_iteration, (void*)&max_sm_flder};
#endif

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
#endif 

// #ifdef WARMUPRUN


// #endif 


#ifdef _TIMER_
  cudaEventRecord(_forma_timer_start_,0);
#endif

#ifdef PERSISTENTLAUNCH
  for(int i=0; i<RUNS; i++)
  {
    cudaLaunchCooperativeKernel((void*)execute_kernel, 
            executeGridDim, executeBlockDim, 
            ExecuteKernelArgs, 
            //KernelArgs4,
            executeSM,0);
  }
#endif
#ifdef TRADITIONLAUNCH
  for(int i=0; i<RUNS; i++)
  {
    execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
            (input, width_y, width_x, __var_2__);

    for(int i=1; i<iteration; i++)
    {
       execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
            (__var_2__, width_y, width_x , __var_1__);
      REAL* tmp = __var_2__;
      __var_2__=__var_1__;
      __var_1__= tmp;
    }
  }
#endif


#ifdef CHECK
  cudaDeviceSynchronize();
  cudaCheckError();
#endif

#ifndef __PRINT__  
  printf("sm_count is %d\n", sm_count);
  printf("MAX shared memory is %f KB but only use %f KB\n", maxSharedMemory/1024.0,SharedMemoryUsed/1024.0);
  printf(" shared meomory size is %ld KB\n", executeSM/1024);

#endif


#ifdef __PRINT__
  printf("%d\t%d\t%d\t",ptx,sizeof(REAL)/4,(int)async);
  printf("%d\t%d\t%d\t",width_x,width_y,iteration);
  printf("%d\t<%d,%d>\t%d\t",executeBlockDim.x,
        executeGridDim.x,executeGridDim.y,
        (executeGridDim.x)*(executeGridDim.y)/sm_count);
  #ifndef NAIVE
  printf("%f\t",(double)sharememory_basic/1024);
  #endif
#endif

#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  #ifdef __PRINT__
    printf("%f\t%f\t",elapsedTime,(REAL)iteration*(width_y-2*HALO)*(width_x-2*HALO)/ (elapsedTime/RUNS)/1000/1000);
    #if defined(GEN) || defined(GENWR)
    printf("%d\t%d\t%d\t%d\t",REG_FOLDER_Y,max_sm_flder,(int)bdimx*grid_dim.x,(int)(max_sm_flder+REG_FOLDER_Y)*LOCAL_RTILE_Y*grid_dim.y);
    #endif
    #ifndef NAIVE
      printf("%d\t",LOCAL_RTILE_Y);
    #endif
    printf("\n");
  #else
    printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime/RUNS);
    printf("[FORMA] Speed(GCells/s) : %lf\n",(REAL)iteration*(width_y)*(width_x)/ elapsedTime/1000/1000*RUNS);
    printf("[FORMA] Speed(GFLOPS/s) : %lf\n", (REAL)17*iteration*(width_y)*(width_x)/ elapsedTime/1000/1000*RUNS);
    printf("[FORMA] bandwidth(GB/s) : %lf\n", (REAL)sizeof(REAL)*iteration*((width_y)*(width_x)+width_x*width_y)/ elapsedTime/1000/1000*RUNS);
    printf("[FORMA] width_x:width_y=%d:%d\n",(int)width_x, (int)width_y);
    printf("[FORMA] gdimx:gdimy=%d:%d\n",(int)executeGridDim.x, (int)executeGridDim.y);
    #if defined(GEN) || defined(GENWR)
      printf("[FORMA] cached width_x:width_y=%d:%d\n",(int)bdimx*grid_dim.x, (int)(max_sm_flder+REG_FOLDER_Y)*LOCAL_RTILE_Y*grid_dim.y);
      printf("[FORMA] cached b:sf:rf=%d:%d:%d\n", (int)LOCAL_RTILE_Y, (int)max_sm_flder, (int)REG_FOLDER_Y);
    #endif
  #endif

  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif


//finalization
#ifdef CHECK
  // printf("check error here*\n");
  cudaDeviceSynchronize();
  cudaCheckError();
#endif
  cudaDeviceSynchronize();
  cudaCheckError();
#if defined(GEN) || defined(GENWR) || defined(PERSISTENT)
  if(iteration%2==1)
    cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(__var_0__,input, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#else
  cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#endif
/*Kernel Launch End */
/* Host Free Begin */
  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);

#if defined(GEN) || defined(GENWR)  || defined(PERSISTENT)
  cudaFree(L2_cache3);
#endif
  return 0;
  #undef LOCAL_RTILE_Y
}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_ITERATIVE);


