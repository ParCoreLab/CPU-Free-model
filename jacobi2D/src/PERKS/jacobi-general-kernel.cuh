// #ifndef CONFIGURE
//   #include "./config.cuh"
//   #include "./genconfig.cuh"
// #endif
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/types.hpp"
#include <math.h>

#include <cooperative_groups.h>

#ifdef SMASYNC
  // #if PERKS_ARCH<800 
    // #error "unsupport architecture"
  // #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif

namespace cg = cooperative_groups;


template<class REAL, int LOCAL_TILE_Y, int halo, int reg_folder_y, bool UseSMCache>
__device__ __forceinline__ void inner_general
(REAL *  input, int width_y, int width_x, 
  REAL *  __var_4__, 
  REAL *  l2_cache_o,REAL *  l2_cache_i,
  int iteration,
  int max_sm_flder)
{
  if(!UseSMCache) max_sm_flder=0;
  #define UseRegCache (reg_folder_y!=0)
  #ifdef BOX
    #define SM2REG sm2regs
    #define REG2REG regs2regs
  #else
    #define SM2REG sm2reg
    #define REG2REG reg2reg
  #endif
  stencilParaT;
  //basic pointer
  cg::grid_group gg = cg::this_grid();
  //extern __shared__ REAL sm[];
  extern __shared__ char sm[];


  const int total_sm_tile_y = LOCAL_TILE_Y*max_sm_flder;//SM_FOLER_Y;//consider how to automatically compute it later
  const int total_reg_tile_y = LOCAL_TILE_Y*reg_folder_y;
  const int total_tile_y = total_sm_tile_y+total_reg_tile_y;
  const int total_reg_tile_y_with_halo = total_reg_tile_y+2*halo;

  const int sizeof_rspace = total_reg_tile_y_with_halo;
  const int sizeof_rbuffer = LOCAL_TILE_Y+2*halo;

  const int tile_x = blockDim.x;
  const int tile_x_with_halo = tile_x + 2*halo;
  const int tile_y_with_halo = LOCAL_TILE_Y+2*halo;
  const int basic_sm_space=tile_x_with_halo*tile_y_with_halo;

  const int boundary_line_size = total_tile_y+isBOX;
  const int e_step = 0;
  const int w_step = boundary_line_size*halo;

  REAL* sm_rbuffer =(REAL*)sm+1;

  REAL* boundary_buffer = sm_rbuffer + basic_sm_space;
  REAL* sm_space = boundary_buffer+(2*halo*boundary_line_size);//BOX need add additional stuffs. 


  //boundary space
  //register buffer space
  //seems use much space than necessary when no use register version. 
  register REAL r_space[total_reg_tile_y_with_halo];
#ifndef BOX
  register REAL r_smbuffer[2*halo+LOCAL_TILE_Y];
#else
  register REAL r_smbuffer[2*halo+1][2*halo+LOCAL_TILE_Y];
#endif

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = halo;
  const int ps_x = halo;
 // const int tile_x_with_halo = blockDim.x + 2*halo;

  const int p_x = blockIdx.x * tile_x ;

  int blocksize_y=(width_y/gridDim.y);
  int y_quotient = width_y%gridDim.y;
  
  const int p_y =  blockIdx.y * (blocksize_y) + (blockIdx.y<=y_quotient?blockIdx.y:y_quotient);
  blocksize_y += (blockIdx.y<y_quotient?1:0);
  const int p_y_cache_end = p_y + total_reg_tile_y + total_sm_tile_y;
  const int p_y_end = p_y + (blocksize_y);

  //load data global to register
  // #pragma unroll
  if(UseRegCache)
  {
    global2reg<REAL,sizeof_rspace,total_reg_tile_y>(input, r_space,
                                              p_y, width_y,
                                              p_x+tid, width_x,
                                              halo);
  }
  // load data global to sm
  if(UseSMCache)
  {
    global2sm<REAL,0>(input,sm_space,
                                        total_sm_tile_y,
                                        p_y+total_reg_tile_y, width_y,
                                        p_x, width_x,
                                        ps_y, ps_x, tile_x_with_halo,
                                        tid);
  }
  //load ew boundary
  if(UseRegCache||UseSMCache)
  {
    for(int local_y=tid; local_y<boundary_line_size; local_y+=blockDim.x)
    {
      for(int l_x=0; l_x<halo; l_x++)
      {
        //east
        int global_x = p_x + tile_x + l_x;
        global_x = MIN(width_x-1,global_x);
        boundary_buffer[e_step+local_y + l_x*boundary_line_size] = input[MIN((p_y + local_y),width_y-1) * width_x + global_x];
        //west
        global_x = p_x - halo + l_x;
        global_x = MAX(0,global_x);
        boundary_buffer[w_step+local_y + l_x*boundary_line_size] =  input[MIN((p_y + local_y),width_y-1) * width_x + global_x];
      }
    }
    // sdfa
  }
  __syncthreads();
 
  for(int iter=0; iter<iteration; iter++)
  {
    int local_x=tid;
    //prefetch the boundary data
    //north south
    {
      //register
      if(UseRegCache||UseSMCache)
      {
        // #pragma unroll
        if(UseSMCache)
        {
          //need to deal with boundary
          // global2sm<REAL,isBOX,true,true>(input,sm_space,
          global2sm<REAL,isBOX,false,true>(input,sm_space,
                                      halo,
                                      p_y+total_reg_tile_y+total_sm_tile_y, width_y,
                                      p_x, width_x,
                                      ps_y+total_sm_tile_y, ps_x, tile_x_with_halo,
                                      tid);
        }
        else
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<halo; l_y++)
          {
            int global_y = (p_y-halo+l_y);
            global_y=MAX(0,global_y);

            global_y=(p_y+(total_sm_tile_y+total_reg_tile_y)+l_y);
            global_y=MIN(global_y,width_y-1);
            //north
            
            //need to deal with boundary
            r_space[total_reg_tile_y+halo+l_y]=(input[(global_y) * width_x + p_x + tid]);
          }
        }
        if(UseRegCache && UseSMCache)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<halo; l_y++)
          {
            //north of register
            r_space[total_reg_tile_y+halo+l_y]=sm_space[(ps_y+l_y) * tile_x_with_halo + tid + ps_x];
            //south of sm
          }
        }
      }
    }
    // #ifndef SMALL
    //south
    global2sm<REAL,halo,ISINITI,SYNC>(input, sm_rbuffer, 
                                            halo*2,
                                            p_y-halo, width_y,
                                            p_x, width_x,
                                            ps_y-halo, ps_x, tile_x_with_halo,
                                            tid);
    if(UseRegCache)
    {
      // #ifndef BOX
      sm2reg<REAL,sizeof_rspace, halo*2,isBOX>(sm_rbuffer, r_space, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);
    
      
      #ifdef BOX
        /* code */
        SM2REG<REAL,sizeof_rbuffer, halo*2,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);    
      #endif
    }
    else
    {
          
      SM2REG<REAL,sizeof_rbuffer, halo*2,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);    
    }

    // #endif
    __syncthreads();
    //computation of register space
    if(UseRegCache)
    {
      _Pragma("unroll")
      for(int local_y=0; local_y<total_reg_tile_y; local_y+=LOCAL_TILE_Y)
      {
        //deal with ew boundary
        _Pragma("unroll")
        for(int l_y=tid; l_y<LOCAL_TILE_Y+isBOX; l_y+=blockDim.x)
        {
          _Pragma("unroll")
          for(int l_x=0; l_x<halo; l_x++)
          {
            // east
            sm_rbuffer[(l_y+ps_y)*tile_x_with_halo+ tile_x + ps_x + l_x]=boundary_buffer[e_step + l_y + local_y + l_x * boundary_line_size];
            // west
            sm_rbuffer[(l_y+ps_y)*tile_x_with_halo+(-halo) + ps_x + l_x]=boundary_buffer[w_step + l_y + local_y + l_x * boundary_line_size];
          }
        }
        #ifndef BOX
        reg2sm<REAL, sizeof_rspace, LOCAL_TILE_Y>(r_space, sm_rbuffer, 
                                  ps_y, ps_x, tid, 
                                  tile_x_with_halo, local_y+halo);
        
        #else
        // reg2sm<REAL, sizeof_rspace, LOCAL_TILE_Y>(r_space, sm_rbuffer, 
        //                           ps_y+halo, ps_x, tid, 
        //                           tile_x_with_halo, local_y+halo+halo);
        reg2sm<REAL, sizeof_rspace, LOCAL_TILE_Y+2*halo>(r_space, sm_rbuffer, 
                                  ps_y-halo, ps_x, tid, 
                                  tile_x_with_halo, local_y);
        #endif 
        __syncthreads();
        #ifdef BOX
        sm2regs_nomiddle<REAL,sizeof_rbuffer, LOCAL_TILE_Y,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    2*halo,
                                                    ps_x, tid,
                                                    tile_x_with_halo,
                                                    2*halo); 
        // _Pragma("unroll")
        // for(int i=0; i<LOCAL_TILE_Y+2*halo; i++)
        // {
        //   r_smbuffer[halo][i]=r_space[i+local_y];
        // }
        #endif
        REAL sum[LOCAL_TILE_Y];
        init_reg_array<REAL,LOCAL_TILE_Y>(sum,0); 
        #ifdef BOX
        computation<REAL,LOCAL_TILE_Y,halo,sizeof_rbuffer,sizeof_rspace>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_smbuffer,
                                      r_space, halo+local_y,
                                      stencilParaInput);
        // computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                      // sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      // r_smbuffer, halo,
                                      // stencilParaInput);
        #else
        computation<REAL,LOCAL_TILE_Y,halo,sizeof_rspace>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_space, local_y+halo,
                                      stencilParaInput);
        #endif
        // __syncthreads();
        // _Pragma("unroll")
        // for(int i=LOCAL_TILE_Y; i<LOCAL_TILE_Y+2*halo; i++)
        // {
        //   r_smbuffer[halo][i]=r_space[i+local_y];
        // }
        reg2reg<REAL, LOCAL_TILE_Y, sizeof_rspace, LOCAL_TILE_Y>(sum,r_space, 0, local_y);
        #ifdef BOX
        // ptrselfcp<REAL,-halo, halo,halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);
        REG2REG<REAL, sizeof_rbuffer, sizeof_rbuffer, 2*halo,isBOX>
                (r_smbuffer,r_smbuffer, LOCAL_TILE_Y, 0);
        #endif
        __syncthreads();
      }
      // ptrselfcp<REAL,-halo, halo,halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);
      // __syncthreads();
      // #ifndef BOX
        reg2sm<REAL, sizeof_rspace, 2*halo>(r_space, sm_rbuffer, 
                                    ps_y-halo, ps_x, tid, 
                                    tile_x_with_halo, sizeof_rspace-halo*2);
          
        __syncthreads();
      // #endif

      #ifndef BOX
      SM2REG<REAL,sizeof_rbuffer, halo*2,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);  
      #else
      _Pragma("unroll")
      for(int i=0; i<2*halo; i++)
      {
        r_smbuffer[halo][i]=r_space[i+sizeof_rspace-2*halo];
      }
      #endif      
    }
    if(UseSMCache)
    //computation of share memory space
    {
      //load shared memory boundary
      for(int local_y=tid; local_y<total_sm_tile_y+isBOX; local_y+=blockDim.x)
      {
        // _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          // east
          sm_space[(ps_y + local_y)*tile_x_with_halo+ tile_x + ps_x+l_x] = boundary_buffer[e_step + local_y + total_reg_tile_y + l_x*boundary_line_size];
          //west
          sm_space[(ps_y + local_y)*tile_x_with_halo+(-halo) + ps_x+l_x] = boundary_buffer[w_step + local_y + total_reg_tile_y + l_x*boundary_line_size];
        }
      }
      __syncthreads();
      //computation of shared space 
      for ( size_t local_y = 0; local_y < total_sm_tile_y; local_y+=LOCAL_TILE_Y) 
      {
        SM2REG<REAL,sizeof_rbuffer,LOCAL_TILE_Y,isBOX>(sm_space, r_smbuffer, 
                                          ps_y+local_y+halo, 
                                          ps_x, tid,
                                          tile_x_with_halo,
                                          halo*2);
        REAL sum[LOCAL_TILE_Y];
        init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
        
        computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                    sm_space, ps_y+local_y, local_x+ps_x, tile_x_with_halo,
                                    r_smbuffer, halo,
                                    stencilParaInput);
        __syncthreads();
        reg2sm<REAL, LOCAL_TILE_Y, LOCAL_TILE_Y>(sum, sm_space,
                                    ps_y+local_y,
                                    ps_x, tid,
                                    tile_x_with_halo,
                                    0);
        __syncthreads();
        REG2REG<REAL, sizeof_rbuffer, sizeof_rbuffer, 2*halo,isBOX>
                (r_smbuffer,r_smbuffer, LOCAL_TILE_Y, 0);
      }
    }
    if(UseSMCache)
    {
      //ALL information already loaded, 
      //theoretically can use information inside chip instead of loading from global memory. 
      // global2sm<REAL,halo,ISINITI>(input, sm_rbuffer, 
      global2sm<REAL,halo,false>(input, sm_rbuffer, 
                                            halo*2,
                                            p_y_cache_end-halo, width_y,
                                            p_x, width_x,
                                            ps_y-halo, ps_x, tile_x_with_halo,                                    
                                            tid);
    }
    else if(UseRegCache)
    {
      // global2sm<REAL,halo,ISINITI,SYNC>(input, sm_rbuffer, 
      global2sm<REAL,halo,false,SYNC>(input, sm_rbuffer, 
                                            halo,
                                            p_y_cache_end, width_y,
                                            p_x, width_x,
                                            ps_y, ps_x, tile_x_with_halo,                                    
                                            tid);
    }
#ifndef SMALL
    for(int global_y=p_y_cache_end; global_y<p_y_end; global_y+=LOCAL_TILE_Y)
    {

      global2sm<REAL,halo>(input, sm_rbuffer,
                                          LOCAL_TILE_Y, 
                                          global_y+halo, width_y,
                                          p_x, width_x,
                                          ps_y+halo, ps_x, tile_x_with_halo,
                                          tid);
      SM2REG<REAL,sizeof_rbuffer, LOCAL_TILE_Y, isBOX>(sm_rbuffer, r_smbuffer, 
                                                    2*halo,
                                                    ps_x, tid,
                                                    tile_x_with_halo,
                                                    2*halo);
      REAL sum[LOCAL_TILE_Y];
      init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
      computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_smbuffer, halo,
                                      stencilParaInput);
      reg2global<REAL,LOCAL_TILE_Y,LOCAL_TILE_Y>(sum, __var_4__, 
                  global_y,p_y_end, 
                  p_x+local_x, width_x);
      __syncthreads();
      ptrselfcp<REAL,-halo, halo, halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);
      REG2REG<REAL, sizeof_rbuffer, sizeof_rbuffer, 2*halo,isBOX>
                (r_smbuffer,r_smbuffer, LOCAL_TILE_Y, 0);
    }
#endif
    if(iter==iteration-1)break;
    //register memory related boundary
    //south
    //*******************
    if(UseRegCache)
    {
      if(tid>=blockDim.x-halo)
      {
        int l_x=tid-blockDim.x+halo;
        //east
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<total_reg_tile_y; l_y++)
        {
          boundary_buffer[e_step + l_y + l_x*boundary_line_size] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
      else if(tid<halo)
      {
        int l_x=tid;
        //west
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<total_reg_tile_y; l_y++)
        {
          boundary_buffer[w_step + l_y + l_x*boundary_line_size] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
    }
    //store sm related boundary
    if(UseSMCache)
    {
      _Pragma("unroll")
      for(int local_y=tid; local_y<total_sm_tile_y; local_y+=blockDim.x)
      {
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          //east
          boundary_buffer[e_step+local_y+total_reg_tile_y + l_x*boundary_line_size] = sm_space[(ps_y + local_y) * tile_x_with_halo + tile_x + ps_x - halo + l_x];
          //west
          boundary_buffer[w_step+local_y+total_reg_tile_y + l_x*boundary_line_size] = sm_space[(ps_y + local_y) * tile_x_with_halo + ps_x + l_x];
        }
      }
    }
    //deal with sm related boundary
    //*******************
    //store boundary to global (NS)
    if(UseRegCache||UseSMCache)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<halo; l_y++)
      {
        //north
        if(UseSMCache)
        {
          __var_4__[(p_y+(total_sm_tile_y+total_reg_tile_y)-halo+l_y) * width_x + p_x + tid]=sm_space[(ps_y + total_sm_tile_y - halo+l_y) * tile_x_with_halo + tid + ps_x];//boundary_buffer[N_STEP+tid+l_y*TILE_X];//
        }
        else
        {
          __var_4__[(p_y+(total_sm_tile_y+total_reg_tile_y)-halo+l_y) * width_x + p_x + tid]=r_space[l_y+total_reg_tile_y-halo];
        }
         //south
        if(UseRegCache)
        {
          __var_4__[(p_y+l_y) * width_x + p_x + tid]= r_space[l_y];
        }
        else
        {
          __var_4__[(p_y+l_y) * width_x + p_x + tid]= sm_space[(ps_y + l_y) * tile_x_with_halo + tid + ps_x];
        }
      }
    }
    //*******************
    //store register part boundary
    __syncthreads();
    // store the whole boundary space to l2 cache
    if(UseSMCache||UseRegCache)
    {
      _Pragma("unroll")
      for(int lid=tid; lid<boundary_line_size-isBOX; lid+=blockDim.x)
      {
        //east
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
           //east
          l2_cache_o[(((blockIdx.x* 2 + 1 )* halo+l_x)*width_y)  + p_y +lid] = boundary_buffer[e_step+lid +l_x*boundary_line_size];
          //west
          l2_cache_o[(((blockIdx.x* 2 + 0) * halo+l_x)*width_y)  + p_y +lid] = boundary_buffer[w_step+lid+l_x*boundary_line_size];
        }
      }
    }
    gg.sync();

    REAL* tmp_ptr=__var_4__;
    __var_4__=input;
    input=tmp_ptr;

    if(UseRegCache||UseSMCache)
    {
      tmp_ptr=l2_cache_o;
      l2_cache_o=l2_cache_i;
      l2_cache_i=tmp_ptr;
    
      _Pragma("unroll")
      for(int local_y=tid; local_y<boundary_line_size-isBOX; local_y+=blockDim.x)
      {
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          int cache_y=min(p_y + local_y,width_y-1);
          // east
           boundary_buffer[e_step+local_y+l_x*boundary_line_size] = ((blockIdx.x == gridDim.x-1)?boundary_buffer[e_step+local_y+(halo-1)*boundary_line_size]:
             l2_cache_i[(((blockIdx.x+1)*2+0)* halo+l_x)*width_y + cache_y]);
           //west
           boundary_buffer[w_step+local_y+l_x*boundary_line_size] = ((blockIdx.x == 0)?boundary_buffer[w_step+local_y+0*boundary_line_size]:
            l2_cache_i[(((blockIdx.x-1)*2+1)* halo+l_x)*width_y + cache_y]);
        }
      }
      for(int local_y=tid; local_y<isBOX; local_y+=blockDim.x)
      {
        for(int l_x=0; l_x<halo; l_x++)
        {
          //east
          int global_x = p_x + tile_x + l_x;
          global_x = MIN(width_x-1,global_x);
          boundary_buffer[e_step+local_y +boundary_line_size-isBOX+ l_x*boundary_line_size] 
            = input[MIN((p_y + local_y+boundary_line_size-isBOX),width_y-1) * width_x + global_x];
          //west
          global_x = p_x - halo + l_x;
          global_x = MAX(0,global_x);
          boundary_buffer[w_step+local_y +boundary_line_size-isBOX+ l_x*boundary_line_size] 
            = input[MIN((p_y + local_y+boundary_line_size-isBOX),width_y-1) * width_x + global_x];
        }
      }
    }

    if(UseRegCache)
    {
      _Pragma("unroll")
      for(int l_y=total_reg_tile_y-1; l_y>=0; l_y--)
      {
        r_space[l_y+halo]=r_space[l_y];
      }
    }

  }

  if(UseRegCache)
  {
    // register->global
    reg2global<REAL, sizeof_rspace, total_reg_tile_y, false>(r_space, __var_4__,
                                      p_y, width_y,
                                      p_x+tid, width_x,
                                      0);
  }
  
  if(UseSMCache)
  {
    __syncthreads();
    // shared memory -> global
    sm2global<REAL,false>(sm_space, __var_4__, 
                                    total_sm_tile_y,
                                    p_y+total_reg_tile_y, width_y,
                                    p_x, width_x,
                                    ps_y, ps_x, tile_x_with_halo ,
                                    tid);
  }
  #undef UseRegCache
  #undef SM2REG
  #undef REG2REG
}

