//#include "../common/common.hpp"
//#include "../common/types.hpp"
//#include "../common/jacobi_reference.hpp"
//
//// #ifndef REAL
//// #define REAL float
//// #endif
//
// template<class REAL>
// static void j3d_step
//(const REAL* l_input, int height, int width_y, int width_x, REAL* l_output, int step)
//{
//  const REAL (*input)[width_y][width_x] =
//    (const REAL (*)[width_y][width_x])l_input;
//  REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])l_output;
//
//  for (int l_h = 0; l_h < height-0; l_h++)
//    for (int l_y = 0; l_y < width_y-0; l_y++)
//      for (int l_x = 0; l_x < width_x-0; l_x++) {
//
//        int b0=(l_h-1>0?l_h-1:0);
//        int t0=(l_h+1<=height-1?l_h+1:height-1);
//        int w0 = (l_x-1>0?l_x-1:0);
//        int e0 = (l_x+1<width_x-1?l_x+1:width_x-1);
//        int n0 = (l_y+1<width_y-1?l_y+1:width_y-1);
//        int s0 = (l_y-1>0?l_y-1:0);
//
//        output[l_h][l_y][l_x] =
//          0.161f * input[l_h][l_y][e0] + 0.162f * input[l_h][l_y][w0] +
//          0.163f * input[l_h][n0][l_x] + 0.164f * input[l_h][s0][l_x] +
//          0.165f * input[t0][l_y][l_x] +
//          0.166f * input[b0][l_y][l_x]
//          -1.67f * input[l_h][l_y][l_x];
//	  //if (step == 0) printf ("output[%d][%d][%d] = %.6f (%.6f)\n", i, j, k, output[i][j][k],
// input[i+1][j][k]);
//      }
//}
//
// template<class REAL>
// void j3d_gold_iterative
//(REAL *l_input, int height, int width_y, int width_x, REAL* l_output, int iteration)
//{
//  REAL* temp = getZero3DArray<REAL>(height, width_y, width_x);
//  if(iteration%2==1)
//  {
//    j3d_step(l_input, height, width_y, width_x, l_output, 0);
//    for(int i=1; i<iteration; i++)
//    {
//      j3d_step( l_output,height, width_y, width_x, temp, i);
//      REAL *temp2=temp;
//      temp=l_output;
//      l_output=temp2;
//    }
//  }
//  else
//  {
//    j3d_step(l_input, height, width_y, width_x, temp, 0);
//    for(int i=1; i<iteration; i++)
//    {
//      j3d_step(temp, height, width_y, width_x, l_output, i);
//      REAL *temp2=temp;
//      temp=l_output;
//      l_output=temp2;
//    }
//  }
//
//  // delete[] temp;
//}
//
// PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE_ITERATIVE);