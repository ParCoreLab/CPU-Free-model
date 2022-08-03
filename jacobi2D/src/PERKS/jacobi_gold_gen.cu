#include "./common/jacobi_reference.hpp"
#include "./common/types.hpp"

#include "../../include/common.h"

real *_getZero2DArray(int width_y, int width_x) {
  real (*a)[width_x] = (real (*)[width_x])new real[width_y*width_x];
  memset((void*)a, 0, sizeof(real) * width_y * width_x);
  return (real*)a;
}

static void print_matrix
(int width_y, int width_x, real* l_output)
{
  real (*output)[width_x] = (real (*)[width_x])l_output;
  for( int i = 1 ; i < width_y-1 ; i++ )
    for( int j = 1 ; j < width_x-1 ; j++ )
      printf ("Output[%d][%d] = %.6f\n", i, j, output[i][j]);
}

static void jacobi_step
(const real* l_input, int width_y, int width_x, real* l_output, int step)
{
  const real (*input)[width_x] = (const real (*)[width_x])l_input;
  real (*output)[width_x] = (real (*)[width_x])l_output;
  for( int l_y = 0 ; l_y < width_y-0 ; l_y++ )
  {
    for( int l_x = 0 ; l_x < width_x-0 ; l_x++ )
    {
      int w = (l_x == 0)        ? 0 : 0 - 1;
      int e = (l_x == width_x-1)     ? 0 : 0 + 1;
      int s = (l_y == 0)        ? 0 : 0 - 1;
      int n = (l_y == width_y-1)     ? 0 : 0 + 1;    

      output[l_y][l_x] = 
      ( 
        5*  input[l_y+s] [l_x] + 
        12* input[l_y]   [l_x+w] + 
        15* input[l_y]   [l_x]  +
        12* input[l_y]   [l_x+e] + 
        5*  input[l_y+n] [l_x]
        ) / 118;  

    }  
  } 
}


void jacobi_gold_iterative
(real* l_input, int width_y, int width_x, real* l_output, int iteration)
{

  real* temp = _getZero2DArray(width_y, width_x);
  if(iteration%2==1)
  {
    jacobi_step(l_input, width_y, width_x, l_output, 0);
    for(int i=1; i<iteration; i++)
    {
      jacobi_step( l_output, width_y, width_x, temp, i);
      real *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }
  else
  {
    jacobi_step(l_input, width_y, width_x, temp, 0);
    for(int i=1; i<iteration; i++)
    {
      jacobi_step(temp, width_y, width_x, l_output, i);
      real *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }
}

// PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE_ITERATIVE);