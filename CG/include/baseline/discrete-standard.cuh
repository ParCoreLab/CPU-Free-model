#ifndef INC_CG_BASELINE_DISCRETE_STANDARD_CUH
#define INC_CG_BASELINE_DISCRETE_STANDARD_CUH

#include "../common.h"

namespace BaselineDiscreteStandard {
int init(int *device_csrRowIndices, int *device_csrColIndices, real *device_csrVal,
         const int num_rows, const int nnz, bool matrix_is_zero_indexed, const int iter_max,
         real *x_final_result, const double single_gpu_runtime, bool compare_to_single_gpu,
         bool compare_to_cpu, real *x_ref_single_gpu, real *x_ref_cpu);
}

#endif  // INC_CG_BASELINE_DISCRETE_STANDARD_CUH
