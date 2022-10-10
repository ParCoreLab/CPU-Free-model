#ifndef INC_2D_STENCIL_PERKS_COMMON_CUH
#define INC_2D_STENCIL_PERKS_COMMON_CUH

#include <cmath>

real get_random() {
    return ((real)(rand()) / (real)(RAND_MAX - 1));
    // return 1;
}

real *getRandom2DArray(int width_y, int width_x) {
    real(*a)[width_x] = (real(*)[width_x]) new real[width_y * width_x];
    for (int j = 0; j < width_y; j++)
        for (int k = 0; k < width_x; k++) {
            a[j][k] = get_random();
        }
    return (real *)a;
}

real *getZero2DArray(int width_y, int width_x) {
    real(*a)[width_x] = (real(*)[width_x]) new real[width_y * width_x];
    memset((void *)a, 0, sizeof(real) * width_y * width_x);
    return (real *)a;
}

static double checkError2D(int width_x, const real *l_output, const real *l_reference, int y_lb,
                           int y_ub, int x_lb, int x_ub) {
    const real(*output)[width_x] = (const real(*)[width_x])(l_output);
    const real(*reference)[width_x] = (const real(*)[width_x])(l_reference);
    double error = 0.0;
    double max_error = 0.0;
    int max_k = 0, max_j = 0;
    for (int j = y_lb; j < y_ub; j++)
        for (int k = x_lb; k < x_ub; k++) {
            // printf ("Values at index (%d,%d) are %.6f and %.6f\n", j, k, reference[j][k],
            // output[j][k]);
            double curr_error = output[j][k] - reference[j][k];
            curr_error = (curr_error < 0.0 ? -curr_error : curr_error);
            error += curr_error * curr_error;
            if (curr_error > max_error) {
                printf("Values at index (%d,%d) differ : %f and %f\n", j, k, reference[j][k],
                       output[j][k]);
                max_error = curr_error;
                max_k = k;
                max_j = j;
            }
        }
    printf("[Test] Max Error : %e @ (,%d,%d)\n", max_error, max_j, max_k);
    error = sqrt(error / ((y_ub - y_lb) * (x_ub - x_lb)));

    return error;
}

#endif  // INC_2D_STENCIL_PERKS_COMMON_CUH
