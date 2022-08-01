#include "../include/common.h"

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        } else {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

// Host functions

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha, float *inputVecX,
             float *outputVecY) {
    for (int i = 0; i < num_rows; i++) {
        int num_elems_this_row = I[i + 1] - I[i];

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++) {
            output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
        }
        outputVecY[i] = output;
    }

    return;
}

float dotProduct(float *vecA, float *vecB, int size) {
    float result = 0.0;

    for (int i = 0; i < size; i++) {
        result = result + (vecA[i] * vecB[i]);
    }

    return result;
}

void scaleVector(float *vec, float alpha, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = alpha * vec[i];
    }
}

void saxpy(float *x, float *y, float a, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p, float *r, int nnz,
                      int N, float tol) {
    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;

    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    saxpy(Ax, r, alpham1, N);

    float r1 = dotProduct(r, r, N);

    int k = 1;

    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            scaleVector(p, b, N);

            saxpy(r, p, alpha, N);
        } else {
            for (int i = 0; i < N; i++) p[i] = r[i];
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        float dot = dotProduct(p, Ax, N);
        a = r1 / dot;

        saxpy(p, x, a, N);
        na = -a;
        saxpy(Ax, r, na, N);

        r0 = r1;
        r1 = dotProduct(r, r, N);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}
