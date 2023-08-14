#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <omp.h>
#include <gsl/gsl_blas.h>

#include "timer.h"

#define N (long) 1024

void rand_mat(double* mat, size_t n) {
    for(size_t i = 0; i < n; i++)
        for(size_t j = 0; j < n; j++) {
            mat[i*n + j] = ((double) rand()) / RAND_MAX;
        }
}

void init_mat(double* mat, size_t n, double val) {
    for(size_t i = 0; i < n; i++)
        for(size_t j = 0; j < n; j++) {
            mat[i*n + j] = val;
        }
}

void print_mat(double* mat, size_t n) {
    printf("%.1f  %.1f  %0.1f", mat[0], mat[1], mat[2]);
    printf(" ... ");
    printf("%.1f  %.1f\n", mat[n-2], mat[n-1]);

    printf("%.1f  %.1f  %0.1f", mat[n], mat[n+1], mat[n+2]);
    printf(" ... ");
    printf("%.1f  %.1f\n", mat[2*n-2], mat[2*n-1]);

    printf("%.1f  %.1f  %0.1f", mat[2*n], mat[2*n+1], mat[2*n+2]);
    printf(" ... ");
    printf("%.1f  %.1f\n", mat[3*n-2], mat[3*n-1]);

    printf(" .    .    .       .    . \n");
    printf(" .    .    .       .    . \n");
    printf(" .    .    .       .    . \n");

    printf("%.1f  %.1f  %0.1f", mat[(n-2)*n], mat[(n-2)*n+1], mat[(n-2)*n+2]);
    printf(" ... ");
    printf("%.1f  %.1f\n", mat[(n-1)*n-2], mat[(n-1)*n-1]);

    printf("%.1f  %.1f  %0.1f", mat[(n-1)*n], mat[(n-1)*n+1], mat[(n-1)*n+2]);
    printf(" ... ");
    printf("%.1f  %.1f\n", mat[n*n-2], mat[n*n-1]);
}

void matmul_naive() {
    double* A = (double*)malloc(N * N * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* C = (double*)malloc(N * N * sizeof(double));

    init_mat(A, N, 1.0);
    init_mat(B, N, 2.0);
    init_mat(C, N, 0.0);

    uint64_t start = get_nsecs();

#ifdef OPENMP
#pragma omp parallel for
#endif // OPENMP
    for(size_t i = 0; i < N; i++)
        for(size_t j = 0; j < N; j++)
            for(size_t k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
        }

    uint64_t end = get_nsecs();
    uint64_t diff = end - start;
    printf("%" PRIu64 "ms\n", diff / 1000000);

    print_mat(C, N);

    free(A);
    free(B);
    free(C);
}

void gsl_matmul() {
    gsl_matrix* A = gsl_matrix_alloc(N, N);
    gsl_matrix* B = gsl_matrix_alloc(N, N);
    gsl_matrix* C = gsl_matrix_alloc(N, N);

    gsl_matrix_set_all(A, 1.0);
    gsl_matrix_set_all(B, 2.0);
    gsl_matrix_set_all(C, 0.0);

    uint64_t start = get_nsecs();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
    uint64_t end = get_nsecs();
    uint64_t diff = end - start;
    printf("%" PRIu64 "ms\n", diff / 1000000);

    print_mat(C, N);

    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
}

int main() {
    srand(time(NULL)); // random seed
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    matmul_naive();
    gsl_matmul();

    return 0;
}
