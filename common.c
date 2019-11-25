#include <math.h>
#include <stdio.h>
#include <string.h>

#include "defs.h"
#include "blas.h"
#include "lapack.h"

FLOAT *alloc_matrix(int m, int n)
{
    FLOAT *A = ALLOC(FLOAT, m * n);
    srand(1);
    for (int i = 0; i < m * n; i++)
        A[i] = 1.0 * rand() / RAND_MAX;
    return A;
}

void test_matrix(int m, int n, FLOAT *A, FLOAT *R, FLOAT rho)
{
    FLOAT *tau = ALLOC(FLOAT, n);
    FLOAT work_tmp[2];
    lapack_geqrf(m, n, A, m, tau, work_tmp, -1);
    lapack_orgqr(m, n, n, A, m, tau, work_tmp + 1, -1);
    int lwork = work_tmp[0] > work_tmp[1] ? work_tmp[0] : work_tmp[1];
    FLOAT *work = ALLOC(FLOAT, lwork);

    // Compute A with rho
    srand(1);
    for (int i = 0; i < m * n; i++)
        A[i] = 1.0 * rand() / RAND_MAX;
    lapack_geqrf(m, n, A, m, tau, work, lwork);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            R[i + j * n] = A[i + j * m];
    lapack_orgqr(m, n, n, A, m, tau, work, lwork);
    int k = floor(n / 2.0);
    R[k + k * n] = rho;
    blas_trmm('R', 'U', 'N', 'N', m, n, 1.0, R, n, A, m);

    free(tau);
    free(work);
}

void print_matrix(char *format, int m, int n, FLOAT *A, int lda)
{
    for (int i = 0; i < m; i++) {
        printf(format, A[i]);
        for (int j = 1; j < n; j++) {
            putchar(' ');
            printf(format, A[i + j * lda]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void print_matrix2(char *format, int m, int n, FLOAT *A, int lda, FLOAT *B, int ldb)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf(format, A[i + j * lda]);
            putchar(' ');
        }
        putchar('|');
        for (int j = 0; j < n; j++) {
            putchar(' ');
            printf(format, B[i + j * ldb]);
        }
        putchar('\n');
    }
    putchar('\n');
}

int double_cmp(const void *a, const void *b)
{
    double x = *(double *)a;
    double y = *(double *)b;
    if (x < y) return -1;
    else if (x > y) return 1;
    else return 0;
}

FLOAT check_orthog(int m, int n, FLOAT *A, FLOAT *W)
{
    blas_gemm('T', 'N', n, n, m, 1.0, A, m, A, m, 0.0, W, n);
    for (int i = 0; i < n; i++) W[i + i * n] -= 1.0;
    return lapack_lange(MATRIX_NORM, n, n, W, n);
}

FLOAT check_residual(int m, int n, FLOAT *Q, FLOAT *R, FLOAT *A)
{
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < n; i++)
            R[i + j * n] = 0.0;
    blas_gemm('N', 'N', m, n, n, 1.0, Q, m, R, n, -1.0, A, m);
    FLOAT res = lapack_lange(MATRIX_NORM, m, n, A, m);
    return res;
}

void copy_triang(int m, int n, const FLOAT *B, FLOAT *R)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++)
            R[i + j * n] = B[i + j * m];
        for (int i = j + 1; i < n; i++)
            R[i + j * n] = 0.0;
    }
}
