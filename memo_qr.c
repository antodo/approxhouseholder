#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>

#include "defs.h"
#include "blas.h"
#include "lapack.h"

double memo_qr(int m, int n, int bs, FLOAT *A, int lda, FLOAT *tau)
{
    FLOAT *B = ALLOC(FLOAT, bs * bs);
    FLOAT *C = ALLOC(FLOAT, bs * bs);
    FLOAT *R = ALLOC(FLOAT, bs * bs);
    FLOAT *work = ALLOC(FLOAT, m * bs);
    double t1 = get_time();

    for (int k = 0, b = bs; k < n; k += b) {
        if (k + bs >= n) bs = n - k; // adjust for last block

        // compute panel
        blas_syrk('L', 'T', bs, m - k, 1.0, A + k + k * lda, lda, 0.0, C, bs);
        b = memo_qr_panel(bs, A + k + k * lda, lda, R, tau + k, B, C, work);
        blas_trmm('R', 'L', 'T', 'N', m - k - bs, b, 1.0, B, bs, A + k + bs + k * lda, lda);

        // compute T
        #ifdef T_INV_CPU
            compute_inv_t(m - k, b, A + k + lda * k, lda, C, b);
        #else
            compute_t(m - k, b, A + k + lda * k, lda, tau + k, C, b);
        #endif

        // update trailing matrix
        int r = n - k - b;
        if (r > 0) {
            blas_gemm('T', 'N', b, r, m - k, 1.0, A + k + lda * k, lda, A + k + (k + b) * lda, lda, 0.0, work, b);
            #ifdef T_INV_CPU
                blas_trsm('L', 'L', 'N', 'N', b, r, 1.0, C, b, work, b);
            #else
                blas_trmm('L', 'L', 'N', 'N', b, r, 1.0, C, b, work, b);
            #endif
            blas_gemm('N', 'N', m - k, r, b, -1.0, A + k + k * lda, lda, work, b, 1.0, A + k + (k + b) * lda, lda);
        }

        // store back R in upper part of panel
        for (int j = 0; j < b; j++)
            for (int i = 0; i <= j; i++)
                A[k + i + (k + j) * lda] = R[i + j * bs];
    }

    double t2 = get_time();
    free(B);
    free(C);
    free(R);
    free(work);
    return t2 - t1;
}

int main (int argc, char *argv[])
{
    if (argc != 6) {
        fprintf(stderr, "Missing arguments: m n block rep rho\n");
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int block_size = atoi(argv[3]);
    int rep = atoi(argv[4]);
    FLOAT rho = atof(argv[5]);

    FLOAT *A = ALLOC(FLOAT, m * n);
    FLOAT *B = ALLOC(FLOAT, m * n);
    FLOAT *R = ALLOC(FLOAT, n * n);

    FLOAT *tau = ALLOC(FLOAT, n);
    FLOAT tmp;
    lapack_geqrf(m, n, A, m, tau, &tmp, -1);
    int lwork = tmp;
    lapack_orgqr(m, n, n, A, m, tau, &tmp, -1);
    if (tmp > lwork) lwork = tmp;
    FLOAT *work = ALLOC(FLOAT, lwork);

    double time_r[rep];
    printf("%d %d %d %e\t", m, n, block_size, rho);

    // LAPACK
    test_matrix(m, n, A, R, rho);
    FLOAT norm_A = lapack_lange(MATRIX_NORM, m, n, A, m), residual, orthog;
    for (int i = 0; i < rep; i++) {
        memcpy(B, A, sizeof(FLOAT) * m * n);
        double t1 = get_time();
        lapack_geqrf(m, n, B, m, tau, work, lwork);
        double t2 = get_time();
        time_r[i] = t2 - t1;
    }

    copy_triang(m, n, B, R);
    lapack_orgqr(m, n, n, B, m, tau, work, lwork);
    residual = check_residual(m, n, B, R, A) / norm_A;
    orthog = check_orthog(m, n, B, R);
    qsort(time_r, rep, sizeof(double), double_cmp);
    printf("%e %e ", orthog, residual);
    printf("%e %e %e\t", time_r[0], time_r[rep - 1], time_r[rep / 2]);


    // Approximate Householder QR
    test_matrix(m, n, A, R, rho);
    for (int i = 0; i < rep; i++) {
        memcpy(B, A, sizeof(FLOAT) * m * n);
        time_r[i] = memo_qr(m, n, block_size, B, m, tau);
    }

    copy_triang(m, n, B, R);
    lapack_orgqr(m, n, n, B, m, tau, work, lwork);
    residual = check_residual(m, n, B, R, A) / norm_A;
    orthog = check_orthog(m, n, B, R);
    qsort(time_r, rep, sizeof(double), double_cmp);
    printf("%e %e ", orthog , residual);
    printf("%e %e %e\n", time_r[0], time_r[rep - 1], time_r[rep / 2]);
}
