#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>

#include "defs.h"
#include "blas.h"
#include "lapack.h"

void apply_right(int n, FLOAT *A, int lda, int k, FLOAT *x)
{
    // Optimized for A upper triangular initialized to the identity
    for (int i = 0; i < k; i++)
        A[i + k * lda] /= x[0];
    A[k + k * lda] = 1.0 / x[0];
    #pragma omp parallel for schedule(static,1)
    for (int j = k + 1; j < n; j++) {
        for (int i = 0; i < k; i++)
            A[i + j * lda] += A[i + k * lda] * x[j - k];
        A[k + j * lda] = A[k + k * lda] * x[j - k];
    }
}

void apply_left_right(int n, FLOAT *A, int lda, int k, FLOAT *x, FLOAT *work)
{
    // lower storage version
    // columns up to k
    #pragma omp parallel for schedule(static,1)
    for (int j = 0; j < k; j++) {
        A[k + j * lda] /= x[0];
        for (int i = k + 1; i < n; i++)
            A[i + j * lda] += A[k + j * lda] * x[i - k];
    }
    // column k
    work[0] = A[k + k * lda] /= x[0];
    A[k + k * lda] /= x[0];
    for (int i = k + 1; i < n; i++) {
        work[i - k] = A[i + k * lda] / x[0];
        A[i + k * lda] = work[i - k] + A[k + k * lda] * x[i - k]; // A[i + k * lda] = (A[i + k * lda] + work[0] * x[i - k]) / x[0];
    }
    // columns from k
    #pragma omp parallel for schedule(static,1)
    for (int j = k + 1; j < n; j++) {
        for (int i = j; i < n; i++)
            A[i + j * lda] += work[j - k] * x[i - k] + A[i + k * lda] * x[j - k];
    }
}

int memo_qr_panel(int n, FLOAT *A, int lda, FLOAT *R, FLOAT *tau, FLOAT *B, FLOAT *C, FLOAT *work)
{
    const FLOAT tol = sqrt(EPSILON);
    // printf("tol=%e\n", tol);

    // copy A to R
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            R[i + j * n] = A[i + j * lda];

    FLOAT *vn = ALLOC(FLOAT, n);
    for (int j = 0; j < n; j++)
        vn[j] = C[j + j * n] = sqrt(C[j + j * n]);

    int k = 0;
    bool breakdown = false;
    for (; k < n && !breakdown; k++) {

        // build HouseHolder vector
        // FLOAT a = blas_dot(m - k, A + k * lda + k, 1, A + k * lda + k, 1);
        FLOAT a = vn[k];
        if (R[k * n + k] > 0.0) a = -a;
        FLOAT scale = R[k * n + k] - a;
        R[k * n + k] = a;
        tau[k] = -scale / a;
        for (int i = 0; i < k; i++) A[k * lda + i] = 0.0;
        A[k * lda + k] = 1.0;
        for (int i = k + 1; i < n; i++)
            A[k * lda + i] = R[k * n + i] /= scale;

        // apply HouseHolder vector
        /*
        A[k * lda + k] = 1.0;
        blas_gemv('T', m - k, n - k - 1, 1.0, A + (k + 1) * lda + k, lda, A + k * lda + k, 1, 0.0, work + 1, 1);
        blas_ger(m - k, n - k - 1, -tau[k], A + k * lda + k, 1, work + 1, 1, A + (k + 1) * lda + k, lda);
        A[k * lda + k] = a;
        */
        work[k + k * n] = scale;
        #pragma omp parallel for schedule(static)
        for (int j = k + 1; j < n; j++) {
            // work[j - k] = -tau[k] * ((C[k * n + j] - a_kk * A[j * lda + k]) / scale + A[j * lda + k]);
            // R[j * n + k] += work[j - k];
            FLOAT b = C[k * n + j] / a;
            FLOAT w = work[j * n + k] = b - R[j * n + k];
            R[j * n + k] = b;
            for (int i = k + 1; i < n; i++) {
                R[j * n + i] += w * A[k * lda + i];
            }
        }

        // update vn and C
        // #pragma omp parallel for schedule(static, 1)
        for (int j = k + 1; j < n; j++) {
            FLOAT t = R[j * n + k] / vn[j];
            t = (1.0 - t) * (1.0 + t);
            if (t < 0.0) t = 0.0;
            FLOAT t2 = vn[j] / C[j + j * n];
            t2 = t * t2 * t2;
            if (t2 <= tol) {
                // printf("breakdown %d vn = %e %e %e\n", j, vn[j], t, t2);
                breakdown = true;
            }
            vn[j] *= sqrt(t);
            /* if (C[j + j * n] / vn[j] > 10) {
                // printf("breakdown %d norm = %e vn = %e\n", j, C[j + j * n], vn[j]);
                breakdown = true;
            } */
            // printf("[%d] norm=%e vn=%e t=%e t2=%e\n", j, C[j + j * n], vn[j], t, t2);
            for (int i = j + 1; i < n; i++)
                C[j * n + i] -= R[j * n + k] * R[i * n + k];
        }
        // printf("%e\n", C[n * n - 1]);

        // update B
        // apply_right(n, B, n, k, work + k + k * n);

    }

    // compute B (lower triangular)
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < k; i++) {
        B[i * n + i] = 1.0 / work[i * n + i];
        for (int j = i + 1; j < k; j++) {
            FLOAT b = 0.0;
            for (int l = i; l < j; l++) {
                b += B[i * n + l] * work[j * n + l];
            }
            B[i * n + j] = b / work[j * n + j];
        }
    }

    free(vn);
    return k;
}

void memo_qr_panel_left(int m, int n, FLOAT *A, int lda, FLOAT *R, FLOAT *tau, FLOAT *work)
{
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < k; i++) {
            R[k * n + i] = A[k * lda + i];
            A[k * lda + i] = 0.0;
        }
        // build HouseHolder vector
        FLOAT a = sqrt(blas_dot(m - k, A + k * lda + k, 1, A + k * lda + k, 1));
        if (A[k * n + k] > 0.0) a = -a;
        FLOAT scale = A[k * n + k] - a;
        R[k * n + k] = a;
        tau[k] = -scale / a;
        A[k * lda + k] = 1.0;
        for (int i = k + 1; i < m; i++)
            A[k * lda + i] /= scale;

        // apply HouseHolder vector
        blas_gemv('T', m - k, n - k - 1, 1.0, A + (k + 1) * lda + k, lda, A + k * lda + k, 1, 0.0, work, 1);
        blas_ger(m - k, n - k - 1, -tau[k], A + k * lda + k, 1, work, 1, A + (k + 1) * lda + k, lda);
    }
}

void compute_t(int m, int k, FLOAT *v, int ldv, FLOAT *tau, FLOAT *t, int ldt)
{
    blas_syrk('L', 'T', k, m, 1.0, v, ldv, 0.0, t, k);
    #pragma omp parallel for schedule(static,1)
    for (int j = 0; j < k; j++) {
        t[j + j * ldt] = tau[j];
        for (int i = j + 1; i < k; i++)
            t[i + j * ldt] *= -tau[i];
    }
    for (int i = 0; i < k; i++) {
        blas_trmv('L', 'T', 'N', i, t, ldt, t + i, ldt);
    }
}

void compute_inv_t(int m, int k, FLOAT *v, int ldv, FLOAT *t, int ldt)
{
    blas_syrk('L', 'T', k, m, 1.0, v, ldv, 0.0, t, k);
    for (int j = 0; j < k; j++)
        t[j + j * ldt] /= 2.0;
}
