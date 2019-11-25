void sgemm_(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);

void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);

static inline void blas_gemm(char transa, char transb, int m, int n, int k, FLOAT alpha,
            FLOAT *a, int lda, FLOAT *b, int ldb, FLOAT beta, FLOAT *c, int ldc)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_gemm %c %c %d %d %d %e %p %d %p %d %e %p %d\n",
         t2 - t1, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}

void ssyrk_(char *, char *, int *, int *, const float *, const float *, int *, const float *, float *, int *);

void dsyrk_(char *, char *, int *, int *, const double *, const double *, int *, const double *, double *, int *);

static inline void blas_syrk(char uplo, char trans, int n, int k, FLOAT alpha, FLOAT *A, int lda, FLOAT beta, FLOAT *C, int ldc)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    ssyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
#else
    dsyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_syrk %c %c %d %d %e %p %d %e %p %d\n",
         t2 - t1, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
}

void strsm_(char *, char *, char *, char *, int *, int *, const float *, const float *, int *, float *, int *);

void dtrsm_(char *, char *, char *, char *, int *, int *, const double *, const double *, int *, double *, int *);

static inline void blas_trsm(char side, char uplo, char trans, char diag, int m, int n, FLOAT alpha, FLOAT *A, int lda, FLOAT *B, int ldb)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    strsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
#else
    dtrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_trsm %c %c %c %c %d %d %e %p %d %p %d\n",
        t2 - t1, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#endif
}

void strmm_(char *, char *, char *, char *, int *, int *, const float *, const float *, int *, float *, int *);

void dtrmm_(char *, char *, char *, char *, int *, int *, const double *, const double *, int *, double *, int *);

static inline void blas_trmm(char side, char uplo, char transa, char diag, int m, int n, FLOAT alpha, const FLOAT *a, int lda, FLOAT *b, int ldb)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    strmm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#else
    dtrmm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_trmm %c %c %c %c %d %d %e %p %d %p %d\n",
        t2 - t1, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
#endif
}

void sger_(int *, int *, float *, float *, int *, float *, int *, float *, int *);

void dger_(int *, int *, double *, double *, int *, double *, int *, double *, int *);

static inline void blas_ger(int m, int n, FLOAT alpha, FLOAT *x, int incx, FLOAT *y, int incy, FLOAT *A, int lda)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    sger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
#else
    dger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_ger %d %d %e %p %d %p %d %p %d\n",
        t2 - t1, m, n, alpha, x, incx, y, incy, A, lda);
#endif
}

float sdot_(int *n, float* x, int *incx, float *y, int *incy);

double ddot_(int *n, double* x, int *incx, double *y, int *incy);

static inline FLOAT blas_dot(int n, FLOAT* x, int incx, FLOAT *y, int incy)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    FLOAT d = sdot_(&n, x, &incx, y, &incy);
#else
    FLOAT d = ddot_(&n, x, &incx, y, &incy);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_dot %d %p %d %p %d\n",
        t2 - t1, n, x, incx, y, incy);
#endif
    return d;
}

void dgemv_(char *, int *, int *, const double *, const double *, int *, const double *, int *, const double *, double *, int *);

void sgemv_(char *, int *, int *, const float *, const float *, int *, const float *, int *, const float *, float *, int *);

static inline void blas_gemv(char transa, int m, int n, FLOAT alpha, FLOAT *a, int lda, FLOAT *x, int incx, FLOAT beta, FLOAT *y, int incy)
{
#ifdef TRACE
    double t1 = get_time();
#endif
#ifdef USE_FLOAT
    sgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#else
    dgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#endif
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_gemv %c %d %d %e %p %d %p %d %p %d\n",
        t2 - t1, transa, m, n, alpha, x, incx, a, lda, y, incy);
#endif
}

void strmv_(char*, char*, char*, int*, const float*, int*, float*, int*);

void dtrmv_(char*, char*, char*, int*, const double*, int*, double*, int*);

static inline void blas_trmv(char Uplo, char TransA, char Diag, int N, const FLOAT *A, int lda, FLOAT *X, int incX)
{
#ifdef USE_FLOAT
    strmv_(&Uplo, &TransA, &Diag, &N, A, &lda, X, &incX);
#else
    dtrmv_(&Uplo, &TransA, &Diag, &N, A, &lda, X, &incX);
#endif
}

void sscal_(int *n, float *alpha, float *x, int *incx);

void dscal_(int *n, double *alpha, double *x, int *incx);

static inline void blas_scal(int n, FLOAT alpha, FLOAT *x, int incx)
{
#ifdef USE_FLOAT
    sscal_(&n, &alpha, x, &incx);
#else
    dscal_(&n, &alpha, x, &incx);
#endif
}
