void sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int *info);

void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int *info);

static inline int lapack_geqrf(int m, int n, FLOAT* a, int lda, FLOAT* tau, FLOAT* work, int lwork)
{
    int info;
#ifdef USE_FLOAT
    sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
#else
    dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
#endif
    if (info) {
        fprintf(stderr, "Error in xGEQRF: %d\n", info);
        abort();
    }
    return info;
}

void sorgqr_(int* m, int* n, int* k, float* a, int* lda, const float* tau, float* work, int* lwork, int *info);

void dorgqr_(int* m, int* n, int* k, double* a, int* lda, const double* tau, double* work, int* lwork, int *info);

static inline int lapack_orgqr(int m, int n, int k, FLOAT* a, int lda, const FLOAT* tau, FLOAT* work, int lwork)
{
    int info;
#ifdef USE_FLOAT
    sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
#else
    dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
#endif
    if (info) {
        fprintf(stderr, "Error in xORGQR: %d\n", info);
        abort();
    }
    return info;
}

void ssyevd_(char* jobz, char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* iwork, int* liwork, int *info);

void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int *info);

static inline int lapack_syevd(char jobz, char uplo, int n, FLOAT* a, int lda, FLOAT* w, FLOAT* work, int lwork, int* iwork, int liwork)
{
    int info;
#ifdef USE_FLOAT
    ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
#else
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
#endif
    if (info) {
        fprintf(stderr, "Error in xSYEVD: %d\n", info);
        abort();
    }
    return info;
}

void spotrf_(char* uplo, int* n, float* a, int* lda, int *info);

void dpotrf_(char* uplo, int* n, double* a, int* lda, int *info);

static inline int lapack_potrf(char uplo, int n, FLOAT* a, int lda)
{
#ifdef TRACE
    double t1 = get_time();
#endif
    int info;
#ifdef USE_FLOAT
    spotrf_(&uplo, &n, a, &lda, &info);
#else
    dpotrf_(&uplo, &n, a, &lda, &info);
#endif
    /* if (info) {
        fprintf(stderr, "Error in xPOTRF: %d\n", info);
        abort();
    } */
    return info;
#ifdef TRACE
    double t2 = get_time();
    printf("%e blas_trmm %c %d %p %d\n",
        t2 - t1, uplo, n, a, lda);
#endif
}

void strtri_(char* uplo, char* diag, int* n, const float* a, int* lda, int* info);

void dtrtri_(char* uplo, char* diag, int* n, const double* a, int* lda, int* info);

inline int lapack_trtri(char uplo, char diag, int n, const FLOAT* a, int lda)
{
    int info;
#ifdef USE_FLOAT
    strtri_(&uplo, &diag, &n, a, &lda, &info);
#else
    dtrtri_(&uplo, &diag, &n, a, &lda, &info);
#endif
    return info;
}

float slange_(char* norm, int* m, int* n, const float* a, int* lda, float* work);

double dlange_(char* norm, int* m, int* n, const double* a, int* lda, double* work);

static inline FLOAT lapack_lange(char norm, int m, int n, const FLOAT* a, int lda)
{
    FLOAT *work = norm == 'I' ? malloc(sizeof(FLOAT) * m) : NULL;
#ifdef USE_FLOAT
    FLOAT ret = slange_(&norm, &m, &n, a, &lda, work);
#else
    FLOAT ret = dlange_(&norm, &m, &n, a, &lda, work);
#endif
    free(work);
    return ret;
}

void slacpy_(char* uplo, int* m, int* n, const float* a, int* lda, float* b, int* ldb);
void dlacpy_(char* uplo, int* m, int* n, const double* a, int* lda, double* b, int* ldb);

static inline void lapack_lacpy(char uplo, int m, int n, const FLOAT* a, int lda, FLOAT* b, int ldb)
{
#ifdef USE_FLOAT
    slacpy_(&uplo, &m, &n, a, &lda, b, &ldb);
#else
    dlacpy_(&uplo, &m, &n, a, &lda, b, &ldb);
#endif
}

void slaset_(char* uplo, int* m, int* n, float* alpha, float* beta, float* a, int* lda);

void dlaset_(char* uplo, int* m, int* n, double* alpha, double* beta, double* a, int* lda);

static inline void lapack_laset(char uplo, int m, int n, FLOAT alpha, FLOAT beta, FLOAT* a, int lda)
{
#ifdef USE_FLOAT
    slaset_(&uplo, &m, &n, &alpha, &beta, a, &lda);
#else
    dlaset_(&uplo, &m, &n, &alpha, &beta, a, &lda);
#endif
}

int ilaenv_(int* ispec, char* name, char* opts, int* n1, int* n2, int* n3, int* n4, int, int);

static inline int lapack_laenv(int ispec, char* name, char* opts, int n1, int n2, int n3, int n4)
{
    return ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4, strlen(name), strlen(opts));
}
