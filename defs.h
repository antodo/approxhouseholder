#include <stdlib.h>
#include <time.h>
#include <float.h>

#ifdef USE_FLOAT
#define FLOAT float
#define DOUBLE double
#define EPSILON FLT_EPSILON
#else
#define FLOAT double
#define DOUBLE __float128
#define EPSILON DBL_EPSILON
#endif

// #define TRACE

#define MATRIX_NORM 'F'

#define ALLOC(t, l) (t*)alloc(sizeof(t), l, __FILE__, __func__, __LINE__)

static inline void *alloc(size_t t, size_t l, const char *file, const char *func, int line)
{
    void *p = malloc(t * l );
    if (p == NULL) {
        fprintf(stderr, "Out of memory in %s (%s:%d) allocating %zd elements of size %zd\n",
            func, file, line, l, t);
        abort();
    }
    return p;
}

static inline int max(int a, int b)
{
    return a > b ? a : b;
}

static inline int min(int a, int b)
{
    return a < b ? a : b;
}

static inline double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

FLOAT *alloc_matrix(int m, int n);
void test_matrix(int m, int n, FLOAT *A, FLOAT *R, FLOAT rho);

void print_matrix(char *format, int m, int n, FLOAT *A, int lda);
void print_matrix2(char *format, int m, int n, FLOAT *A, int lda, FLOAT *B, int ldb);
int double_cmp(const void *a, const void *b);
FLOAT check_orthog(int m, int n, FLOAT *A, FLOAT *work);
FLOAT check_residual(int m, int n, FLOAT *Q, FLOAT *R, FLOAT *A);
void copy_triang(int m, int n, const FLOAT *B, FLOAT *R);

int memo_qr_panel(int n, FLOAT *A, int lda, FLOAT *R, FLOAT *tau, FLOAT *B, FLOAT *C, FLOAT *work);
void memo_qr_panel_left(int m, int n, FLOAT *A, int lda, FLOAT *R, FLOAT *tau, FLOAT *work);
void compute_t(int m, int k, FLOAT *v, int ldv, FLOAT *tau, FLOAT *t, int ldt);
void compute_inv_t(int m, int k, FLOAT *v, int ldv, FLOAT *t, int ldt);
