#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#define TOL_FLOAT  1e-5f
#define TOL_DOUBLE 1e-10

static int fail_count = 0;
static int pass_count = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { printf("[PASS] %s\n", msg); pass_count++; } \
        else      { printf("[FAIL] %s\n", msg); fail_count++; } \
    } while(0)

void test_chemv_identity(void) {
    float A[8] = {1,0, 0,0,  0,0, 1,0};
    float x[4] = {1,1, 2,-1};
    float y[4] = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[1] - 1.0f) < TOL_FLOAT &&
          fabsf(y[2] - 2.0f) < TOL_FLOAT &&
          fabsf(y[3] - (-1.0f)) < TOL_FLOAT,
          "chemv: identity upper, y=x");
}

void test_chemv_real_diagonal(void) {
    float A[8] = {3,0, 0,0,  0,0, 5,0};
    float x[4] = {1,0, 1,0};
    float y[4] = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 3.0f) < TOL_FLOAT &&
          fabsf(y[1] - 0.0f) < TOL_FLOAT &&
          fabsf(y[2] - 5.0f) < TOL_FLOAT &&
          fabsf(y[3] - 0.0f) < TOL_FLOAT,
          "chemv: real diagonal 2x2 upper");
}

void test_chemv_lower(void) {
    float A[8] = {3,0, 0,0,  0,0, 5,0};
    float x[4] = {1,0, 1,0};
    float y[4] = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasLower, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 3.0f) < TOL_FLOAT &&
          fabsf(y[2] - 5.0f) < TOL_FLOAT,
          "chemv: real diagonal 2x2 lower");
}

void test_chemv_off_diagonal(void) {
    float A[8] = {2,0, 1,1,  1,-1, 3,0};
    float x[4] = {1,0, 0,1};
    float y[4] = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[1] - 1.0f) < TOL_FLOAT &&
          fabsf(y[2] - 1.0f) < TOL_FLOAT &&
          fabsf(y[3] - 2.0f) < TOL_FLOAT,
          "chemv: off-diagonal Hermitian 2x2");
}

void test_chemv_alpha_beta(void) {
    float A[8] = {1,0, 0,0,  0,0, 1,0};
    float x[4] = {1,0, 1,0};
    float y[4] = {1,0, 1,0};
    float alpha[2] = {2.0f, 0.0f};
    float beta[2]  = {3.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 5.0f) < TOL_FLOAT &&
          fabsf(y[2] - 5.0f) < TOL_FLOAT,
          "chemv: alpha=2, beta=3");
}

void test_chemv_incx_incy(void) {
    float A[8] = {1,0, 0,0,  0,0, 1,0};
    float x[8] = {1,0, 99,99, 2,0, 99,99};
    float y[8] = {0,0, 0,0,   0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_chemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 2, beta, y, 2);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[4] - 2.0f) < TOL_FLOAT,
          "chemv: incx=2, incy=2");
}

void test_zhemv_basic(void) {
    double A[8] = {4,0, 0,0,  0,0, 6,0};
    double x[4] = {1,0, 1,0};
    double y[4] = {0,0, 0,0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zhemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabs(y[0] - 4.0) < TOL_DOUBLE &&
          fabs(y[2] - 6.0) < TOL_DOUBLE,
          "zhemv: real diagonal 2x2");
}

void test_zhemv_lower(void) {
    double A[8] = {4,0, 0,0,  0,0, 6,0};
    double x[4] = {1,0, 1,0};
    double y[4] = {0,0, 0,0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zhemv(CblasRowMajor, CblasLower, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabs(y[0] - 4.0) < TOL_DOUBLE &&
          fabs(y[2] - 6.0) < TOL_DOUBLE,
          "zhemv: lower triangle 2x2");
}

void test_zhemv_off_diagonal(void) {
    double A[8] = {1,0, 0,1,  0,-1, 1,0};
    double x[4] = {1,0, 1,0};
    double y[4] = {0,0, 0,0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zhemv(CblasRowMajor, CblasUpper, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabs(y[0] - 1.0) < TOL_DOUBLE &&
          fabs(y[1] - 1.0) < TOL_DOUBLE &&
          fabs(y[2] - 1.0) < TOL_DOUBLE &&
          fabs(y[3] - (-1.0)) < TOL_DOUBLE,
          "zhemv: off-diagonal complex 2x2");
}

int main(void) {
    printf("=== cblas_?hemv interface tests ===\n\n");

    test_chemv_identity();
    test_chemv_real_diagonal();
    test_chemv_lower();
    test_chemv_off_diagonal();
    test_chemv_alpha_beta();
    test_chemv_incx_incy();

    test_zhemv_basic();
    test_zhemv_lower();
    test_zhemv_off_diagonal();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}