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

void test_cgeru_basic(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {1,0, 0,1};
    float y[4] = {1,0, 0,1};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cgeru(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] -  1.0f) < TOL_FLOAT &&
          fabsf(A[1] -  0.0f) < TOL_FLOAT &&
          fabsf(A[2] -  0.0f) < TOL_FLOAT &&
          fabsf(A[3] -  1.0f) < TOL_FLOAT &&
          fabsf(A[4] -  0.0f) < TOL_FLOAT &&
          fabsf(A[5] -  1.0f) < TOL_FLOAT &&
          fabsf(A[6] - (-1.0f)) < TOL_FLOAT &&
          fabsf(A[7] -  0.0f) < TOL_FLOAT,
          "cgeru: A=0 + x*y^T (unconjugated)");
}

void test_cgeru_real(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {2,0, 3,0};
    float y[4] = {1,0, 4,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cgeru(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] -  2.0f) < TOL_FLOAT &&
          fabsf(A[2] -  8.0f) < TOL_FLOAT &&
          fabsf(A[4] -  3.0f) < TOL_FLOAT &&
          fabsf(A[6] - 12.0f) < TOL_FLOAT,
          "cgeru: real vectors, result matches dger");
}

void test_cgeru_alpha_complex(void) {
    float A[2] = {0,0};
    float x[2] = {1,0};
    float y[2] = {1,0};
    float alpha[2] = {0.0f, 1.0f};

    cblas_cgeru(CblasRowMajor, 1, 1, alpha, x, 1, y, 1, A, 1);

    CHECK(fabsf(A[0] - 0.0f) < TOL_FLOAT &&
          fabsf(A[1] - 1.0f) < TOL_FLOAT,
          "cgeru: complex alpha");
}

void test_cgeru_incx_incy(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[8] = {1,0, 99,99, 0,1, 99,99};
    float y[8] = {1,0, 99,99, 0,1, 99,99};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cgeru(CblasRowMajor, 2, 2, alpha, x, 2, y, 2, A, 2);

    CHECK(fabsf(A[0] -  1.0f) < TOL_FLOAT &&
          fabsf(A[6] - (-1.0f)) < TOL_FLOAT,
          "cgeru: incx=2, incy=2");
}

void test_cgerc_basic(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {1,0, 0,1};
    float y[4] = {1,0, 0,1};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cgerc(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] -  1.0f) < TOL_FLOAT &&
          fabsf(A[1] -  0.0f) < TOL_FLOAT &&
          fabsf(A[2] -  0.0f) < TOL_FLOAT &&
          fabsf(A[3] - (-1.0f)) < TOL_FLOAT &&
          fabsf(A[4] -  0.0f) < TOL_FLOAT &&
          fabsf(A[5] -  1.0f) < TOL_FLOAT &&
          fabsf(A[6] -  1.0f) < TOL_FLOAT &&
          fabsf(A[7] -  0.0f) < TOL_FLOAT,
          "cgerc: A=0 + x*y^H (conjugated)");
}

void test_cgerc_vs_geru_real(void) {
    float A_u[8] = {0,0, 0,0, 0,0, 0,0};
    float A_c[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {1,0, 2,0};
    float y[4] = {3,0, 4,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cgeru(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A_u, 2);
    cblas_cgerc(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A_c, 2);

    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(A_u[i] - A_c[i]) >= TOL_FLOAT) ok = 0;
    CHECK(ok, "cgerc vs cgeru: identical for real vectors");
}

void test_zgeru_basic(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {1,0, 0,1};
    double y[4] = {1,0, 0,1};
    double alpha[2] = {1.0, 0.0};

    cblas_zgeru(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] -  1.0) < TOL_DOUBLE &&
          fabs(A[6] - (-1.0)) < TOL_DOUBLE,
          "zgeru: basic complex");
}

void test_zgerc_basic(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {1,0, 0,1};
    double y[4] = {1,0, 0,1};
    double alpha[2] = {1.0, 0.0};

    cblas_zgerc(CblasRowMajor, 2, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[3] - (-1.0)) < TOL_DOUBLE &&
          fabs(A[6] - 1.0)  < TOL_DOUBLE,
          "zgerc: basic complex conjugated");
}

int main(void) {
    printf("=== cblas_?geru / cblas_?gerc interface tests ===\n\n");

    test_cgeru_basic();
    test_cgeru_real();
    test_cgeru_alpha_complex();
    test_cgeru_incx_incy();

    test_cgerc_basic();
    test_cgerc_vs_geru_real();

    test_zgeru_basic();
    test_zgerc_basic();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}