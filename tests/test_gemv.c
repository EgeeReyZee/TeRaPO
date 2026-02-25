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


void test_sgemv_basic(void) {
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 3.0f) < TOL_FLOAT &&
          fabsf(y[1] - 7.0f) < TOL_FLOAT,
          "sgemv: basic 2x2 NoTrans");
}

void test_sgemv_trans(void) {
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_sgemv(CblasRowMajor, CblasTrans, 2, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 4.0f) < TOL_FLOAT &&
          fabsf(y[1] - 6.0f) < TOL_FLOAT,
          "sgemv: basic 2x2 Trans");
}

void test_sgemv_alpha_beta(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {2.0f, 3.0f};
    float y[2] = {1.0f, 1.0f};

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                2.0f, A, 2, x, 1, 3.0f, y, 1);

    CHECK(fabsf(y[0] - 7.0f) < TOL_FLOAT &&
          fabsf(y[1] - 9.0f) < TOL_FLOAT,
          "sgemv: alpha=2, beta=3");
}

void test_sgemv_col_major(void) {
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_sgemv(CblasColMajor, CblasNoTrans, 2, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 4.0f) < TOL_FLOAT &&
          fabsf(y[1] - 6.0f) < TOL_FLOAT,
          "sgemv: ColMajor 2x2");
}

void test_sgemv_incx_incy(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[4] = {1.0f, 99.0f, 2.0f, 99.0f};
    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                1.0f, A, 2, x, 2, 0.0f, y, 2);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[2] - 2.0f) < TOL_FLOAT,
          "sgemv: incx=2, incy=2");
}

void test_sgemv_non_square(void) {
    float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float x[2] = {1.0f, 1.0f};
    float y[3] = {0.0f, 0.0f, 0.0f};

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 3.0f)  < TOL_FLOAT &&
          fabsf(y[1] - 7.0f)  < TOL_FLOAT &&
          fabsf(y[2] - 11.0f) < TOL_FLOAT,
          "sgemv: non-square 3x2");
}

void test_dgemv_basic(void) {
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                1.0, A, 2, x, 1, 0.0, y, 1);

    CHECK(fabs(y[0] - 3.0) < TOL_DOUBLE &&
          fabs(y[1] - 7.0) < TOL_DOUBLE,
          "dgemv: basic 2x2 NoTrans");
}

void test_dgemv_trans(void) {
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dgemv(CblasRowMajor, CblasTrans, 2, 2,
                1.0, A, 2, x, 1, 0.0, y, 1);

    CHECK(fabs(y[0] - 4.0) < TOL_DOUBLE &&
          fabs(y[1] - 6.0) < TOL_DOUBLE,
          "dgemv: basic 2x2 Trans");
}

void test_dgemv_large(void) {
    int n = 4;
    double A[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {0.0, 0.0, 0.0, 0.0};

    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n,
                1.0, A, n, x, 1, 0.0, y, 1);

    int ok = 1;
    for (int i = 0; i < n; i++)
        if (fabs(y[i] - x[i]) >= TOL_DOUBLE) ok = 0;
    CHECK(ok, "dgemv: 4x4 identity");
}

void test_cgemv_basic(void) {
    float A[8]  = {1,0, 0,0,  0,0, 1,0};
    float x[4]  = {1,0, 0,1};
    float y[4]  = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_cgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[1] - 0.0f) < TOL_FLOAT &&
          fabsf(y[2] - 0.0f) < TOL_FLOAT &&
          fabsf(y[3] - 1.0f) < TOL_FLOAT,
          "cgemv: 2x2 complex identity NoTrans");
}

void test_cgemv_conj_trans(void) {

    float A[8] = {0,1, 0,0,  0,0, 0,1};
    float x[4] = {1,0, 1,0};
    float y[4] = {0,0, 0,0};
    float alpha[2] = {1.0f, 0.0f};
    float beta[2]  = {0.0f, 0.0f};

    cblas_cgemv(CblasRowMajor, CblasConjTrans, 2, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabsf(y[0] - 0.0f)  < TOL_FLOAT &&
          fabsf(y[1] - (-1.0f)) < TOL_FLOAT &&
          fabsf(y[2] - 0.0f)  < TOL_FLOAT &&
          fabsf(y[3] - (-1.0f)) < TOL_FLOAT,
          "cgemv: 2x2 ConjTrans");
}

void test_zgemv_basic(void) {
    double A[8]  = {2,0, 0,0,  0,0, 2,0};
    double x[4]  = {1,0, 1,0};
    double y[4]  = {0,0, 0,0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zgemv(CblasRowMajor, CblasNoTrans, 2, 2,
                alpha, A, 2, x, 1, beta, y, 1);

    CHECK(fabs(y[0] - 2.0) < TOL_DOUBLE &&
          fabs(y[2] - 2.0) < TOL_DOUBLE,
          "zgemv: 2x2 diagonal complex matrix");
}

int main(void) {
    printf("=== cblas_?gemv interface tests ===\n\n");

    test_sgemv_basic();
    test_sgemv_trans();
    test_sgemv_alpha_beta();
    test_sgemv_col_major();
    test_sgemv_incx_incy();
    test_sgemv_non_square();

    test_dgemv_basic();
    test_dgemv_trans();
    test_dgemv_large();

    test_cgemv_basic();
    test_cgemv_conj_trans();

    test_zgemv_basic();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}