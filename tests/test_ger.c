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

void test_sger_basic(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};

    cblas_sger(CblasRowMajor, 2, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 3.0f) < TOL_FLOAT &&
          fabsf(A[1] - 4.0f) < TOL_FLOAT &&
          fabsf(A[2] - 6.0f) < TOL_FLOAT &&
          fabsf(A[3] - 8.0f) < TOL_FLOAT,
          "sger: A=0, alpha=1, rank-1 update");
}

void test_sger_alpha(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {1.0f, 1.0f};

    cblas_sger(CblasRowMajor, 2, 2, 2.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT &&
          fabsf(A[3] - 2.0f) < TOL_FLOAT,
          "sger: alpha=2");
}

void test_sger_accumulate(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};

    cblas_sger(CblasRowMajor, 2, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 1.0f) < TOL_FLOAT &&
          fabsf(A[2] - 0.0f) < TOL_FLOAT &&
          fabsf(A[3] - 1.0f) < TOL_FLOAT,
          "sger: accumulate on existing A");
}

void test_sger_non_square(void) {
    float A[6] = {0,0, 0,0, 0,0};
    float x[3] = {1.0f, 2.0f, 3.0f};
    float y[2] = {1.0f, 1.0f};

    cblas_sger(CblasRowMajor, 3, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[2] - 2.0f) < TOL_FLOAT &&
          fabsf(A[4] - 3.0f) < TOL_FLOAT,
          "sger: non-square 3x2");
}

void test_sger_incx_incy(void) {
    float A[4] = {0,0, 0,0};
    float x[4] = {1.0f, 99.0f, 2.0f, 99.0f};
    float y[4] = {3.0f, 99.0f, 4.0f, 99.0f};

    cblas_sger(CblasRowMajor, 2, 2, 1.0f, x, 2, y, 2, A, 2);

    CHECK(fabsf(A[0] - 3.0f) < TOL_FLOAT &&
          fabsf(A[3] - 8.0f) < TOL_FLOAT,
          "sger: incx=2, incy=2");
}

void test_sger_col_major(void) {
    float A[4] = {0,0, 0,0};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};

    cblas_sger(CblasColMajor, 2, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 3.0f) < TOL_FLOAT &&
          fabsf(A[1] - 6.0f) < TOL_FLOAT &&
          fabsf(A[2] - 4.0f) < TOL_FLOAT &&
          fabsf(A[3] - 8.0f) < TOL_FLOAT,
          "sger: ColMajor");
}

void test_dger_basic(void) {
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {1.0, 2.0};
    double y[2] = {3.0, 4.0};

    cblas_dger(CblasRowMajor, 2, 2, 1.0, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] - 3.0) < TOL_DOUBLE &&
          fabs(A[1] - 4.0) < TOL_DOUBLE &&
          fabs(A[2] - 6.0) < TOL_DOUBLE &&
          fabs(A[3] - 8.0) < TOL_DOUBLE,
          "dger: basic rank-1 update");
}

void test_dger_negative_alpha(void) {
    double A[4] = {2.0, 0.0, 0.0, 2.0};
    double x[2] = {1.0, 0.0};
    double y[2] = {1.0, 0.0};

    cblas_dger(CblasRowMajor, 2, 2, -1.0, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[3] - 2.0) < TOL_DOUBLE,
          "dger: alpha=-1 (subtract outer product)");
}

int main(void) {
    printf("=== cblas_?ger interface tests ===\n\n");

    test_sger_basic();
    test_sger_alpha();
    test_sger_accumulate();
    test_sger_non_square();
    test_sger_incx_incy();
    test_sger_col_major();

    test_dger_basic();
    test_dger_negative_alpha();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}