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

void test_cher_upper_basic(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {1,0, 0,1};

    cblas_cher(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 0.0f) < TOL_FLOAT &&
          fabsf(A[2] - 0.0f) < TOL_FLOAT &&
          fabsf(A[3] - (-1.0f)) < TOL_FLOAT &&
          fabsf(A[6] - 1.0f) < TOL_FLOAT,
          "cher: upper basic rank-1");
}

void test_cher_lower_basic(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {1,0, 0,1};

    cblas_cher(CblasRowMajor, CblasLower, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[4] - 0.0f) < TOL_FLOAT &&
          fabsf(A[5] - 1.0f) < TOL_FLOAT &&
          fabsf(A[6] - 1.0f) < TOL_FLOAT,
          "cher: lower basic rank-1");
}

void test_cher_real_vector(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {2,0, 3,0};

    cblas_cher(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 4.0f) < TOL_FLOAT &&
          fabsf(A[2] - 6.0f) < TOL_FLOAT &&
          fabsf(A[6] - 9.0f) < TOL_FLOAT,
          "cher: real vector result");
}

void test_cher_alpha_scale(void) {
    float A[2] = {0,0};
    float x[2] = {1,0};

    cblas_cher(CblasRowMajor, CblasUpper, 1, 2.0f, x, 1, A, 1);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT,
          "cher: alpha=2 scale");
}

void test_cher_accumulate(void) {
    float A[8] = {1,0, 0,0, 0,0, 1,0};
    float x[4] = {1,0, 0,0};

    cblas_cher(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT &&
          fabsf(A[6] - 1.0f) < TOL_FLOAT,
          "cher: accumulate on identity");
}

void test_cher_incx(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[8] = {1,0, 99,99, 2,0, 99,99};

    cblas_cher(CblasRowMajor, CblasUpper, 2, 1.0f, x, 2, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[2] - 2.0f) < TOL_FLOAT &&
          fabsf(A[6] - 4.0f) < TOL_FLOAT,
          "cher: incx=2");
}

void test_zher_upper_basic(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {1,0, 0,1};

    cblas_zher(CblasRowMajor, CblasUpper, 2, 1.0, x, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[2] - 0.0) < TOL_DOUBLE &&
          fabs(A[3] - (-1.0)) < TOL_DOUBLE &&
          fabs(A[6] - 1.0) < TOL_DOUBLE,
          "zher: upper basic rank-1");
}

void test_zher_lower(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {1,0, 0,1};

    cblas_zher(CblasRowMajor, CblasLower, 2, 1.0, x, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[5] - 1.0) < TOL_DOUBLE &&
          fabs(A[6] - 1.0) < TOL_DOUBLE,
          "zher: lower basic rank-1");
}

int main(void) {
    printf("=== cblas_?her interface tests ===\n\n");

    test_cher_upper_basic();
    test_cher_lower_basic();
    test_cher_real_vector();
    test_cher_alpha_scale();
    test_cher_accumulate();
    test_cher_incx();

    test_zher_upper_basic();
    test_zher_lower();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}