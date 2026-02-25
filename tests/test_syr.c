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

void test_ssyr_upper_basic(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};

    cblas_ssyr(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 2.0f) < TOL_FLOAT &&
          fabsf(A[3] - 4.0f) < TOL_FLOAT,
          "ssyr: upper A=0, alpha=1, x=|1,2|");
}

void test_ssyr_lower_basic(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};

    cblas_ssyr(CblasRowMajor, CblasLower, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[2] - 2.0f) < TOL_FLOAT &&
          fabsf(A[3] - 4.0f) < TOL_FLOAT,
          "ssyr: lower A=0, alpha=1, x=|1,2|");
}

void test_ssyr_alpha(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_ssyr(CblasRowMajor, CblasUpper, 2, 3.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 3.0f) < TOL_FLOAT &&
          fabsf(A[1] - 3.0f) < TOL_FLOAT &&
          fabsf(A[3] - 3.0f) < TOL_FLOAT,
          "ssyr: alpha=3");
}

void test_ssyr_accumulate(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {1.0f, 0.0f};

    cblas_ssyr(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT &&
          fabsf(A[1] - 0.0f) < TOL_FLOAT &&
          fabsf(A[3] - 1.0f) < TOL_FLOAT,
          "ssyr: accumulate on identity");
}

void test_ssyr_incx(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[4] = {1.0f, 99.0f, 3.0f, 99.0f};

    cblas_ssyr(CblasRowMajor, CblasUpper, 2, 1.0f, x, 2, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 3.0f) < TOL_FLOAT &&
          fabsf(A[3] - 9.0f) < TOL_FLOAT,
          "ssyr: incx=2");
}

void test_ssyr_3x3(void) {
    float A[9] = {0,0,0, 0,0,0, 0,0,0};
    float x[3] = {1.0f, 2.0f, 3.0f};

    cblas_ssyr(CblasRowMajor, CblasUpper, 3, 1.0f, x, 1, A, 3);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 2.0f) < TOL_FLOAT &&
          fabsf(A[2] - 3.0f) < TOL_FLOAT &&
          fabsf(A[4] - 4.0f) < TOL_FLOAT &&
          fabsf(A[5] - 6.0f) < TOL_FLOAT &&
          fabsf(A[8] - 9.0f) < TOL_FLOAT,
          "ssyr: 3x3 upper");
}

void test_ssyr_col_major(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};

    cblas_ssyr(CblasColMajor, CblasUpper, 2, 1.0f, x, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[2] - 2.0f) < TOL_FLOAT &&
          fabsf(A[3] - 4.0f) < TOL_FLOAT,
          "ssyr: ColMajor upper");
}

void test_dsyr_upper(void) {
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {1.0, 2.0};

    cblas_dsyr(CblasRowMajor, CblasUpper, 2, 1.0, x, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[1] - 2.0) < TOL_DOUBLE &&
          fabs(A[3] - 4.0) < TOL_DOUBLE,
          "dsyr: upper basic");
}

void test_dsyr_lower(void) {
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {1.0, 2.0};

    cblas_dsyr(CblasRowMajor, CblasLower, 2, 1.0, x, 1, A, 2);

    CHECK(fabs(A[0] - 1.0) < TOL_DOUBLE &&
          fabs(A[2] - 2.0) < TOL_DOUBLE &&
          fabs(A[3] - 4.0) < TOL_DOUBLE,
          "dsyr: lower basic");
}

int main(void) {
    printf("=== cblas_?syr interface tests ===\n\n");

    test_ssyr_upper_basic();
    test_ssyr_lower_basic();
    test_ssyr_alpha();
    test_ssyr_accumulate();
    test_ssyr_incx();
    test_ssyr_3x3();
    test_ssyr_col_major();

    test_dsyr_upper();
    test_dsyr_lower();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}