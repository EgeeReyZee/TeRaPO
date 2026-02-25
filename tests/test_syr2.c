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

void test_ssyr2_upper_basic(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 0.0f) < TOL_FLOAT &&
          fabsf(A[1] - 1.0f) < TOL_FLOAT &&
          fabsf(A[3] - 0.0f) < TOL_FLOAT,
          "ssyr2: upper A=0, e1,e2 vectors");
}

void test_ssyr2_upper_symmetric(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 6.0f)  < TOL_FLOAT &&
          fabsf(A[1] - 10.0f) < TOL_FLOAT &&
          fabsf(A[3] - 16.0f) < TOL_FLOAT,
          "ssyr2: upper x=|1,2|, y=|3,4|");
}

void test_ssyr2_lower(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};

    cblas_ssyr2(CblasRowMajor, CblasLower, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 6.0f)  < TOL_FLOAT &&
          fabsf(A[2] - 10.0f) < TOL_FLOAT &&
          fabsf(A[3] - 16.0f) < TOL_FLOAT,
          "ssyr2: lower x=|1,2|, y=|3,4|");
}

void test_ssyr2_alpha(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 2.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[1] - 2.0f) < TOL_FLOAT,
          "ssyr2: alpha=2");
}

void test_ssyr2_x_equals_y(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {1.0f, 1.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT &&
          fabsf(A[1] - 2.0f) < TOL_FLOAT &&
          fabsf(A[3] - 2.0f) < TOL_FLOAT,
          "ssyr2: x==y gives 2*syr result");
}

void test_ssyr2_accumulate(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 1.0f, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 1.0f) < TOL_FLOAT &&
          fabsf(A[3] - 1.0f) < TOL_FLOAT,
          "ssyr2: accumulate on identity");
}

void test_ssyr2_incx_incy(void) {
    float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x[4] = {1.0f, 99.0f, 2.0f, 99.0f};
    float y[4] = {3.0f, 99.0f, 4.0f, 99.0f};

    cblas_ssyr2(CblasRowMajor, CblasUpper, 2, 1.0f, x, 2, y, 2, A, 2);

    CHECK(fabsf(A[0] - 6.0f)  < TOL_FLOAT &&
          fabsf(A[1] - 10.0f) < TOL_FLOAT &&
          fabsf(A[3] - 16.0f) < TOL_FLOAT,
          "ssyr2: incx=2, incy=2");
}

void test_dsyr2_upper(void) {
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {1.0, 2.0};
    double y[2] = {3.0, 4.0};

    cblas_dsyr2(CblasRowMajor, CblasUpper, 2, 1.0, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] - 6.0)  < TOL_DOUBLE &&
          fabs(A[1] - 10.0) < TOL_DOUBLE &&
          fabs(A[3] - 16.0) < TOL_DOUBLE,
          "dsyr2: upper basic");
}

void test_dsyr2_lower(void) {
    double A[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {1.0, 2.0};
    double y[2] = {3.0, 4.0};

    cblas_dsyr2(CblasRowMajor, CblasLower, 2, 1.0, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] - 6.0)  < TOL_DOUBLE &&
          fabs(A[2] - 10.0) < TOL_DOUBLE &&
          fabs(A[3] - 16.0) < TOL_DOUBLE,
          "dsyr2: lower basic");
}

int main(void) {
    printf("=== cblas_?syr2 interface tests ===\n\n");

    test_ssyr2_upper_basic();
    test_ssyr2_upper_symmetric();
    test_ssyr2_lower();
    test_ssyr2_alpha();
    test_ssyr2_x_equals_y();
    test_ssyr2_accumulate();
    test_ssyr2_incx_incy();

    test_dsyr2_upper();
    test_dsyr2_lower();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}