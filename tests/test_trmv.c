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

void test_strmv_upper_notrans(void) {
    float A[4] = {2.0f, 3.0f, 0.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 5.0f) < TOL_FLOAT &&
          fabsf(x[1] - 4.0f) < TOL_FLOAT,
          "strmv: upper NoTrans NonUnit");
}

void test_strmv_upper_trans(void) {
    float A[4] = {2.0f, 3.0f, 0.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_strmv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 2.0f) < TOL_FLOAT &&
          fabsf(x[1] - 7.0f) < TOL_FLOAT,
          "strmv: upper Trans NonUnit");
}

void test_strmv_lower_notrans(void) {
    float A[4] = {2.0f, 0.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_strmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 2.0f) < TOL_FLOAT &&
          fabsf(x[1] - 7.0f) < TOL_FLOAT,
          "strmv: lower NoTrans NonUnit");
}

void test_strmv_unit_diagonal(void) {
    float A[4] = {99.0f, 5.0f, 0.0f, 99.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 6.0f) < TOL_FLOAT &&
          fabsf(x[1] - 1.0f) < TOL_FLOAT,
          "strmv: upper NoTrans Unit");
}

void test_strmv_incx(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[4] = {3.0f, 99.0f, 4.0f, 99.0f};

    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 2);

    CHECK(fabsf(x[0] - 3.0f) < TOL_FLOAT &&
          fabsf(x[2] - 4.0f) < TOL_FLOAT,
          "strmv: incx=2");
}

void test_strmv_col_major(void) {
    float A[4] = {2.0f, 0.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};

    cblas_strmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 5.0f) < TOL_FLOAT &&
          fabsf(x[1] - 4.0f) < TOL_FLOAT,
          "strmv: ColMajor upper NoTrans");
}

void test_dtrmv_upper(void) {
    double A[4] = {2.0, 3.0, 0.0, 4.0};
    double x[2] = {1.0, 1.0};

    cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 5.0) < TOL_DOUBLE &&
          fabs(x[1] - 4.0) < TOL_DOUBLE,
          "dtrmv: upper NoTrans NonUnit");
}

void test_dtrmv_lower(void) {
    double A[4] = {2.0, 0.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};

    cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 2.0) < TOL_DOUBLE &&
          fabs(x[1] - 7.0) < TOL_DOUBLE,
          "dtrmv: lower NoTrans NonUnit");
}

void test_ctrmv_upper(void) {
    float A[8] = {2,0, 1,1,  0,0, 3,0};
    float x[4] = {1,0, 0,1};

    cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 1.0f) < TOL_FLOAT &&
          fabsf(x[1] - 1.0f) < TOL_FLOAT &&
          fabsf(x[2] - 0.0f) < TOL_FLOAT &&
          fabsf(x[3] - 3.0f) < TOL_FLOAT,
          "ctrmv: complex upper NoTrans");
}

void test_ztrmv_upper(void) {
    double A[8] = {2,0, 3,0,  0,0, 4,0};
    double x[4] = {1,0, 1,0};

    cblas_ztrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 5.0) < TOL_DOUBLE &&
          fabs(x[2] - 4.0) < TOL_DOUBLE,
          "ztrmv: double complex upper NoTrans");
}

int main(void) {
    printf("=== cblas_?trmv interface tests ===\n\n");

    test_strmv_upper_notrans();
    test_strmv_upper_trans();
    test_strmv_lower_notrans();
    test_strmv_unit_diagonal();
    test_strmv_incx();
    test_strmv_col_major();

    test_dtrmv_upper();
    test_dtrmv_lower();

    test_ctrmv_upper();
    test_ztrmv_upper();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}