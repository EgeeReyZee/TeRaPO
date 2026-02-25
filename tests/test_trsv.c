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

void test_strsv_upper_notrans(void) {
    float A[4] = {2.0f, 4.0f, 0.0f, 3.0f};
    float x[2] = {10.0f, 6.0f};

    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 1.0f) < TOL_FLOAT &&
          fabsf(x[1] - 2.0f) < TOL_FLOAT,
          "strsv: upper NoTrans NonUnit");
}

void test_strsv_lower_notrans(void) {
    float A[4] = {2.0f, 0.0f, 3.0f, 5.0f};
    float x[2] = {4.0f, 13.0f};

    cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 2.0f)  < TOL_FLOAT &&
          fabsf(x[1] - 1.4f)  < TOL_FLOAT,
          "strsv: lower NoTrans NonUnit");
}

void test_strsv_upper_trans(void) {
    float A[4] = {2.0f, 4.0f, 0.0f, 3.0f};
    float x[2] = {2.0f, 11.0f};

    cblas_strsv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 1.0f)       < TOL_FLOAT &&
          fabsf(x[1] - 7.0f/3.0f)  < TOL_FLOAT,
          "strsv: upper Trans NonUnit");
}

void test_strsv_unit_diagonal(void) {
    float A[4] = {99.0f, 2.0f, 0.0f, 99.0f};
    float x[2] = {5.0f, 3.0f};

    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - (-1.0f)) < TOL_FLOAT &&
          fabsf(x[1] -   3.0f)  < TOL_FLOAT,
          "strsv: upper NoTrans Unit");
}

void test_strsv_incx(void) {
    float A[4] = {2.0f, 0.0f, 0.0f, 2.0f};
    float x[4] = {4.0f, 99.0f, 6.0f, 99.0f};

    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 2);

    CHECK(fabsf(x[0] - 2.0f) < TOL_FLOAT &&
          fabsf(x[2] - 3.0f) < TOL_FLOAT,
          "strsv: incx=2");
}

void test_strsv_3x3(void) {
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 2.0f,
        0.0f, 0.0f, 2.0f
    };
    float x[3] = {14.0f, 8.0f, 6.0f};

    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                3, A, 3, x, 1);

    CHECK(fabsf(x[0] - 1.0f) < TOL_FLOAT &&
          fabsf(x[1] - 2.0f) < TOL_FLOAT &&
          fabsf(x[2] - 3.0f) < TOL_FLOAT,
          "strsv: 3x3 upper NonUnit");
}

void test_dtrsv_upper(void) {
    double A[4] = {2.0, 4.0, 0.0, 3.0};
    double x[2] = {10.0, 6.0};

    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 1.0) < TOL_DOUBLE &&
          fabs(x[1] - 2.0) < TOL_DOUBLE,
          "dtrsv: upper NoTrans NonUnit");
}

void test_dtrsv_lower(void) {
    double A[4] = {2.0, 0.0, 3.0, 5.0};
    double x[2] = {4.0, 13.0};

    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 2.0) < TOL_DOUBLE &&
          fabs(x[1] - 1.4) < TOL_DOUBLE,
          "dtrsv: lower NoTrans NonUnit");
}

void test_ctrsv_upper(void) {
    float A[8] = {2,0, 0,0,  0,0, 2,0};
    float x[4] = {4,0, 6,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_ctrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabsf(x[0] - 2.0f) < TOL_FLOAT &&
          fabsf(x[2] - 3.0f) < TOL_FLOAT,
          "ctrsv: complex upper NoTrans");
}

void test_ztrsv_upper(void) {
    double A[8] = {3,0, 0,0,  0,0, 3,0};
    double x[4] = {6,0, 9,0};

    cblas_ztrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                2, A, 2, x, 1);

    CHECK(fabs(x[0] - 2.0) < TOL_DOUBLE &&
          fabs(x[2] - 3.0) < TOL_DOUBLE,
          "ztrsv: double complex upper NoTrans");
}

int main(void) {
    printf("=== cblas_?trsv interface tests ===\n\n");

    test_strsv_upper_notrans();
    test_strsv_lower_notrans();
    test_strsv_upper_trans();
    test_strsv_unit_diagonal();
    test_strsv_incx();
    test_strsv_3x3();

    test_dtrsv_upper();
    test_dtrsv_lower();

    test_ctrsv_upper();
    test_ztrsv_upper();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}