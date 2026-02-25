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

void test_ssymv_identity(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[1] - 2.0f) < TOL_FLOAT,
          "ssymv: identity upper, y=x");
}

void test_ssymv_diagonal(void) {
    float A[4] = {3.0f, 0.0f, 0.0f, 5.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 3.0f) < TOL_FLOAT &&
          fabsf(y[1] - 5.0f) < TOL_FLOAT,
          "ssymv: diagonal 2x2");
}

void test_ssymv_off_diagonal(void) {
    float A[4] = {2.0f, 3.0f, 0.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 5.0f) < TOL_FLOAT &&
          fabsf(y[1] - 7.0f) < TOL_FLOAT,
          "ssymv: off-diagonal 2x2 upper");
}

void test_ssymv_lower(void) {
    float A[4] = {2.0f, 0.0f, 3.0f, 4.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasLower, 2,
                1.0f, A, 2, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 5.0f) < TOL_FLOAT &&
          fabsf(y[1] - 7.0f) < TOL_FLOAT,
          "ssymv: off-diagonal 2x2 lower");
}

void test_ssymv_alpha_beta(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[2] = {1.0f, 1.0f};
    float y[2] = {1.0f, 1.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 2,
                2.0f, A, 2, x, 1, 3.0f, y, 1);

    CHECK(fabsf(y[0] - 5.0f) < TOL_FLOAT &&
          fabsf(y[1] - 5.0f) < TOL_FLOAT,
          "ssymv: alpha=2, beta=3");
}

void test_ssymv_incx_incy(void) {
    float A[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float x[4] = {1.0f, 99.0f, 2.0f, 99.0f};
    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 2,
                1.0f, A, 2, x, 2, 0.0f, y, 2);

    CHECK(fabsf(y[0] - 1.0f) < TOL_FLOAT &&
          fabsf(y[2] - 2.0f) < TOL_FLOAT,
          "ssymv: incx=2, incy=2");
}

void test_ssymv_3x3(void) {
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 6.0f
    };
    float x[3] = {1.0f, 1.0f, 1.0f};
    float y[3] = {0.0f, 0.0f, 0.0f};

    cblas_ssymv(CblasRowMajor, CblasUpper, 3,
                1.0f, A, 3, x, 1, 0.0f, y, 1);

    CHECK(fabsf(y[0] - 6.0f)  < TOL_FLOAT &&
          fabsf(y[1] - 11.0f) < TOL_FLOAT &&
          fabsf(y[2] - 14.0f) < TOL_FLOAT,
          "ssymv: 3x3 upper");
}

void test_dsymv_basic(void) {
    double A[4] = {2.0, 3.0, 0.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dsymv(CblasRowMajor, CblasUpper, 2,
                1.0, A, 2, x, 1, 0.0, y, 1);

    CHECK(fabs(y[0] - 5.0) < TOL_DOUBLE &&
          fabs(y[1] - 7.0) < TOL_DOUBLE,
          "dsymv: basic 2x2 upper");
}

void test_dsymv_lower(void) {
    double A[4] = {2.0, 0.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dsymv(CblasRowMajor, CblasLower, 2,
                1.0, A, 2, x, 1, 0.0, y, 1);

    CHECK(fabs(y[0] - 5.0) < TOL_DOUBLE &&
          fabs(y[1] - 7.0) < TOL_DOUBLE,
          "dsymv: basic 2x2 lower");
}

void test_dsymv_col_major(void) {
    double A[4] = {2.0, 3.0, 3.0, 4.0};
    double x[2] = {1.0, 1.0};
    double y[2] = {0.0, 0.0};

    cblas_dsymv(CblasColMajor, CblasUpper, 2,
                1.0, A, 2, x, 1, 0.0, y, 1);

    CHECK(fabs(y[0] - 5.0) < TOL_DOUBLE &&
          fabs(y[1] - 7.0) < TOL_DOUBLE,
          "dsymv: ColMajor upper 2x2");
}

int main(void) {
    printf("=== cblas_?symv interface tests ===\n\n");

    test_ssymv_identity();
    test_ssymv_diagonal();
    test_ssymv_off_diagonal();
    test_ssymv_lower();
    test_ssymv_alpha_beta();
    test_ssymv_incx_incy();
    test_ssymv_3x3();

    test_dsymv_basic();
    test_dsymv_lower();
    test_dsymv_col_major();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}