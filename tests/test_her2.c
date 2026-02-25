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

void test_cher2_real_upper(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {2,0, 3,0};
    float y[4] = {1,0, 4,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cher2(CblasRowMajor, CblasUpper, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] -  4.0f) < TOL_FLOAT &&
          fabsf(A[1] -  0.0f) < TOL_FLOAT &&
          fabsf(A[2] - 11.0f) < TOL_FLOAT &&
          fabsf(A[3] -  0.0f) < TOL_FLOAT &&
          fabsf(A[6] - 24.0f) < TOL_FLOAT,
          "cher2: real vectors upper");
}

void test_cher2_lower_real(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[4] = {2,0, 3,0};
    float y[4] = {1,0, 4,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cher2(CblasRowMajor, CblasLower, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] -  4.0f) < TOL_FLOAT &&
          fabsf(A[4] - 11.0f) < TOL_FLOAT &&
          fabsf(A[6] - 24.0f) < TOL_FLOAT,
          "cher2: real vectors lower");
}

void test_cher2_complex_diagonal_real(void) {
    float A[2] = {0,0};
    float x[2] = {1,1};
    float y[2] = {1,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cher2(CblasRowMajor, CblasUpper, 1, alpha, x, 1, y, 1, A, 1);

    CHECK(fabsf(A[0] - 2.0f) < TOL_FLOAT &&
          fabsf(A[1] - 0.0f) < TOL_FLOAT,
          "cher2: diagonal stays real for complex x,y");
}

void test_cher2_complex_alpha(void) {
    float A[2] = {0,0};
    float x[2] = {1,0};
    float y[2] = {1,0};
    float alpha[2] = {0.0f, 1.0f};

    cblas_cher2(CblasRowMajor, CblasUpper, 1, alpha, x, 1, y, 1, A, 1);

    CHECK(fabsf(A[0] - 0.0f) < TOL_FLOAT &&
          fabsf(A[1] - 0.0f) < TOL_FLOAT,
          "cher2: imaginary alpha, diagonal stays real (=0)");
}

void test_cher2_accumulate(void) {
    /*
     * A = identity, alpha=1
     * x=|(1+0i),(0+0i)|, y=|(0+0i),(1+0i)|
     * A += x*y^H + y*x^H
     * x*y^H: [0][1] = x[0]*conj(y[1]) = (1+0i)*conj(1+0i) = 1*1 = 1
     * y*x^H: [0][1] = y[0]*conj(x[1]) = 0*0 = 0
     * => A[0][1] += 1+0 = 1  (was 0, becomes 1+0i)
     * A[0][0] unchanged=1, A[1][1] unchanged=1
     */
    float A[8] = {1,0, 0,0, 0,0, 1,0};
    float x[4] = {1,0, 0,0};
    float y[4] = {0,0, 1,0};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cher2(CblasRowMajor, CblasUpper, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabsf(A[0] - 1.0f) < TOL_FLOAT &&
          fabsf(A[1] - 0.0f) < TOL_FLOAT &&
          fabsf(A[2] - 1.0f) < TOL_FLOAT &&
          fabsf(A[3] - 0.0f) < TOL_FLOAT &&
          fabsf(A[6] - 1.0f) < TOL_FLOAT,
          "cher2: accumulate on identity");
}

void test_cher2_incx_incy(void) {
    float A[8] = {0,0, 0,0, 0,0, 0,0};
    float x[8] = {2,0, 99,99, 3,0, 99,99};
    float y[8] = {1,0, 99,99, 4,0, 99,99};
    float alpha[2] = {1.0f, 0.0f};

    cblas_cher2(CblasRowMajor, CblasUpper, 2, alpha, x, 2, y, 2, A, 2);

    CHECK(fabsf(A[0] -  4.0f) < TOL_FLOAT &&
          fabsf(A[2] - 11.0f) < TOL_FLOAT &&
          fabsf(A[6] - 24.0f) < TOL_FLOAT,
          "cher2: incx=2, incy=2");
}

void test_zher2_real_upper(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {2,0, 3,0};
    double y[4] = {1,0, 4,0};
    double alpha[2] = {1.0, 0.0};

    cblas_zher2(CblasRowMajor, CblasUpper, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] -  4.0) < TOL_DOUBLE &&
          fabs(A[2] - 11.0) < TOL_DOUBLE &&
          fabs(A[6] - 24.0) < TOL_DOUBLE,
          "zher2: real vectors upper");
}

void test_zher2_lower(void) {
    double A[8] = {0,0, 0,0, 0,0, 0,0};
    double x[4] = {2,0, 3,0};
    double y[4] = {1,0, 4,0};
    double alpha[2] = {1.0, 0.0};

    cblas_zher2(CblasRowMajor, CblasLower, 2, alpha, x, 1, y, 1, A, 2);

    CHECK(fabs(A[0] -  4.0) < TOL_DOUBLE &&
          fabs(A[4] - 11.0) < TOL_DOUBLE &&
          fabs(A[6] - 24.0) < TOL_DOUBLE,
          "zher2: real vectors lower");
}

int main(void) {
    printf("=== cblas_?her2 interface tests ===\n\n");

    test_cher2_real_upper();
    test_cher2_lower_real();
    test_cher2_complex_diagonal_real();
    test_cher2_complex_alpha();
    test_cher2_accumulate();
    test_cher2_incx_incy();

    test_zher2_real_upper();
    test_zher2_lower();

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}