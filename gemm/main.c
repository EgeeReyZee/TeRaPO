#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <cblas.h>

#define N 1000
#define NUM_RUNS 10
#define NUM_THREAD_CONFIGS 5
int thread_configs[NUM_THREAD_CONFIGS] = {1, 2, 4, 8, 16};

typedef double real; 

typedef struct {
    int rows;
    int cols;
    real *data;
} Matrix;

typedef struct {
    int start_row, end_row;
    Matrix A, B, C;
} ThreadData;

Matrix create_matrix(int rows, int cols) {
    Matrix m = {rows, cols, (real *)malloc(rows * cols * sizeof(real))};
    if (!m.data) exit(1);
    return m;
}

void fill_random(Matrix m) {
    for (int i = 0; i < m.rows * m.cols; i++) m.data[i] = (real)rand() / RAND_MAX;
}

void* my_gemm_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    for (int i = td->start_row; i < td->end_row; i++) {
        for (int j = 0; j < td->B.cols; j++) {
            real sum = 0;
            for (int k = 0; k < td->A.cols; k++) {
                sum += td->A.data[i * td->A.cols + k] * td->B.data[k * td->B.cols + j];
            }
            td->C.data[i * td->C.cols + j] = sum;
        }
    }
    return NULL;
}

double run_my_gemm(Matrix A, Matrix B, Matrix C, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int chunk = A.rows / num_threads;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        td[i].start_row = i * chunk;
        td[i].end_row = (i == num_threads - 1) ? A.rows : (i + 1) * chunk;
        td[i].A = A; td[i].B = B; td[i].C = C;
        pthread_create(&threads[i], NULL, my_gemm_thread, &td[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

double run_openblas_gemm(Matrix A, Matrix B, Matrix C) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (sizeof(real) == sizeof(double)) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    A.rows, B.cols, A.cols, 1.0, A.data, A.cols, B.data, B.cols, 0.0, C.data, C.cols);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    A.rows, B.cols, A.cols, 1.0f, (float*)A.data, A.cols, (float*)B.data, B.cols, 0.0f, (float*)C.data, C.cols);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}


void free_matrix(Matrix *m) {
    if (m->data != NULL) {
        free(m->data);
        m->data = NULL;
    }
}

void print_matrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%.2f ", m.data[i * m.cols + j]);
        }
        printf("\n");
    }
}

void gemm_single_threaded(Matrix A, Matrix B, Matrix C) {
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; k++) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            C.data[i * C.cols + j] = sum;
        }
    }
}


int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== GEMM Benchmark: My_GEMM vs OpenBLAS ===\n\n");
    srand(time(NULL));
    Matrix A = create_matrix(N, N);
    Matrix B = create_matrix(N, N);
    Matrix C_my = create_matrix(N, N);
    Matrix C_blas = create_matrix(N, N);
    fill_random(A); fill_random(B);

    printf("Тестирование на матрицах %dx%d (%s)\n", N, N, sizeof(real) == 8 ? "double" : "float");

    for (int t = 0; t < NUM_THREAD_CONFIGS; t++) {
        int threads = thread_configs[t];
        printf("\n=== Тестирование с количеством потоков: %d ===\n", threads);
        
        double geo_mean_sum = 0.0;

        for (int r = 0; r < NUM_RUNS; r++) {
            double t_my = run_my_gemm(A, B, C_my, threads);
            double t_blas = run_openblas_gemm(A, B, C_blas);
            double rel_perf = (t_blas / t_my) * 100.0;
            
            geo_mean_sum += log(rel_perf);

            printf("  Прогон %2d: My_gemm = %8.4f с, OpenBLAS = %8.4f с, Отн. произв. = %.4f %%\n", 
                   r + 1, t_my, t_blas, rel_perf);
        }

        printf("Среднегеометрическая производительность: %.4f %%\n", exp(geo_mean_sum / NUM_RUNS));
    }

    free(A.data); free(B.data); free(C_my.data); free(C_blas.data);
    return 0;
}

