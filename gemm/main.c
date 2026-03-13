#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)malloc(rows * cols * sizeof(double));
    
    if (m.data == NULL) {
        fprintf(stderr, "Ошибка: не удалось выделить память для матрицы %dx%d\n", rows, cols);
        exit(1);
    }
    
    return m;
}

void fill_random(Matrix m) {
    for (int i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = (double)rand() / RAND_MAX;
    }
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
    srand((unsigned int)time(NULL));

    int N = 1024;

    Matrix A = create_matrix(N, N);
    Matrix B = create_matrix(N, N);
    Matrix C = create_matrix(N, N);

    fill_random(A);
    fill_random(B);

    for (int i = 0; i < N * N; i++) {
        C.data[i] = 0.0;
    }

    printf("Матрицы успешно созданы и заполнены случайными числами.\n");
    // printf("Матрица A:\n");
    // print_matrix(A);
    // printf("Матрица B:\n");
    // print_matrix(B);
    // printf("Матрица C:\n");
    // print_matrix(C);

    clock_t start = clock();
    
    gemm_single_threaded(A, B, C);
    
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Вычисления завершены.\n");
    printf("Время выполнения (1 поток): %.4f секунд\n", cpu_time_used);


    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);

    return 0;
}
