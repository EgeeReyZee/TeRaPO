# Путь к корню проекта (относительно папки gemm)
PROJECT_ROOT = ..
OPENBLAS_DIR = $(PROJECT_ROOT)/OpenBLAS-bin

CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -pthread -I$(OPENBLAS_DIR)/include
# Для Windows указываем путь к .lib или .a и линкуем саму библиотеку
LDFLAGS = -L$(OPENBLAS_DIR)/lib -lopenblas -lm

NTHREADS ?= 1

TARGET = gemm_benchmark
SRC    = main.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
  $(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
  @echo "======================================================"
  @echo "Запуск бенчмарка GEMM"
  @echo "OpenBLAS threads: $(NTHREADS)"
  @echo "======================================================"
  @# В Windows PowerShell переменные окружения ставятся так:
  @$env:OPENBLAS_NUM_THREADS=$(NTHREADS); $env:OMP_NUM_THREADS=$(NTHREADS); ./$(TARGET)
  @echo "======================================================"

clean:
  rm -f $(TARGET).exe
