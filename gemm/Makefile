<<<<<<< HEAD
CC=gcc
CFLAGS=-Wall -O3 -pthread -I/usr/local/include
LDFLAGS=-L/usr/local/lib -lopenblas -lm
all: gemm_benchmark
gemm_benchmark: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
clean:
	rm -f gemm_benchmark
=======
# Makefile для gemm
CC = gcc
CFLAGS = -O2 -march=native -Wall -I../OpenBLAS-bin/include
LDFLAGS = -L../OpenBLAS-bin/lib -lopenblas -lm
TARGET = gemm_test
OBJS = main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	del /Q $(OBJS) $(TARGET).exe 2>nul || rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	$(TARGET).exe
>>>>>>> 0ff7048630cd752cb66401127a8b1e91bc413f71
