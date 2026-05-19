CC=gcc
CFLAGS=-Wall -O3 -pthread -I/usr/local/include
LDFLAGS=-L/usr/local/lib -lopenblas -lm
all: gemm_benchmark
gemm_benchmark: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
clean:
	rm -f gemm_benchmark
