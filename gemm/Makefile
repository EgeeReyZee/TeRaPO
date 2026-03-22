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