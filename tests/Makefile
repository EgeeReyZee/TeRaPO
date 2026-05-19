CC=gcc
CFLAGS=-Wall -O2 -pthread -I/usr/local/include
LDFLAGS=-L/usr/local/lib -lopenblas -lm
TESTS=$(basename $(wildcard *.c))
all: $(TESTS)
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
clean:
	rm -f $(TESTS)
