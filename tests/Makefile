CC = gcc
CFLAGS = -O2 -I../OpenBLAS-bin/include
LDFLAGS = -lopenblas -lm

TESTS = test_gemv test_ger test_geru_gerc test_hemv test_her test_her2 \
        test_symv test_syr test_syr2 test_trmv test_trsv

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TESTS)