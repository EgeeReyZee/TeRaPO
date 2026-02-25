# Makefile for CBLAS Level 2 interface tests
# Requires OpenBLAS installed (or other CBLAS-compatible library)
#
# Usage:
#   make             - build all tests
#   make run         - build and run all tests
#   make clean       - remove binaries
#   make NTHREADS=4  - run with 4 OpenBLAS threads (default: 1)
#
# Custom OpenBLAS path (if not installed system-wide):
#   make OPENBLAS=/path/to/openblas

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -pthread
NTHREADS ?= 1

ifdef OPENBLAS
    CFLAGS  += -I$(OPENBLAS)/include
    LDFLAGS  = -L$(OPENBLAS)/lib -lopenblas -lm
else
    LDFLAGS  = -lopenblas -lm
endif

TESTS = test_gemv \
        test_hemv \
        test_symv \
        test_trmv \
        test_trsv \
        test_ger  \
        test_geru_gerc \
        test_syr  \
        test_her  \
        test_syr2 \
        test_her2

.PHONY: all run clean

all: $(TESTS)

$(TESTS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	@echo "======================================================"
	@echo "Running all CBLAS Level 2 interface tests"
	@echo "OpenBLAS threads: $(NTHREADS)"
	@echo "======================================================"
	@OPENBLAS_NUM_THREADS=$(NTHREADS) OMP_NUM_THREADS=$(NTHREADS); \
	PASS=0; FAIL=0; \
	for t in $(TESTS); do \
		echo ""; \
		echo "------ $$t ------"; \
		./$$t; \
		RET=$$?; \
		if [ $$RET -eq 0 ]; then PASS=$$((PASS+1)); else FAIL=$$((FAIL+1)); fi; \
	done; \
	echo ""; \
	echo "======================================================"; \
	echo "Summary: $$PASS test suites passed, $$FAIL failed"; \
	echo "======================================================"; \
	[ $$FAIL -eq 0 ]

clean:
	rm -f $(TESTS)