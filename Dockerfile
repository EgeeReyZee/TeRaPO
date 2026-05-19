FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential gcc g++ gfortran make wget python3 python3-pip libpq-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install psycopg2-binary
COPY OpenBLAS-bin /usr/local
RUN ldconfig
COPY tests/ /app/tests/
COPY gemm/ /app/gemm/
COPY *.py /app/
RUN cd /app/tests && make && cd /app/gemm && make
CMD python3 /app/run_tests.py && python3 /app/record_results.py
