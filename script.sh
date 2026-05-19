#!/bin/bash
echo "Setting up BLAS tests..."

mkdir -p tests gemm test_results

cat > docker-compose.yml << 'END'
services:
  postgres:
    image: postgres:15-alpine
    container_name: blas_test_db
    environment:
      POSTGRES_USER: blas_user
      POSTGRES_PASSWORD: blas_pass
      POSTGRES_DB: blas_tests
    ports:
      - "5435:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U blas_user"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - blas_network

  blas_tester:
    build: .
    container_name: blas_tester
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DB_PORT: 5435
    volumes:
      - ./test_results:/app/test_results
    networks:
      - blas_network

networks:
  blas_network:
volumes:
  postgres_data:
END

cat > Dockerfile << 'END'
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
END

cat > run_tests.py << 'END'
#!/usr/bin/env python3
import subprocess, json, time, os
TESTS = ["test_gemv", "test_hemv", "test_symv", "test_trmv", "test_trsv", "test_ger", "test_geru_gerc", "test_syr", "test_her", "test_syr2", "test_her2"]
results = {"tests": [], "summary": {"passed": 0, "failed": 0}}
for t in TESTS:
    try:
        start = time.time()
        r = subprocess.run([f"/app/tests/{t}"], capture_output=True, text=True, timeout=60)
        passed = failed = 0
        for line in r.stdout.split('\n'):
            if "passed" in line and "failed" in line:
                import re; m = re.search(r'(\d+).*?(\d+)', line)
                if m: passed, failed = int(m.group(1)), int(m.group(2))
        results["tests"].append({"name": t, "passed": passed, "failed": failed, "time": (time.time()-start)*1000})
        results["summary"]["passed"] += passed
        results["summary"]["failed"] += failed
        print(f"{t}: {passed} passed, {failed} failed")
    except Exception as e:
        results["tests"].append({"name": t, "error": str(e)})
        print(f"{t}: ERROR - {e}")
with open("/app/test_results/results.json", "w") as f: json.dump(results, f)
END

cat > record_results.py << 'END'
#!/usr/bin/env python3
import json, psycopg2, os
if not os.path.exists("/app/test_results/results.json"): exit(0)
with open("/app/test_results/results.json") as f: data = json.load(f)
try:
    conn = psycopg2.connect(host="postgres", port=5432, user="blas_user", password="blas_pass", database="blas_tests")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS results (id SERIAL PRIMARY KEY, suite_name TEXT, test_name TEXT, passed INT, failed INT, time_ms FLOAT, ts TIMESTAMP DEFAULT NOW())")
    cur.execute("INSERT INTO results (suite_name, test_name, passed, failed, time_ms) VALUES (%s, %s, %s, %s, %s)",
                ("BLAS_Tests", "all", data["summary"]["passed"], data["summary"]["failed"], 0))
    for t in data["tests"]:
        if "error" not in t:
            cur.execute("INSERT INTO results (suite_name, test_name, passed, failed, time_ms) VALUES (%s, %s, %s, %s, %s)",
                        ("BLAS_Tests", t["name"], t["passed"], t["failed"], t.get("time", 0)))
    conn.commit()
    print(f"Saved to DB: {data['summary']['passed']} passed, {data['summary']['failed']} failed")
    cur.close(); conn.close()
except Exception as e:
    print(f"DB error: {e}")
END

cat > tests/Makefile << 'END'
CC=gcc
CFLAGS=-Wall -O2 -pthread -I/usr/local/include
LDFLAGS=-L/usr/local/lib -lopenblas -lm
TESTS=$(basename $(wildcard *.c))
all: $(TESTS)
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
clean:
	rm -f $(TESTS)
END

cat > gemm/Makefile << 'END'
CC=gcc
CFLAGS=-Wall -O3 -pthread -I/usr/local/include
LDFLAGS=-L/usr/local/lib -lopenblas -lm
all: gemm_benchmark
gemm_benchmark: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
clean:
	rm -f gemm_benchmark
END

echo "Setup complete. Running docker-compose..."
docker-compose up --build
