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
