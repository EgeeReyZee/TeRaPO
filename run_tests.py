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
