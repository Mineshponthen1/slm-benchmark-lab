import json
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path

import leave

MODEL = "llama3.2:3b-instruct-q4_K_M"
TEMPERATURES = [0.0, 0.7]
RUNS = 5
TODAY = date(2026, 10, 2)  # fixed "today", so the expected answers never change
CASES = json.loads(Path("leave_cases.json").read_text(encoding="utf-8"))


def is_correct(result, expected):
    if result.status != expected["status"]:
        return False
    if expected["status"] != "ok":
        return True
    req = result.request
    assert req is not None
    return (
        req.leave_type == expected["leave_type"]
        and req.start_date.isoformat() == expected["start_date"]
        and req.end_date.isoformat() == expected["end_date"]
        and req.working_days == expected["working_days"]
    )


def fingerprint(result):
    req = result.request
    if req is not None:
        return f"{result.status}|{req.leave_type}|{req.start_date}|{req.end_date}"
    return f"{result.status}|{result.message}"


rows = []
summary = {}

for temp in TEMPERATURES:
    print(f"\n=== temperature {temp} ===")
    consistencies = []
    for case in CASES:
        prints = []
        correct_count = 0
        for run in range(1, RUNS + 1):
            start = time.perf_counter()
            r = leave.extract(case["message"], MODEL, temperature=temp, today=TODAY)
            seconds = time.perf_counter() - start
            correct = is_correct(r, case["expected"])
            correct_count += correct
            prints.append(fingerprint(r))
            rows.append({
                "temperature": temp, "case": case["id"], "run": run,
                "status": r.status, "attempts": r.attempts,
                "first_try_valid": r.first_try_valid, "correct": correct,
                "seconds": round(seconds, 2), "answer": fingerprint(r),
            })
        consistency = Counter(prints).most_common(1)[0][1] / RUNS
        consistencies.append(consistency)
        print(f"  case {case['id']:>2}: {correct_count}/{RUNS} correct, consistency {consistency:.0%}")

    mine = [r for r in rows if r["temperature"] == temp]
    first_fail = [r for r in mine if not r["first_try_valid"]]
    summary[str(temp)] = {
        "first_try_valid": statistics.mean(r["first_try_valid"] for r in mine),
        "rescued_by_retry": (
            statistics.mean(r["status"] != "failed" for r in first_fail) if first_fail else None
        ),
        "not_failed": statistics.mean(r["status"] != "failed" for r in mine),
        "correct": statistics.mean(r["correct"] for r in mine),
        "consistency": statistics.mean(consistencies),
        "avg_seconds": statistics.mean(r["seconds"] for r in mine),
    }

Path("results").mkdir(exist_ok=True)
Path("results/variance.json").write_text(
    json.dumps({"model": MODEL, "runs_per_case": RUNS, "summary": summary, "rows": rows}, indent=2),
    encoding="utf-8",
)


def pct(value):
    return "n/a" if value is None else f"{value:.0%}"


print("\n" + "=" * 52)
print(f"{'Measure':<22}" + "".join(f"{'temp ' + t:>15}" for t in summary))
print("-" * 52)
for key, label in [("first_try_valid", "First-try valid"), ("rescued_by_retry", "Rescued by retry"),
                   ("not_failed", "Not failed"), ("correct", "Correct"), ("consistency", "Consistency")]:
    print(f"{label:<22}" + "".join(f"{pct(s[key]):>15}" for s in summary.values()))
print(f"{'Avg seconds per run':<22}" + "".join(f"{s['avg_seconds']:>15.1f}" for s in summary.values()))
print("\nSaved results/variance.json")