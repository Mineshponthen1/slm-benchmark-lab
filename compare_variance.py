import json
import statistics
from collections import Counter, defaultdict

FILES = {"before (v4)": "results/variance_v4.json", "after (v5)": "results/variance.json"}


def metrics(rows):
    answers_by_case = defaultdict(list)
    for r in rows:
        answers_by_case[r["case"]].append(r["answer"])
    return {
        "First-try valid": statistics.mean(r["first_try_valid"] for r in rows),
        "Not failed": statistics.mean(r["status"] != "failed" for r in rows),
        "Correct": statistics.mean(r["correct"] for r in rows),
        "Silently wrong": statistics.mean(r["status"] == "ok" and not r["correct"] for r in rows),
        "Consistency": statistics.mean(
            Counter(a).most_common(1)[0][1] / len(a) for a in answers_by_case.values()
        ),
    }


data = {name: json.load(open(path, encoding="utf-8"))["rows"] for name, path in FILES.items()}

for temp in (0.0, 0.7):
    table = {name: metrics([r for r in rows if r["temperature"] == temp]) for name, rows in data.items()}
    print(f"\n=== temperature {temp} ===")
    print(f"{'Measure':<20}" + "".join(f"{name:>15}" for name in table))
    for measure in next(iter(table.values())):
        print(f"{measure:<20}" + "".join(f"{table[name][measure]:>15.0%}" for name in table))
