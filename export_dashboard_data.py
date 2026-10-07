import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SHORT = {
    "llama3.2:3b-instruct-q4_K_M": "llama3.2 q4",
    "llama3.2:3b-instruct-q5_K_M": "llama3.2 q5",
    "phi4-mini:latest": "phi4-mini",
    "mistral:latest": "mistral 7B",
}
OUT = Path("dashboard/public/data.json")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validity(rows):
    answers_by_case = defaultdict(list)
    for r in rows:
        answers_by_case[r["case"]].append(r["answer"])
    measures = {
        "First-try valid": statistics.mean(r["first_try_valid"] for r in rows),
        "Not failed": statistics.mean(r["status"] != "failed" for r in rows),
        "Correct": statistics.mean(r["correct"] for r in rows),
        "Silently wrong": statistics.mean(r["status"] == "ok" and not r["correct"] for r in rows),
        "Consistency": statistics.mean(
            Counter(a).most_common(1)[0][1] / len(a) for a in answers_by_case.values()
        ),
    }
    return {name: round(100 * value) for name, value in measures.items()}


speed = load("results/speed.json")
answers = load("results/quality_answers.json")
variance = load("results/variance.json")

models = []
for s in speed:
    mine = [a for a in answers if a["model"] == s["model"]]
    models.append({
        "model": s["model"],
        "short": SHORT.get(s["model"], s["model"]),
        "tokens_per_s": round(s["speed"], 1),
        "ttft_s": round(s["ttft"], 2),
        "memory_gb": round(s["memory_gb"], 1),
        "load_s": round(s["load"], 1),
        "quality_pct": round(100 * sum(a["correct"] for a in mine) / len(mine)),
    })

rows = variance["rows"]
validity_by_temp = {str(t): validity([r for r in rows if r["temperature"] == t]) for t in (0.0, 0.7)}

grounding = None
if Path("results/variance_v4.json").exists():
    before = load("results/variance_v4.json")["rows"]
    grounding = {
        "before": validity([r for r in before if r["temperature"] == 0.0]),
        "after": validity_by_temp["0.0"],
    }

data = {
    "generated": date.today().isoformat(),
    "hardware": "CPU-only laptop | 32 GB RAM | Intel Arc integrated graphics | Ollama running 100% on CPU",
    "extraction_model": variance["model"],
    "models": models,
    "validity": validity_by_temp,
    "grounding": grounding,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"Saved {OUT}")