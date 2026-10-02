import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {
    "llama3.2:3b-instruct-q4_K_M": "llama3.2 q4",
    "llama3.2:3b-instruct-q5_K_M": "llama3.2 q5",
    "phi4-mini:latest": "phi4-mini",
    "mistral:latest": "mistral 7B",
}

with open("results/speed.json", encoding="utf-8") as f:
    speed = json.load(f)
with open("results/quality_answers.json", encoding="utf-8") as f:
    answers = json.load(f)

rows = []
for s in speed:
    model = s["model"]
    mine = [a for a in answers if a["model"] == model]
    correct = sum(a["correct"] for a in mine)
    rows.append({
        "model": model,
        "load_s": round(s["load"], 1),
        "tokens_per_s": round(s["speed"], 1),
        "avg_time_s": round(s["total"], 1),
        "memory_gb": round(s["memory_gb"], 1),
        "quality": f"{correct}/{len(mine)}",
        "quality_pct": round(100 * correct / len(mine)),
    })

with open("results/report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

names = [SHORT.get(r["model"], r["model"]) for r in rows]
charts = [
    ("tokens_per_s", "Speed (tokens/s) - higher is better"),
    ("memory_gb", "Memory used (GB) - lower is better"),
    ("quality_pct", "Quality (% correct) - higher is better"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (key, title) in zip(axes, charts):
    values = [r[key] for r in rows]
    bars = ax.bar(names, values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

fig.suptitle("SLM benchmark - CPU-only laptop (32 GB RAM, Intel Arc, no dedicated GPU)")
fig.tight_layout()
fig.savefig("results/report.png", dpi=150)

print("Saved results/report.csv and results/report.png")