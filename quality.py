import json
import os
from collections import defaultdict
import ollama

MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:latest",
    "mistral:latest",
]
SETTINGS = {"temperature": 0, "seed": 42, "num_predict": 200}

with open("questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

CATEGORIES = list(dict.fromkeys(q["category"] for q in QUESTIONS))
COUNTS = {c: sum(1 for q in QUESTIONS if q["category"] == c) for c in CATEGORIES}


def normalize(text):
    return text.lower().replace(",", "").replace("-", " ")


def is_correct(answer, accept):
    clean = normalize(answer)
    return any(normalize(a) in clean for a in accept)


all_answers = []
scores = {}

for model in MODELS:
    print(f"\nTesting {model} ...")
    scores[model] = defaultdict(int)
    for q in QUESTIONS:
        r = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": q["question"]}],
            options=SETTINGS,
        )
        answer = r["message"]["content"].strip()
        ok = is_correct(answer, q["accept"])
        if ok:
            scores[model][q["category"]] += 1
        else:
            short = answer.replace("\n", " ")[:60]
            print(f"  FAIL  Q{q['id']} ({q['category']}): {short}")
        all_answers.append({"model": model, "id": q["id"], "category": q["category"],
                            "answer": answer, "correct": ok})
    ollama.generate(model=model, prompt="", keep_alive=0)

os.makedirs("results", exist_ok=True)
with open("results/quality_answers.json", "w", encoding="utf-8") as f:
    json.dump(all_answers, f, indent=2, ensure_ascii=False)

header = f"{'Model':<30}" + "".join(f"{c:>12}" for c in CATEGORIES) + f"{'TOTAL':>10}"
print("\n" + "=" * len(header))
print(header)
print("-" * len(header))
for model, cats in scores.items():
    row = f"{model:<30}"
    for c in CATEGORIES:
        cell = f"{cats[c]}/{COUNTS[c]}"
        row += f"{cell:>12}"
    total = sum(cats.values())
    row += f"{total:>7}/{len(QUESTIONS)}"
    print(row)