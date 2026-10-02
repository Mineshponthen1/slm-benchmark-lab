import json
import os
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


def normalize(text):
    return text.lower().replace(",", "").replace("-", " ")


def is_correct(answer, accept):
    clean = normalize(answer)
    return any(normalize(a) in clean for a in accept)


all_answers = []
scores = {}

for model in MODELS:
    print(f"\nTesting {model} ...")
    correct = 0
    for q in QUESTIONS:
        r = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": q["question"]}],
            options=SETTINGS,
        )
        answer = r["message"]["content"].strip()
        ok = is_correct(answer, q["accept"])
        correct += ok
        short = answer.replace("\n", " ")[:60]
        print(f"  {'PASS' if ok else 'FAIL'}  Q{q['id']}: {short}")
        all_answers.append({"model": model, "id": q["id"], "answer": answer, "correct": ok})
    scores[model] = correct
    ollama.generate(model=model, prompt="", keep_alive=0)

os.makedirs("results", exist_ok=True)
with open("results/quality_answers.json", "w", encoding="utf-8") as f:
    json.dump(all_answers, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 45)
print(f"{'Model':<30}{'Score':>15}")
print("-" * 45)
for model, score in scores.items():
    print(f"{model:<30}{score:>10} / {len(QUESTIONS)}")