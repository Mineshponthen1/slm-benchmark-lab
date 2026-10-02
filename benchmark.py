import json
import os
import statistics
import time
import ollama

MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:latest",
    "mistral:latest",
]
WARMUP_QUESTION = "Briefly explain what a large language model is."
QUESTIONS = [
    "In two sentences, what is a small language model?",
    "In two sentences, what is know-your-customer (KYC) in banking?",
    "In two sentences, why do companies run AI models on their own computers?",
]
SETTINGS = {"temperature": 0, "seed": 42, "num_predict": 200}


def ask(model, question):
    start = time.perf_counter()
    ttft = None
    final = None
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options=SETTINGS,
        stream=True,
    )
    for chunk in stream:
        if ttft is None and chunk["message"]["content"]:
            ttft = time.perf_counter() - start
        if chunk["done"]:
            final = chunk
    total = time.perf_counter() - start

    assert final is not None, "Ollama never sent the final summary"
    tokens = final["eval_count"]
    speed = tokens / (final["eval_duration"] / 1e9)
    load = final["load_duration"] / 1e9
    return {"total": total, "tokens": tokens, "speed": speed, "load": load, "ttft": ttft}


def unload(model):
    ollama.generate(model=model, prompt="", keep_alive=0)


def memory_gb(model):
    for m in ollama.ps().models:
        if m.model == model:
            return (m.size or 0) / 1e9
    return 0.0


results = []
for model in MODELS:
    print(f"\nTesting {model} ...")
    unload(model)

    warmup = ask(model, WARMUP_QUESTION)
    mem = memory_gb(model)
    print(f"  warm-up done (load {warmup['load']:.1f} s, memory {mem:.1f} GB)")

    runs = []
    for i, question in enumerate(QUESTIONS, start=1):
        run = ask(model, question)
        runs.append(run)
        print(f"  run {i}: TTFT {run['ttft']:.2f} s, {run['speed']:.1f} tokens/s, "
              f"{run['total']:.1f} s, {run['tokens']} tokens")

    results.append({
        "model": model,
        "load": warmup["load"],
        "ttft": statistics.mean(r["ttft"] for r in runs),
        "speed": statistics.mean(r["speed"] for r in runs),
        "total": statistics.mean(r["total"] for r in runs),
        "tokens": statistics.mean(r["tokens"] for r in runs),
        "memory_gb": mem,
    })
    unload(model)

os.makedirs("results", exist_ok=True)
with open("results/speed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print(f"{'Model':<30}{'Load (s)':>10}{'TTFT (s)':>10}{'Tokens/s':>10}{'Time (s)':>10}{'Tokens':>10}{'Mem (GB)':>10}")
print("-" * 90)
for r in results:
    print(f"{r['model']:<30}{r['load']:>10.1f}{r['ttft']:>10.2f}{r['speed']:>10.1f}"
          f"{r['total']:>10.1f}{r['tokens']:>10.0f}{r['memory_gb']:>10.1f}")
print("\nSaved results/speed.json")