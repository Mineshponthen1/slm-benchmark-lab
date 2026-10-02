import statistics
import time
import ollama

MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:latest",
    "mistral:latest",
]
QUESTION = "In two sentences, what is a small language model?"
RUNS = 3
SETTINGS = {"temperature": 0, "seed": 42, "num_predict": 200}


def ask(model):
    start = time.perf_counter()
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": QUESTION}],
        options=SETTINGS,
    )
    total = time.perf_counter() - start
    tokens = r["eval_count"]
    speed = tokens / (r["eval_duration"] / 1e9)
    load = r["load_duration"] / 1e9
    return {"total": total, "tokens": tokens, "speed": speed, "load": load}


def unload(model):
    ollama.generate(model=model, prompt="", keep_alive=0)


results = []
for model in MODELS:
    print(f"\nTesting {model} ...")
    unload(model)

    warmup = ask(model)
    print(f"  warm-up done (load {warmup['load']:.1f} s, not counted)")

    runs = []
    for i in range(RUNS):
        run = ask(model)
        runs.append(run)
        print(f"  run {i + 1}: {run['speed']:.1f} tokens/s, {run['total']:.1f} s")

    results.append({
        "model": model,
        "load": warmup["load"],
        "speed": statistics.mean(r["speed"] for r in runs),
        "total": statistics.mean(r["total"] for r in runs),
        "tokens": statistics.mean(r["tokens"] for r in runs),
    })
    unload(model)

print("\n" + "=" * 70)
print(f"{'Model':<30}{'Load (s)':>10}{'Tokens/s':>10}{'Time (s)':>10}{'Tokens':>10}")
print("-" * 70)
for r in results:
    print(f"{r['model']:<30}{r['load']:>10.1f}{r['speed']:>10.1f}{r['total']:>10.1f}{r['tokens']:>10.0f}")