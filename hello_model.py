import time
import ollama

MODEL = "llama3.2:3b-instruct-q4_K_M"
QUESTION = "In two sentences, what is a small language model?"

start = time.perf_counter()
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
)
elapsed = time.perf_counter() - start

answer = response["message"]["content"]
tokens = response["eval_count"]
gen_seconds = response["eval_duration"] / 1e9
load_seconds = response["load_duration"] / 1e9

print(answer)
print("-" * 40)
print(f"Total time:        {elapsed:.1f} s")
print(f"Loading the model: {load_seconds:.1f} s")
print(f"Tokens generated:  {tokens}")
print(f"Speed:             {tokens / gen_seconds:.1f} tokens/second")