# SLM Benchmark Lab: Local AI Assistant

![tests](https://github.com/Mineshponthen1/slm-benchmark-lab/actions/workflows/tests.yml/badge.svg)

Which small language model should a bank run on a **normal work laptop**, with no GPU, no cloud, and no data leaving the machine?

This project benchmarks four local models for **speed, memory, and answer quality**, picks a winner based on the evidence, and serves it through a validated **FastAPI** assistant.

![Benchmark results](results/report.png)

## Results

Tested on a CPU-only laptop: 32 GB RAM, Intel Arc integrated graphics (no dedicated GPU). Ollama ran every model 100% on the CPU.

| Model | Speed (tokens/s) | Memory (GB) | Quality (30 questions) |
|---|---|---|---|
| **llama3.2 3B q4_K_M** | **14.6** | **2.6** | **28/30** |
| llama3.2 3B q5_K_M | 14.1 | 2.9 | 27/30 |
| phi4-mini | 12.3 | 3.1 | 27/30 |
| mistral 7B | 7.1 | 5.0 | 27/30 |

### Quality by skill

| Model | Maths | Knowledge | General | Tone | Reasoning | Extraction |
|---|---|---|---|---|---|---|
| llama3.2 q4 | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| llama3.2 q5 | 3/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| phi4-mini | 4/5 | 5/5 | 4/5 | 5/5 | 4/5 | 5/5 |
| mistral 7B | 5/5 | 5/5 | 5/5 | 5/5 | 2/5 | 5/5 |

## Key findings

- **The smallest model was the best value.** llama3.2 q4 was the fastest, the lightest, and scored highest. Mistral 7B used about twice the memory at half the speed, with no quality gain.
- **Speed tracks file size on CPU.** Each generated token requires reading the whole model from memory, so a model twice the size runs at roughly half the speed.
- **Everyday bank tasks are solved even by small models.** Every model scored 100% on knowledge, tone classification (complaint vs. question, sentiment), and information extraction (dates, amounts, emails, job titles).
- **Maths and multi-step reasoning are the weak spot for all models.** In a real system, calculations should be handed to regular code (tool use), not left to the model.
- **Small test sets exaggerate differences.** On an early 10-question version, scores ranged from 7 to 10. On 30 questions, they converged to 27 to 28, which is effectively a tie.

## Method

**Speed** (`benchmark.py`)
- Each model is unloaded first, so every model starts cold.
- 1 warm-up run (only its load time is recorded), then 3 timed runs, averaged.
- Fixed settings for fairness: `temperature 0`, `seed 42`, max 200 tokens.
- Memory is read from Ollama while the model is loaded.

**Quality** (`quality.py`)
- 30 questions in 6 categories (5 each), in `questions.json`.
- Graded by keyword match after normalising case, commas, and hyphens.
- Every answer is saved to `results/quality_answers.json`, and all failures were checked by hand.

**Report** (`report.py`) merges both into `results/report.csv` and `results/report.png`.

## Limitations

- One laptop, CPU only. Results on a GPU machine would differ.
- Speed was measured on one prompt. Longer prompts and answers can change the picture.
- 30 questions is still a small exam, and keyword grading can be fooled.
- Load times vary with Windows file caching, so treat them as rough guidance only.

## The assistant API

`app.py` serves the benchmark winner (llama3.2 q4) with the same settings that were tested.

- `GET /health`: checks that Ollama is running and the default model is installed.
- `POST /ask`: send a question, get the answer plus speed numbers.
- Bad requests (empty question, more than 2,000 characters, or a model not on the allow-list) are rejected with **422** before the model is called. If Ollama is down, the API returns **503**.

Example request:

```json
{ "question": "What does KYC mean and why do banks need it?" }
```

Example response:

```json
{
  "answer": "KYC stands for Know Your Customer...",
  "model": "llama3.2:3b-instruct-q4_K_M",
  "tokens": 42,
  "tokens_per_second": 14.8,
  "seconds": 3.1
}
```

## How to run

Requires Python 3.14 and [Ollama](https://ollama.com) with the four models pulled.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

python benchmark.py                  # speed + memory  -> results/speed.json
python quality.py                    # quality         -> results/quality_answers.json
python report.py                     # CSV + chart     -> results/report.*

uvicorn app:app --reload             # API docs at http://127.0.0.1:8000/docs
```

## Tests and CI

```bash
pytest -v
```

10 tests check the API's behaviour (good requests, rejected requests, Ollama down) using a mocked Ollama, plus the structure of the question set. They run in under a second and need no models, so GitHub Actions runs them on every push.

## Project structure

```
app.py               FastAPI assistant (/health, /ask)
benchmark.py         Speed + memory benchmark
quality.py           30-question quality test
questions.json       The question set and answer keys
report.py            Combined CSV + chart
tests/               pytest suite (mocked Ollama)
results/             Benchmark outputs
.github/workflows/   CI: runs pytest on every push
```