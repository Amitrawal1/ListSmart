# ListSmart fine-tune + evaluation (Tinker)

This folder fine-tunes the ListSmart **listing writer** with LoRA on Tinker, using real,
well-rated product listings, and then **measures** the fine-tuned model against the base
model on the same held-out prompts (the Tinker sponsor requirement).

```
finetune/
  prepare_data.py   download + filter good listings -> data/{train,val,test}.jsonl
  train.py          LoRA SFT on Tinker, loss only on the assistant answer
  evaluate.py       base vs fine-tuned on the test set -> results/eval_*.{json,jsonl,md}
                    (--backend tinker = hosted, --backend ollama = on this laptop)
  export_to_ollama.py   Tinker LoRA -> one GGUF file -> `ollama create listsmart-writer`
  metrics.py        pure metric functions (tested in tests/)
  speed.py          pure speed / hardware / cost helpers for evaluate.py
  common.py         paths, jsonl helpers, imports backend/app/prompts.py (the ONE prompt format)
  data/sample_*.jsonl            small samples kept in git so you can see the format
  results/eval_summary.example.json   EXAMPLE ONLY (placeholder numbers) for frontend work
```

## Base model: `Qwen/Qwen3-8B` (and why not Gemma)

- **Gemma is not offered on Tinker.** We checked the Tinker "Models & Pricing" page and the
  tinker-cookbook model registry (`tinker_cookbook/model_info.py`): no Gemma models (Qwen, Llama,
  DeepSeek, GPT-OSS, Kimi, GLM, Nemotron, Inkling only). So Gemma stays the **vision** model in the
  app (it reads the photos), and the **writer** is fine-tuned on a Tinker model.
- **Qwen3-8B** is one of the cheapest models on Tinker (about $0.44 per 1M training tokens and
  $0.60 per 1M sampled tokens at the time of writing). It handles instructions and JSON well, and
  it is a hybrid model, so we can switch thinking off.
- We use the cookbook renderer **`qwen3_disable_thinking`**, so the model answers with the
  listing JSON straight away (no `<think>` text). Training, eval and the app must all use the
  same model + renderer. `train.py` saves both in `results/train_info.json`.
- Note: `meta-llama/Llama-3.1-8B-Instruct` and `Qwen3-4B-Instruct-2507` are still in the cookbook,
  but they are not on Tinker's current price list. If you want a different model, pass
  `--base-model` (for example `Qwen/Qwen3.5-4B`); the renderer is picked automatically.

## Dataset

[`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
product metadata files `raw/meta_categories/meta_<Category>.jsonl` (public, no login). The files
are several GB, so `prepare_data.py` **streams** them over HTTP and stops once it has enough rows
(default: 700 per category from `Clothing_Shoes_and_Jewelry`, `Home_and_Kitchen`,
`Handmade_Products`; it reads about 15k raw lines in total and takes about 15 seconds).

A listing is "good" when: rating >= 4.0 with >= 5 ratings, title 20-250 chars, English,
description >= 120 chars with >= 2 clean sentences, and >= 3 useful features (care labels and
"closure" lines are dropped). Brand/shop names, links, guarantees and shipping talk are removed,
because the prompt tells the model never to invent brands.

For each listing we build:
- **ProductAttributes** with simple rules (no vision needed): product type from a vocabulary or the
  category name, colours/material/pattern/style from word lists plus the `details` fields, gender
  from the Department field, key features from the feature lines.
- **Reference Listing** that obeys the platform rules: title cut to 100 (Meesho) or 140 (Etsy)
  chars at a word boundary, description in 2-4 paragraphs, exactly 5 bullets, up to 13 tags of
  <= 20 chars, a 3-6 word search query. Platform is chosen at random (seeded).
- A few rows get seller notes ("Handmade item.", "Good as a gift.") only when the text supports them.

Split is by product (one row per `parent_asin`, near-duplicate titles removed), seeded:
**train 1880 / val 100 / test 120**. Every reference passes 100% of the constraint checks.

## Setup

```bash
cd ListSmart/finetune
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Keys are read from the environment or from `finetune/.env` / `ListSmart/.env`:

```
TINKER_API_KEY=...        # needed for train.py and evaluate.py
GOOGLE_API_KEY=...        # needed for the Gemma judge in evaluate.py (skip with --no-judge)
GEMMA_MODEL=gemma-4-26b-a4b-it   # optional, judge model
HF_TOKEN=...              # optional, only if HuggingFace rate-limits the download
```

## Commands, in order

```bash
# 0. Offline checks (no keys needed, a few seconds each)
python -m pytest -q tests          # includes the chat-template proof (downloads the ~11 MB Qwen3 tokenizer once)
python prepare_data.py --dry-run
python train.py --dry-run          # tokenises the data, shows exactly which tokens get loss
python evaluate.py --dry-run       # mock outputs -> results/dry_run/ (never read by the app)
python evaluate.py --backend ollama --dry-run --limit 5   # shows the exact Ollama request, no calls
python export_to_ollama.py --dry-run                      # shows every export step, command, disk + RAM need

# 1. Build the dataset (~15 s). Already done once; re-run to rebuild data/*.jsonl
python prepare_data.py

# 2. Fine-tune on Tinker (needs TINKER_API_KEY). Also saves downloadable "external" weights for step 4.
python train.py
#    quick smoke test first if you like (5 steps, a few cents):
#    python train.py --max-steps 5 --eval-every 0 --name listsmart-smoke

# 3. Evaluate base vs fine-tuned ON TINKER, full 120-row test set
#    (needs TINKER_API_KEY, and GOOGLE_API_KEY for the judge)
python evaluate.py
#    without the judge:      python evaluate.py --no-judge
#    judge again later:      python evaluate.py --reuse-outputs

# 4. Export the fine-tuned model to Ollama (see "Run the fine-tuned model locally" below first!)
brew install cmake                 # once, so the script can build llama-export-lora + llama-quantize
python export_to_ollama.py --cleanup          # all steps; re-run to resume after a stop
#    = setup, download, peft, lora_gguf, base_gguf, merge_gguf, quantize, modelfile, create
#    the last step runs:  ollama create listsmart-writer -f export/Modelfile

# 5. Evaluate LOCALLY on the laptop (Ollama running: `ollama serve` or the Ollama app)
ollama pull qwen3:8b               # base model for the comparison (~5.2 GB)
python evaluate.py --backend ollama --limit 20 --no-judge
#    -> results/ollama/eval_summary.json, and copies the laptop "speed" + "cost_per_1000_listings_inr"
#       into results/eval_summary.json (the file the app shows). Add the Gemma judge by dropping --no-judge.

# 6. Optional: closed-model baseline (only if OPENAI_API_KEY or ANTHROPIC_API_KEY is set; skipped otherwise)
python evaluate.py --backend ollama --limit 20 --no-judge --closed-baseline
```

Then point the app at the model (in `ListSmart/.env`):

```
WRITER_BACKEND=ollama                 # local (default in Part 2 of SPEC.md)
OLLAMA_WRITER_MODEL=listsmart-writer
# hosted fallback:
TINKER_MODEL_PATH=<contents of finetune/results/finetuned_model.txt>
TINKER_BASE_MODEL=Qwen/Qwen3-8B
```

The app must sample with the same chat format as training: Tinker uses the renderer
(`qwen3_disable_thinking`, saved in `results/train_info.json`); Ollama gets a raw prompt from
`backend/app/chat_template.py`, which `tests/test_chat_template.py` proves is token-for-token the same.

### Training defaults (all are flags, see `python train.py --help`)

| flag | default | note |
|---|---|---|
| `--base-model` | `Qwen/Qwen3-8B` | |
| `--lora-rank` | 32 | |
| `--lr` | cookbook's `get_lr()` = 4.7e-4 for Qwen3-8B LoRA | |
| `--epochs` | 3 | 58 steps per epoch -> 174 steps |
| `--batch-size` | 32 | |
| `--max-length` | 2048 | longest example is ~800 tokens, so nothing is cut |
| `--eval-every` | 20 | val loss on 100 held-out rows |

The loss is only on the assistant answer (`TrainOnWhat.LAST_ASSISTANT_MESSAGE`); the system and
user prompt tokens have weight 0. `train.py --dry-run` prints both parts so you can check this.

## Outputs

| file | what |
|---|---|
| `results/train_log.jsonl` | one line per step: train loss, lr, val loss every 20 steps |
| `results/train_info.json` | base model, renderer, hyperparameters, val loss before/after |
| `results/finetuned_model.txt` | the `tinker://...` sampler weights path |
| `results/external_weights.txt` | `tinker://.../external_weights/...` downloadable copy (used by `export_to_ollama.py`) |
| `results/ollama/eval_*` | the local Ollama eval (`--backend ollama`), same files as below |
| `export/` (gitignored) | LoRA download, llama.cpp checkout, GGUF files, `Modelfile` |
| `results/eval_outputs.jsonl` | every test row: raw outputs of both models, parsed listing, scores, judge verdict |
| `results/eval_summary.json` | summary in the SPEC shape, served by `GET /api/eval` |
| `results/eval_report.md` | markdown table for the README / slides |

`results/eval_summary.example.json` is a **placeholder** (it has `"note": "example placeholder,
run evaluate.py for real numbers"`) so the frontend can be built before the real run. Never show
its numbers as results.

## Metrics (`metrics.py`)

1. **format_valid**: `prompts.parse_listing` succeeds and every field is non-empty.
2. **constraint_compliance**: passes all rules at once: title length, exactly 5 bullets, 1-13 tags
   each <= 20 chars, 2-4 description paragraphs, 3-6 word search query. Each rule is also reported
   on its own (`title_length_ok`, `bullets_exactly_5`, `tags_ok`, `description_paragraphs_ok`,
   `search_query_ok`). Checked on the raw JSON, because `parse_listing` quietly drops extra tags.
3. **attribute_coverage**: share of attribute values (product type, colours, material, pattern,
   style) that appear in title + description. The references themselves score about 0.70, so that
   is roughly the ceiling for "copying the real-listing style".
4. **rouge_l**: ROUGE-L F1 (rouge-score, stemmed) of title+description+bullets vs the reference.
5. **judge**: Gemma (`gemma-4-26b-a4b-it`) sees the attributes and both listings as "A" and "B" in a
   random, seeded order and picks the better one (or TIE). If only one output is valid JSON it
   wins without a judge call; if both are invalid it is a tie (`auto_decided` counts these).
   Invalid outputs score 0 on the other metrics.

## Run the fine-tuned model locally (Ollama, 8GB M1)

### Why the LoRA must be merged

Tinker gives us a LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`), not a full model.
**Ollama 0.34.1 and newer refuse LoRA adapters**: `ADAPTER` in a Modelfile fails with
"LoRA adapters are no longer supported" (see `server/create.go` in ollama/ollama; this laptop has 0.34.4).
Ollama also stopped converting safetensors to GGUF itself. So `export_to_ollama.py` bakes the LoRA into
ONE quantised GGUF file with llama.cpp, and `ollama create` just wraps that file.

### Two export paths

| | `--mode adapter` (default) | `--mode merge` |
|---|---|---|
| how | LoRA -> GGUF LoRA (`convert_lora_to_gguf.py`), then `llama-export-lora` merges it into `Qwen/Qwen3-8B-GGUF` **Q8_0**, then `llama-quantize` -> Q4_K_M | cookbook `weights.build_hf_model` merges into the **bf16** HF model (shard by shard), then `convert_hf_to_gguf.py` -> bf16 GGUF -> Q4_K_M |
| download | LoRA + 8.7 GB base GGUF | LoRA + 16.4 GB HF base |
| peak extra disk | ~37 GB, **~27 GB with `--cleanup`**, ~23 GB with `--cleanup --base-gguf ollama` | ~60 GB (~50 GB with `--cleanup`) |
| peak RAM | ~2-3 GB (one tensor at a time) | ~5-6 GB with the cookbook's shard merge (a plain transformers+peft merge of 8B needs ~18 GB: do not do that on 8 GB) |
| time on M1 | ~20-40 min (mostly the 8.7 GB download + writing 16 GB) | ~45-90 min |
| exactness | merge happens on Q8_0 weights (dequantised to f32), tiny extra error before the final Q4_K_M | exact bf16 merge, best quality |

The script checks free disk before starting and stops with these options if it is short.
Steps skip themselves when their output exists, so a stopped run resumes. Partial files are written as
`*.partial.gguf` and renamed only when complete. `--cleanup` deletes each big intermediate once used.

**Caveats**
- The base GGUF must be **the same model the LoRA was trained on**: `Qwen/Qwen3-8B` (the hybrid-thinking
  release), not Qwen3-8B-Base, not Qwen3-4B/2507 variants. Wrong base = garbage output.
- Merging into an already quantised base adds error. Q8_0 is close to bf16 (default). `--base-gguf ollama`
  reuses the `qwen3:8b` blob you already pulled (Q4_K_M: saves 8.7 GB of disk, but it is quantised twice,
  so expect slightly worse quality - use it only if disk is the blocker).
- `--ollama-adapter` (FROM qwen3:8b + ADAPTER lora.gguf, no merge) only works on Ollama <= 0.34.0, and
  even there the LoRA is applied on top of a Q4_K_M base, not the bf16 weights it was trained on.
- Only the 4.9 GB Q4_K_M model is needed at run time; with `qwen3:8b` (5.2 GB) and `gemma3:4b` (3.3 GB)
  for vision, Ollama needs ~13.5 GB in `~/.ollama`. You can `ollama rm qwen3:8b` after the local eval.

**Disk on this laptop:** about 13 GB was free when this was written - not enough for either path.
Options: free ~20 GB; or `--export-dir /Volumes/<external-drive>/listsmart-export`; or run the export on
Colab/Kaggle (Linux, ~100 GB disk) and bring back only the result.

**Easiest: open [`colab_export.ipynb`](colab_export.ipynb) in Google Colab** (CPU runtime is enough). It gets the code
(git clone or zip upload), builds llama.cpp, asks for `TINKER_API_KEY`, runs
`export_to_ollama.py --cleanup --skip create --tinker-path ...`, copies the Q4_K_M `.gguf` + `Modelfile` to Google Drive,
and its last cell lists the Mac commands (`ollama create listsmart-writer -f finetune/export/Modelfile`).
`python3 scripts/doctor.py` (from the project root) then checks that the model is there. The same thing by hand:

```bash
# on Colab: clone the repo, then
cd ListSmart/finetune && pip install -r requirements.txt
export TINKER_API_KEY=...
python export_to_ollama.py --cleanup --steps setup,download,peft,lora_gguf,base_gguf,merge_gguf,quantize,modelfile
# copy export/listsmart-writer-Q4_K_M.gguf (~5 GB) + export/Modelfile to finetune/export/ on the laptop, then:
python export_to_ollama.py --steps modelfile,create      # = ollama create listsmart-writer -f export/Modelfile
# Ollama keeps its own copy, so the .gguf can be deleted afterwards
```

### Downloading the weights from Tinker (real API, tinker 0.34.0 / tinker-cookbook 0.5.7)

- Sampler weights (`tinker://<run>/sampler_weights/<name>`, what `train.py` saves):
  `tinker_cookbook.weights.download(tinker_path=..., output_dir=...)`, which calls
  `RestClient.get_checkpoint_archive_url_from_tinker_path(path)` -> signed URL to a `.tar` with
  `adapter_model.safetensors` + `adapter_config.json`. CLI equivalent: `tinker checkpoint download <path>`.
  The SDK marks this as **deprecated** ("the server may already reject it for your organization").
- The replacement, used when available: `TrainingClient.save_weights_external(name)` (now called at the end
  of `train.py`, path saved in `results/external_weights.txt`) and `RestClient.get_external_weights_urls(path)`
  -> one signed URL per file. `export_to_ollama.py` prefers this path, and handles both a LoRA and a full
  model coming back. This only works for runs trained with the new `train.py`.
- The PEFT conversion uses the cookbook's `weights.build_lora_adapter`. It normally downloads the whole base
  model just to read weight names; we give it a tiny "skeleton" (real `config.json` + weight names read from
  the safetensors headers on HuggingFace, a few KB), so no 16 GB download is needed in adapter mode.

### Local eval output

`evaluate.py --backend ollama` sends each prompt to `POST /api/generate` with `raw: true`, the prompt from
`chat_template.build_raw_prompt`, `stop` = `STOP_STRINGS`, temperature 0, seed 1234, `num_ctx` 4096.
It runs all base rows first, unloads that model, then the fine-tuned rows (two 8B models do not fit in 8 GB).
Per call it records wall latency, Ollama's `eval_count / eval_duration` tokens/s and token counts
(saved under `"call"` in `eval_outputs.jsonl`). The summary adds:

```json
"speed": {"base": {"median_latency_s": 41.2, "tokens_per_s": 9.1, "...": ""},
          "finetuned": {"median_latency_s": 30.5, "tokens_per_s": 9.3}, "hardware": "macOS arm64, Apple M1, 8 GB RAM"},
"cost_per_1000_listings_inr": {"local": 0, "tinker_hosted": 38.4, "closed_api": null, "assumptions": {}}
```

(numbers above are illustrative). `tinker_hosted` = measured tokens per listing x $0.60 / 1M tokens x 84 INR/USD
(same assumptions as `backend/app/costs.py`; change with `--tinker-usd-per-1m`, `--usd-to-inr`). `closed_api`
is filled only by `--closed-baseline` (default models `gpt-4o-mini` / `claude-haiku-4-5` with assumed list
prices, override with `--closed-model`, `--closed-usd-per-1m-in/out`). The Tinker eval also writes `speed`
(hardware "Tinker (hosted ...)"); a later Ollama run moves that to `speed_tinker` and puts the laptop numbers
in `speed`, plus its metrics under `local_eval`.

Time on an 8 GB M1: expect roughly 6-12 tokens/s and ~30-60 s per listing once the model is loaded (close
other apps; the first call also loads ~5 GB). `--limit 20` x 2 models is about 20-40 minutes.

## Time and cost (rough estimates, check the Tinker console)

- Training: ~3.4M tokens (1880 examples x ~600 tokens x 3 epochs) plus ~0.6M tokens of val
  forward passes -> about **$2** on Qwen3-8B. Usually 15-40 minutes depending on queue.
- Eval sampling: 240 generations (120 x 2 models), ~0.3M tokens -> well under **$1**, a few minutes.
- Judge: 120 Gemma calls at ~2 s each (rate limit sleep) -> about **5 minutes**, free tier.

## What to watch for

- `val_loss` at step 0 vs the end (in `train_log.jsonl` / `train_info.json`): it should clearly
  drop. If val loss starts rising while train loss keeps falling, it is overfitting: try
  `--epochs 2` or a lower `--lr`.
- Train loss stuck or NaN: lower `--lr` (try 2e-4).
- Base model outputs with prose around the JSON still parse (we strip text around the `{...}`),
  so format_valid failures mean truly broken JSON or missing fields.
- Judge 429 / quota errors: raise `--judge-sleep`, or run `--no-judge` first and later
  `--reuse-outputs` to judge without re-sampling.
- Temperature is 0 by default, so re-running gives (almost) the same outputs.
- Reference listings come from Amazon, so some still mention sizes or details that are not in the
  attributes. That is a known limit of using real listings as targets.
