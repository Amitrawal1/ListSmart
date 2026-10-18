# ListSmart fine-tune + evaluation (Tinker)

This folder fine-tunes the ListSmart **listing writer** with LoRA on Tinker, using real,
well-rated product listings, and then **measures** the fine-tuned model against the base
model on the same held-out prompts (the Tinker sponsor requirement).

```
finetune/
  prepare_data.py   download + filter good listings -> data/{train,val,test}.jsonl
  train.py          LoRA SFT on Tinker, loss only on the assistant answer
  evaluate.py       base vs fine-tuned on the test set -> results/eval_*.{json,jsonl,md}
  metrics.py        pure metric functions (tested in tests/)
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
GEMMA_MODEL=gemma-3-27b-it   # optional, judge model
HF_TOKEN=...              # optional, only if HuggingFace rate-limits the download
```

## Commands, in order

```bash
# 0. Offline checks (no keys needed, a few seconds each)
python -m pytest -q tests
python prepare_data.py --dry-run
python train.py --dry-run          # tokenises the data, shows exactly which tokens get loss
python evaluate.py --dry-run       # mock outputs -> results/dry_run/ (never read by the app)

# 1. Build the dataset (~15 s). Already done once; re-run to rebuild data/*.jsonl
python prepare_data.py

# 2. Fine-tune on Tinker (needs TINKER_API_KEY)
python train.py
#    quick smoke test first if you like (5 steps, a few cents):
#    python train.py --max-steps 5 --eval-every 0 --name listsmart-smoke

# 3. Evaluate base vs fine-tuned (needs TINKER_API_KEY, and GOOGLE_API_KEY for the judge)
python evaluate.py
#    without the judge:      python evaluate.py --no-judge
#    judge again later:      python evaluate.py --reuse-outputs
```

Then point the app at the new model (in `ListSmart/.env`):

```
WRITER_BACKEND=tinker
TINKER_MODEL_PATH=<contents of finetune/results/finetuned_model.txt>
TINKER_BASE_MODEL=Qwen/Qwen3-8B
```

The app must sample with the same renderer (`qwen3_disable_thinking`, saved in
`results/train_info.json`), otherwise the prompt tokens differ from training.

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
5. **judge**: Gemma (`gemma-3-27b-it`) sees the attributes and both listings as "A" and "B" in a
   random, seeded order and picks the better one (or TIE). If only one output is valid JSON it
   wins without a judge call; if both are invalid it is a tie (`auto_decided` counts these).
   Invalid outputs score 0 on the other metrics.

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
