# ListSmart — Meesho/Etsy listing writer (fine-tuned)

Seller uploads product photos -> app writes a marketplace listing (title, description,
bullets, tags), pulls competitor prices with SerpApi, and suggests a price.

Sponsors used: **Gemma** (vision: reads photos), **Tinker** (LoRA fine-tune of the
listing writer on good listings + measured eval vs base model — REQUIRED),
**SerpApi** (Google Shopping competitor prices), **Render** (deploy).

## Folder layout
```
ListSmart/
  SPEC.md            this file (shared contract, do not break it)
  MEMORY.md          project history log (how/why decisions were made)
  README.md          setup + run + demo instructions
  render.yaml        Render blueprint (one web service: FastAPI also serves frontend/)
  .env.example       all env vars
  backend/           FastAPI app (Python 3.11+)
  finetune/          Tinker dataset prep, training, eval scripts + results/
  frontend/          static HTML/CSS/JS (no build step), served by backend at "/"
```

## Pipeline
1. `vision.py`  — Gemma (Google AI Studio via `google-genai`, model from env `GEMMA_MODEL`,
   default `gemma-3-27b-it`) looks at 1-5 photos and returns **product attributes JSON**
   (see ProductAttributes). 
2. `writer.py`  — turns attributes + platform + seller notes into a **Listing JSON**.
   Backends chosen by env `WRITER_BACKEND`:
   - `tinker`  : fine-tuned model sampled through Tinker (`TINKER_API_KEY`,
                 `TINKER_MODEL_PATH` = tinker://... checkpoint, `TINKER_BASE_MODEL`)
   - `gemma`   : base Gemma text prompt (fallback / baseline)
   - `mock`    : deterministic offline output (no keys needed, for dev + tests)
3. `pricing.py` — SerpApi `google_shopping` engine (`SERPAPI_KEY`, `gl=in`, `hl=en`,
   currency INR) using the listing's `search_query`. Clean prices, drop outliers (IQR),
   suggest price. No key -> mock competitor list.

The SAME prompt format must be used by finetune/ (training + eval) and backend/writer.py.
It lives in ONE place: `backend/app/prompts.py` (finetune imports it by adding
`../backend` to sys.path). Functions:
- `build_writer_messages(attributes: dict, platform: str, notes: str) -> list[dict]`
  returns chat messages [{"role":"system",...},{"role":"user",...}]
- `parse_listing(text: str) -> dict` robust JSON extraction + validation into Listing

## Data shapes (JSON)
ProductAttributes:
```json
{"product_type":"kurti","category":"Women Ethnic Wear","colors":["maroon"],
 "material":"cotton","pattern":"block print","style":"straight","gender":"women",
 "key_features":["3/4 sleeves","mandarin collar"],"visible_text":"","condition":"new"}
```
Listing:
```json
{"title":"... (<=100 chars Meesho, <=140 Etsy)","description":"2-4 short paragraphs",
 "bullets":["5 bullets"],"tags":["up to 13 tags, each <=20 chars"],
 "search_query":"short query for competitor search"}
```
platform: "meesho" | "etsy"

## HTTP API (backend)
- `GET  /api/health` -> `{"ok":true,"writer_backend":"mock","gemma":bool,"serpapi":bool,"tinker":bool}`
- `POST /api/generate` multipart: `images` (1-5 files, jpg/png/webp, <=8MB each),
  `platform` (meesho|etsy), `notes` (optional str), `cost_price` (optional float, INR)
  -> 
```json
{"attributes":{...},"listing":{...},
 "pricing":{"currency":"INR","competitors":[{"title":"","price":499.0,"source":"","link":"","thumbnail":""}],
   "stats":{"count":10,"min":299,"median":499,"max":899,"p25":399,"p75":599},
   "suggested_price":479,"strategy":"text explaining why","margin_pct":null},
 "meta":{"writer_backend":"mock","latency_ms":1234}}
```
- `GET /api/eval` -> contents of `finetune/results/eval_summary.json` if present (for the
  frontend "Base vs Fine-tuned" panel), else `{"available":false}`.
- `GET /` serves `frontend/index.html`, static files under `/static` -> frontend/.

## Pricing rule (simple, explainable)
suggested = round_to_ending_9(median * 0.95) (undercut median a little); if cost_price given,
floor at cost_price * 1.25; clamp inside [p25, p75] when possible. Explain in `strategy`.

## Fine-tune + eval (finetune/)
- Data: good real listings (e.g. HuggingFace Amazon product metadata, fashion/home/handmade
  categories) -> convert each to (ProductAttributes, Listing) pairs. Attributes are derived
  from the listing text/metadata (so no vision needed at train time). Split train/val/test
  by product, test set >= 100 items, never seen in training.
- Train: LoRA on Tinker with the chat-format prompt from prompts.py.
- Eval: base model vs fine-tuned on test set, SAME prompts. Metrics:
  1. format validity (parse_listing succeeds, all fields present)
  2. constraint compliance (title length, 5 bullets, tag count/length)
  3. attribute coverage (share of attribute values mentioned in title/description)
  4. ROUGE-L vs reference listing
  5. LLM-judge pairwise win-rate (Gemma as judge, randomised order, A/B blind)
- Output `finetune/results/eval_summary.json`:
```json
{"available":true,"base_model":"...","finetuned_model":"tinker://...","n_test":120,
 "metrics":{"format_valid":{"base":0.71,"finetuned":0.98}, "...":{}},
 "judge":{"finetuned_win":0.64,"base_win":0.22,"tie":0.14},
 "examples":[{"attributes":{},"base":{},"finetuned":{},"reference":{}}]}
```

## Rules for all code
- Simple plain-English comments (one line above non-obvious blocks), readable by beginners.
- Every external service must have an offline mock path so the app runs with zero keys.
- Never commit secrets; read keys from env (.env via python-dotenv).
- Do NOT run long training jobs; write scripts and document exact commands.
