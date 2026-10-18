# ListSmart

**Snap a few product photos, get a ready-to-post Meesho or Etsy listing and a smart price —
with open AI models running on your own laptop.**

Small Indian sellers spend a lot of time writing titles, descriptions and tags for every
product, and they often guess the price. ListSmart does it in one click:

1. **Gemma 3** (open-weight vision model, `gemma3:4b` via Ollama) looks at 1–5 photos and pulls out
   product attributes (type, colour, fabric, pattern, features).
2. **Our fine-tuned writer** — open **Qwen3-8B** + a LoRA we trained on Tinker, merged and run
   locally with Ollama as `listsmart-writer` — turns those attributes into a title, description,
   5 bullet points and tags that follow the platform's rules (Meesho title ≤ 100 chars, Etsy ≤ 140, tags ≤ 20 chars).
3. **SerpApi** (Google Shopping, India) fetches competitor prices, removes outliers and suggests a price
   with a plain-English reason. This is the **only** online step, and **Private mode** skips it.
4. Hosted fallbacks (Gemma API, Tinker sampling) and a zero-key mock mode keep it working anywhere,
   including one web service on **Render**.

**Why local-first:** photos never leave the seller's laptop, each listing costs ₹0, and the
fine-tuned model is a file they own (no lock-in). Target machine: an 8 GB M1 MacBook.

We also **measure** the fine-tune: base model vs fine-tuned model on a held-out test set
(format validity, rule compliance, attribute coverage, ROUGE-L, a blind LLM-judge win rate,
plus speed on a laptop and cost per 1,000 listings). The results show up in the app's
"Base vs Fine-tuned" panel.

More docs:
- [docs/FIRST_USER.md](docs/FIRST_USER.md) — playbook for getting 1–3 real sellers to try it this week
- [docs/SUBMISSION.md](docs/SUBMISSION.md) — draft hackathon post (fill the `{{placeholders}}` from real results)
- [finetune/README.md](finetune/README.md) — dataset, LoRA training, evaluation, building `listsmart-writer`
- [SPEC.md](SPEC.md) — API and data contract

## Architecture

```
                 Browser (frontend/: plain HTML + CSS + JS, no build step)
                   |  POST /api/generate (photos, platform, notes, cost price, offline)
                   |  POST /api/feedback   GET /api/feedback/summary
                   |  GET  /api/health     GET /api/eval
                   v
  +----------------------------- FastAPI (backend/app), on your laptop ------------------------+
  |                                                                                            |
  |  photos --> vision.py ----------> writer.py -------------------> pricing.py ------------> JSON
  |             VISION_BACKEND=        WRITER_BACKEND=                SerpApi google_shopping
  |              ollama: gemma3:4b      ollama: listsmart-writer      (search text only; skipped
  |              gemma_api (hosted)     tinker / gemma (hosted)        in Private mode)
  |              mock                   mock                          -> suggested price
  |                     \                    |
  |                      +---- Ollama (localhost:11434): open models, nothing leaves the laptop
  |  prompts.py + chat_template.py = one shared prompt format (same as training)
  |  feedback -> data/feedback.jsonl (local file, gitignored)
  +--------------------------------------------------------------------------------------------+
                                       ^
                                       | same prompts.py
  finetune/ : build dataset -> LoRA train on Tinker -> merge + GGUF -> `ollama create listsmart-writer`
              -> evaluate base vs fine-tuned -> finetune/results/eval_summary.json (GET /api/eval)
```

Every external service has an offline **mock** path, so the whole app runs with zero API keys.

## Folder layout

```
ListSmart/
  SPEC.md        shared contract (API shapes, data formats)
  README.md      this file
  docs/          first-user playbook + hackathon submission draft
  render.yaml    Render Blueprint
  .env.example   all environment variables
  backend/       FastAPI app (Python 3.11+)
  finetune/      Tinker dataset prep, training, evaluation, results/
  frontend/      index.html, styles.css, app.js
  data/          feedback.jsonl (created at runtime, not in git)
```

## Run locally (local-first, recommended)

Needs Python 3.11+ and [Ollama](https://ollama.com) (free, open source).

### 1. Get the open models

```bash
ollama pull gemma3:4b        # the photo reader (about 3 GB)
```

For the listing writer, build our fine-tuned model `listsmart-writer` by following
[finetune/README.md](finetune/README.md) (merge the Tinker LoRA into Qwen3-8B, convert to GGUF
Q4_K_M, then `ollama create listsmart-writer ...`). **Until you have built it**, use the open base model:

```bash
ollama pull qwen3:8b
```

### 2. Start the app

```bash
cd ListSmart
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # VISION_BACKEND=ollama and WRITER_BACKEND=ollama are the defaults
cd backend && uvicorn app.main:app --reload
# no listsmart-writer yet? start it like this instead:
#   OLLAMA_WRITER_MODEL=qwen3:8b uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The badge at the top right shows whether Ollama is running and how
many models it has. If a model is missing, a yellow "Finish local setup" box tells you the exact
command to run.

### Private mode

The **"Private mode — runs on this laptop"** switch above the Generate button sends `offline=true`.
Photos and listing writing already run locally with Ollama; the only online step is the competitor
price check (a short search text such as "maroon cotton kurti" sent to SerpApi). Private mode skips
it, so **nothing leaves the device**. The choice is remembered in your browser.
`OFFLINE_MODE=true` in `.env` makes private mode the server default.

After every result, a small strip shows: **Ran locally ✓**, **Data that left this device**
(none, or "SerpApi: search query only"), **Cost** (₹0.00 for local) and how long each step took.

### Feedback

Under each result, sellers can click 👍/👎, "I used this listing" (also sent automatically when
they click **Copy all**), and answer "Would you pay for this?" (yes / maybe / no, a fair monthly
price in ₹, and what to build next). Each answer is one line in `data/feedback.jsonl` with the
request id and the listing as the seller edited it. The footer shows a running count from
`GET /api/feedback/summary`. See [docs/FIRST_USER.md](docs/FIRST_USER.md) for how to read the file.

### Zero keys, no Ollama (mock mode)

Set `VISION_BACKEND=mock` and `WRITER_BACKEND=mock` in `.env` (or just leave Ollama off — the
app falls back to hosted, then mock). You get a realistic sample listing and sample competitor prices.

### Hosted keys (optional fallbacks)

Fill in `.env` (see `.env.example` for comments):

| Variable | What it does | Where to get it |
|---|---|---|
| `VISION_BACKEND` | `ollama` (default), `gemma_api` or `mock` | – |
| `WRITER_BACKEND` | `ollama` (default), `tinker`, `gemma` or `mock` | – |
| `OLLAMA_HOST` | Ollama address (default `http://localhost:11434`) | – |
| `OLLAMA_VISION_MODEL` | default `gemma3:4b` | `ollama pull` |
| `OLLAMA_WRITER_MODEL` | default `listsmart-writer`; `qwen3:8b` for the base model | `finetune/` |
| `OFFLINE_MODE` | `true` = never call SerpApi | – |
| `GOOGLE_API_KEY` | Hosted Gemma vision + base writer + LLM judge | https://aistudio.google.com/apikey |
| `GEMMA_MODEL` | Hosted Gemma model name (default `gemma-3-27b-it`) | – |
| `SERPAPI_KEY` | Google Shopping competitor prices | https://serpapi.com/manage-api-key |
| `TINKER_API_KEY` | Sample the fine-tuned writer on Tinker | Tinker console |
| `TINKER_MODEL_PATH` | `tinker://...` checkpoint printed by the training script | `finetune/` |
| `TINKER_BASE_MODEL` | Base model the LoRA was trained on | `finetune/` |

Any key you leave empty just switches that step to the next fallback. The health badge tells you which.

### Frontend only (no backend)

Open `frontend/index.html` directly in a browser, or serve the folder and add `?demo=1`:

```bash
cd frontend && python3 -m http.server 5500
# open http://127.0.0.1:5500/?demo=1
```

Demo mode uses a built-in sample response (including the privacy strip, feedback form and sample
eval numbers, clearly marked as samples), so you can try the whole UI without Python.
You can also add `?demo=1` to the deployed URL.

## Fine-tuning with Tinker

All dataset, training and evaluation steps (with exact commands) are in
[finetune/README.md](finetune/README.md). In short: build (attributes -> listing) pairs from real
product listings, train a LoRA on Tinker with the same prompt as `backend/app/prompts.py`, then run
the evaluation script. It writes `finetune/results/eval_summary.json`, which the app shows in the
"Base vs Fine-tuned" panel. To run the fine-tuned writer **locally**, follow the Ollama export steps
there to build `listsmart-writer` (then `WRITER_BACKEND=ollama`, the default). To sample it on Tinker
instead, set `WRITER_BACKEND=tinker` and `TINKER_MODEL_PATH`. `evaluate.py --backend ollama` compares
`qwen3:8b` and `listsmart-writer` on your own laptop and also records speed and cost per 1,000 listings.

## Deploy on Render

**Important:** this `ListSmart` folder currently sits inside a bigger git repo on the Desktop.
Render deploys a whole repo, so push `ListSmart` as **its own GitHub repo** first (with
`render.yaml` at the repo root):

```bash
cd ListSmart
git init
git add .
git commit -m "ListSmart"
git branch -M main
git remote add origin https://github.com/<you>/listsmart.git
git push -u origin main
```

(Check that `.env` is not committed. Only `.env.example` should be in the repo.)

Then:

1. Render Dashboard -> **New** -> **Blueprint** -> pick the `listsmart` repo.
2. Render reads `render.yaml` and creates one Python web service:
   - build: `pip install -r backend/requirements.txt`
   - start: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - health check: `/api/health`
   - Python 3.11.9
3. Render asks for the secret values (`GOOGLE_API_KEY`, `SERPAPI_KEY`, `TINKER_API_KEY`,
   `TINKER_MODEL_PATH`, `TINKER_BASE_MODEL`). Leave any of them empty to keep that step in mock mode.
4. Render has no Ollama (no GPU, little RAM), so the Blueprint sets the hosted paths:
   `VISION_BACKEND=gemma_api` and `WRITER_BACKEND=tinker` (falls back to `gemma`, then `mock`, if the
   Tinker keys are missing). The deployed app is a **public demo**; the private, ₹0 version is the
   one sellers run on their own laptop. The privacy strip shows the difference honestly.
5. Commit `finetune/results/eval_summary.json` so the deployed app can show the measured results.

The Blueprint uses the `free` plan. Free services sleep when idle, so open the URL a minute
before a demo to wake it up.

## Demo script for judges (about 3 minutes)

Best done on a laptop running locally with Ollama (Wi-Fi can even be off in Private mode).

1. **Open the app.** Point at the badge at the top right: "Ollama on · 3 models" means the open models
   are on this laptop. The other pills show hosted fallbacks (Gemma API, SerpApi, Tinker).
2. **Turn on "Private mode — runs on this laptop".** Explain: the only online step is the competitor
   price search, and private mode skips it.
3. **Upload 2–3 photos** of a product (for example a kurti: front view plus a fabric close-up).
   Show drag & drop, the previews and the remove button.
4. **Choose Meesho**, type a short note ("pure cotton, sizes S–XXL"), and enter a cost price such as ₹250.
5. **Click "Generate listing".** The steps read "Reading photos with Gemma 3 on your laptop…",
   "Writing listing with our fine-tuned Qwen3 on your laptop…", "Skipping competitor prices…".
6. **Privacy strip:** "Ran locally ✓ · Data that left this device: none · Cost: ₹0.00" plus the time per step.
7. **Walk through the result:**
   - The attribute chips: what Gemma saw in the photos.
   - The title counter (stays within Meesho's 100 characters). Switch to **Etsy** to show the 140 limit.
   - Edit a bullet, add a tag, show a long tag being flagged in red, then **Copy all**
     (this also records "I used this listing").
8. **Turn Private mode off and generate again** to show live competitor prices: the dot plot, the
   suggested price with a plain-English reason, and the strip now says "SerpApi: search query only (no photos)".
9. **Feedback:** click 👍 and fill in "Would you pay for this?" — the footer counter goes up.
10. **Base vs Fine-tuned panel:** the metric table with deltas, the blind LLM-judge win rate, speed on a
    laptop, cost per 1,000 listings, and one test example side by side.
11. **Backup plan:** if something fails, add `?demo=1` to the URL for the full offline demo (sample data).
