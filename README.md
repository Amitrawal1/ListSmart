# ListSmart

**Snap a few product photos, get a ready-to-post Meesho or Etsy listing and a smart price.**

Small Indian sellers spend a lot of time writing titles, descriptions and tags for every
product, and they often guess the price. ListSmart does it in one click:

1. **Gemma** (vision) looks at 1–5 photos and pulls out product attributes (type, colour, fabric, pattern, features).
2. A **listing writer fine-tuned with LoRA on Tinker** turns those attributes into a title, description,
   5 bullet points and tags that follow the platform's rules (Meesho title ≤ 100 chars, Etsy ≤ 140, tags ≤ 20 chars).
3. **SerpApi** (Google Shopping, India) fetches competitor prices, removes outliers and suggests a price
   with a plain-English reason.
4. Everything is deployed as one web service on **Render**.

We also **measure** the fine-tune: base model vs fine-tuned model on a held-out test set
(format validity, rule compliance, attribute coverage, ROUGE-L and a blind LLM-judge win rate).
The results show up in the app's "Base vs Fine-tuned" panel.

## Architecture

```
                 Browser (frontend/: plain HTML + CSS + JS, no build step)
                   |  POST /api/generate (photos, platform, notes, cost price)
                   |  GET  /api/health   GET /api/eval
                   v
  +-------------------------------- FastAPI (backend/app) -------------------------------+
  |                                                                                      |
  |   photos --> vision.py --------> writer.py ------------------> pricing.py ---------> JSON
  |              Gemma 3 27B         WRITER_BACKEND=               SerpApi              response
  |              (Google AI Studio)    tinker : fine-tuned LoRA    google_shopping, gl=in
  |              -> attributes JSON    gemma  : base Gemma         IQR outlier removal
  |                                    mock   : offline            -> suggested price
  |                                  prompts.py (one shared prompt format)                |
  |   serves frontend/ at "/" and "/static/*"                                             |
  +--------------------------------------------------------------------------------------+
                                       ^
                                       | same prompts.py
  finetune/ : build dataset -> LoRA train on Tinker -> evaluate base vs fine-tuned
              -> finetune/results/eval_summary.json  (read by GET /api/eval)

  Deploy: Render web service (render.yaml)
```

Every external service has an offline **mock** path, so the whole app runs with zero API keys.

## Folder layout

```
ListSmart/
  SPEC.md        shared contract (API shapes, data formats)
  README.md      this file
  render.yaml    Render Blueprint
  .env.example   all environment variables
  backend/       FastAPI app (Python 3.11+)
  finetune/      Tinker dataset prep, training, evaluation, results/
  frontend/      index.html, styles.css, app.js
```

## Run locally

Needs Python 3.11+.

### 1. Zero keys (mock mode)

```bash
cd ListSmart
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # leave every key empty, WRITER_BACKEND=mock
cd backend && uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The badge at the top right shows each service as **mock**.
You get a realistic sample listing and sample competitor prices for any photo.

### 2. With real keys

Fill in `.env` (see `.env.example` for comments):

| Variable | What it does | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemma vision + base writer + LLM judge | https://aistudio.google.com/apikey |
| `GEMMA_MODEL` | Gemma model name (default `gemma-3-27b-it`) | – |
| `SERPAPI_KEY` | Google Shopping competitor prices | https://serpapi.com/manage-api-key |
| `WRITER_BACKEND` | `tinker`, `gemma` or `mock` | – |
| `TINKER_API_KEY` | Sample the fine-tuned writer | Tinker console |
| `TINKER_MODEL_PATH` | `tinker://...` checkpoint printed by the training script | `finetune/` |
| `TINKER_BASE_MODEL` | Base model the LoRA was trained on | `finetune/` |

Then run the backend the same way:

```bash
cd backend && uvicorn app.main:app --reload
```

Any key you leave empty just switches that step back to mock mode. The health badge tells you which.

### Frontend only (no backend)

Open `frontend/index.html` directly in a browser, or serve the folder and add `?demo=1`:

```bash
cd frontend && python3 -m http.server 5500
# open http://127.0.0.1:5500/?demo=1
```

Demo mode uses a built-in sample response, so you can try the whole UI without Python dependencies.
You can also add `?demo=1` to the deployed URL.

## Fine-tuning with Tinker

All dataset, training and evaluation steps (with exact commands) are in
[finetune/README.md](finetune/README.md). In short: build (attributes -> listing) pairs from real
product listings, train a LoRA on Tinker with the same prompt as `backend/app/prompts.py`, then run
the evaluation script. It writes `finetune/results/eval_summary.json`, which the app shows in the
"Base vs Fine-tuned" panel. Set `WRITER_BACKEND=tinker` and `TINKER_MODEL_PATH` to use the fine-tuned writer.

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
4. `WRITER_BACKEND` defaults to `gemma`. After fine-tuning, change it to `tinker` in the service's
   **Environment** tab.
5. Commit `finetune/results/eval_summary.json` so the deployed app can show the measured results.

The Blueprint uses the `free` plan. Free services sleep when idle, so open the URL a minute
before a demo to wake it up.

## Demo script for judges (about 3 minutes)

1. **Open the app.** Point at the badge at the top right: it shows which services are live
   (Gemma, SerpApi, Tinker) and which writer is in use.
2. **Upload 2–3 photos** of a product (for example a kurti: front view plus a fabric close-up).
   Show drag & drop, the previews and the remove button.
3. **Choose Meesho**, type a short note ("pure cotton, sizes S–XXL"), and enter a cost price
   such as ₹250.
4. **Click "Generate listing".** Narrate the steps: Gemma reads the photos, the fine-tuned writer
   drafts the listing, SerpApi checks competitor prices.
5. **Walk through the result:**
   - The attribute chips: what Gemma saw in the photos.
   - The title counter (stays within Meesho's 100 characters). Switch to **Etsy** to show the 140 limit.
   - Edit a bullet, add a tag, show a long tag being flagged in red, then **Copy all**.
6. **Pricing panel:** the big suggested price in ₹, the reason in plain English, the dot plot
   of competitor prices with our price marked, and the list of real competitors with links.
   Mention the cost-price floor (at least 25% margin).
7. **Base vs Fine-tuned panel:** the metric table with deltas, the blind LLM-judge win rate,
   and one test example side by side. This is the measured proof that the Tinker fine-tune helps.
8. **Backup plan:** if Wi-Fi or a key fails, add `?demo=1` to the URL for the full offline demo.
