*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12)*

## What I Built

**ListSmart** turns a few product photos into a ready-to-post **Meesho or Etsy listing** (title, description, 5 bullet points and search tags) and suggests a **price** based on what competitors charge.

**The problem:** small sellers in India who sell on Meesho, Instagram shops or Etsy write every listing by hand, often in their second language, late at night after packing orders. A seller with 50 products spends hours on titles and tags, and still guesses the price.

**What ListSmart does for them:**
- 📸 **Reads the photos.** An open vision model (Gemma) finds the product type, colour, material, pattern and key details.
- ✍️ **Writes the listing.** A **fine-tuned Qwen3-8B** writes it within each platform's rules: Meesho titles ≤ 100 characters, Etsy ≤ 140, tags ≤ 20 characters, exactly 5 bullets.
- 💰 **Suggests a price.** It pulls competitor prices from Google Shopping (SerpApi), drops outliers, and suggests a price just under the market median that never goes below the seller's cost + 25%. It explains why in plain words.
- 🔒 **Private mode.** Everything runs on the seller's own laptop, so photos never leave the device and each listing costs **₹0**.
- ✏️ **Editable and ready to paste.** Every field is editable, with live character counters and one-click copy.

**Who would pay:** sellers who list many products every week. The plan is 20 free listings, then about **₹199/month**, less than an hour of a freelancer's time. I'm still testing that price with real sellers.

## Demo

The code and setup steps are on GitHub (below). To try it locally:

```bash
git clone https://github.com/Amitrawal1/ListSmart.git
cd ListSmart/backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8001
# open http://localhost:8001  (add ?demo=1 to try it with sample data, no keys needed)
```

Every external service has an offline fallback, so the app runs with **zero API keys**. Run `python3 scripts/doctor.py` to see what's live and what to set up next.

## Code

{% github Amitrawal1/ListSmart %}

```
backend/    FastAPI: Gemma vision -> fine-tuned writer -> SerpApi pricing; also serves the frontend
finetune/   data prep, Tinker LoRA training, evaluation (base vs fine-tuned), export to Ollama + Colab notebook
frontend/   plain HTML/CSS/JS, no build step
scripts/    doctor.py (setup checker), run_v2.sh (data -> train -> eval), fill_submission.py
```

There are 120 automated tests (backend, fine-tune pipeline, setup checker), and every model call has an offline mock.

## How I Built It

**Open models at the core:**

| Job | Open model | Where it runs |
|---|---|---|
| Read the product photos | **Gemma 3 4B** (local), **Gemma 4** (hosted fallback) | laptop via **Ollama**, or Google AI Studio |
| Write the listing | **Qwen3-8B + my LoRA fine-tune** | laptop via **Ollama**, or **Tinker** |
| Clean the training data | **Qwen3.6-35B** as a "teacher" | **Tinker** |
| Judge base vs fine-tuned | **Gemma 4 26B**, blind A/B | Google AI Studio |

**Pipeline:** photos → Gemma pulls out product details as JSON → the fine-tuned writer turns them into listing JSON → SerpApi pulls competitor prices → a simple, explainable pricing rule. The app and the training code share a single prompt file, so the model sees the exact same format in training and in the app.

**Fine-tuning on Tinker:** a LoRA (rank 32) on `Qwen/Qwen3-8B` with "thinking" turned off. It's trained on product details → listing pairs built from **2,100 good real product listings** (Amazon Reviews 2023 metadata: clothing, home and handmade categories, 4+ star ratings). Loss is counted only on the listing the model writes, not on the prompt. **120 test products** are held out and never used in training.

### My first fine-tune lost, and that's where the real work was

Version 1 trained for 3 epochs (~$2 on Tinker). Validation loss dropped from **2.61 to 1.48**, which looked great. Then I measured it against the untrained base model on the 120 test products:

| Metric | Base Qwen3-8B | Fine-tune v1 |
|---|---|---|
| Valid JSON listing | **99.2%** | 84.2% |
| Follows platform rules | **85.8%** | 74.2% |
| Mentions the product's details | **89.7%** | 59.5% |
| Wording overlap with the real listing (ROUGE-L) | 0.21 | **0.27** |
| Blind Gemma judge preferred it | **77.5%** | 0.8% (tie 21.7%) |

**Lower loss, worse model.** Reading the outputs showed why:
1. **Repetition loops.** 19 of 120 outputs got stuck, e.g. *"10 Pack Door Stopper, 10 Pack Door Stopper, …"*. Real Amazon listings are keyword-stuffed: **38%** of my training targets repeated 4-word phrases a lot, and the model learned it.
2. **Made-up facts.** The real listings contained sizes and details that my simple rule-based extractor never put into the input. The model learned to **invent** facts, like *"10.5 x 10.5 inches"* for napkins and *"Kanchipuram silk"* for a plain maroon fabric photo.

It copied the real listings' wording better (higher ROUGE-L) but became less honest and less reliable. That's exactly the wrong trade for a seller.

### The fix: v2 with grounded data (in progress)

- A bigger open model, **Qwen3.6-35B on Tinker**, re-reads every original listing, pulls out the **true** product details, and rewrites each training target using **only** those details: no repetition, exactly 5 bullets, platform limits enforced.
- Every row is checked automatically (valid format, no repeated phrases, ≥ 80% of the details mentioned, no numbers that aren't in the input, no brand names), and rows that fail are dropped.
- The 120 test products stay exactly the same, so the comparison is fair.
- Training drops to 2 epochs: v1's validation loss was lowest around epoch 1.7.
- The eval samples both models at the app's real temperature (0.7) instead of 0.

A 60-row sample of the cleaned data cost **$0.11**, and the outputs were clean and specific: *"set of 12 · 2.16 in × 2.16 in × 2.60 in · for votive or tea light candles"* instead of *"MEASUREMENTS/MATERIAL: Each one measures 2"*. The full v2 run (about $6 in total) is next, and I'll update this post with the measured results, whether it wins or not.

### Running on a laptop

Recent Ollama versions no longer accept LoRA adapters, so the export script bakes the LoRA into one ~5 GB **Q4_K_M GGUF** with **llama.cpp** and loads it with `ollama create listsmart-writer`. My laptop is an 8 GB M1 with only 12 GB of free disk, so I build the GGUF on free Google Colab (the notebook is in the repo) and download only the final file.

### Built with a team of AI agents

I built this with **Claude Code** working as several agents in parallel. One wrote a shared spec, then three agents built the backend, the fine-tune/eval pipeline and the frontend at the same time against it. Later agents did a bug review (it found that the Render deploy would have silently skipped the fine-tuned model) and built the data-cleaning step. I ran every training and eval job myself and made the calls on cost and direction.

## Why Does Open Innovation Matter?

- **I could fix the model, not just the prompt.** When v1 lost, I could open every training example, find the keyword stuffing and the missing facts, rebuild the data and retrain. With a closed API, all I could have done is tweak the prompt and hope.
- **The seller's photos stay on their laptop.** In private mode, Gemma and the fine-tuned Qwen run in Ollama on the seller's own machine. Unreleased product photos and cost prices never reach someone else's server.
- **₹0 per listing makes the business work.** Local listings cost nothing. The same writer hosted on Tinker cost about **₹0.03** per listing in my live test. A ₹199/month plan stays profitable even for a seller who lists hundreds of products, because there's no API bill that grows with usage.
- **It works offline.** Private mode skips the only online step, the competitor price check, so sellers with patchy internet can still write listings.
- **No lock-in.** The base model, the LoRA and the GGUF are files I own. During this build Tinker retired several models (Qwen3-235B, Llama-3.3-70B) and Google's API stopped serving the Gemma 3 model I'd started with. Because the stack is open and the model names are just settings, switching took minutes, not a rewrite.

## Prize Categories

- **Gemma**: reads the product photos (Gemma 3 locally, Gemma 4 hosted) and judges the blind A/B eval.
- **Tinker**: LoRA fine-tune of Qwen3-8B, a teacher model for grounded data, and a measured base-vs-fine-tuned eval (v1 reported honestly above).
- **SerpApi**: Google Shopping competitor prices behind the price suggestion.

<!-- Thanks for participating! -->
