<!--
DEV post for the Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle.
Title:  ListSmart: product photos in, Meesho/Etsy listing out, on your own laptop
Tags:   #devchallenge #hf26challenge #opensource #ai
Cover:  1000x420 screenshot of the app with a finished listing + price
Before posting: replace every [FILL: ...] with real results (run `bash scripts/run_v2.sh`,
then `python3 scripts/fill_submission.py` prints the numbers). Delete any section you can't fill honestly.
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12)*

## What I Built

**ListSmart** turns a few product photos into a ready-to-post **Meesho or Etsy listing** (title, description, 5 bullet points and search tags) and suggests a **price** based on what competitors charge.

**Who it's for:** small sellers in India who list on Meesho, Instagram shops or Etsy. Many of them write every listing by hand, in a second language, at night after packing orders. A seller with 50 products spends hours on titles and tags, and still guesses the price.

**What it does for them:**
- 📸 **Reads the photos.** An open vision model (Gemma) finds the product type, colour, material, pattern and key details.
- ✍️ **Writes the listing.** Our **fine-tuned Qwen3-8B** writes it within each platform's rules (Meesho titles ≤ 100 characters, Etsy ≤ 140, tags ≤ 20 characters, exactly 5 bullets).
- 💰 **Prices it.** It pulls competitor prices from Google Shopping (SerpApi), drops outliers, and suggests a price just under the market median that never goes below the seller's cost + 25%.
- 🔒 **Keeps it private.** In private mode everything runs on the seller's laptop, so photos never leave the device and each listing costs **₹0**.

**Who would pay:** sellers who list many products a week. The idea is the first 20 listings free, then about **₹199/month**, less than one hour of a freelancer's time. (That price is a guess I'm still testing with sellers; see *My First User*.)

## Demo

- 🌐 Live demo: [FILL: Render link]
- 🎥 Video (2 min): [FILL: video link]

To try it without any keys, open the app with `?demo=1` to use built-in sample data.

[FILL: 2-3 screenshots: upload → listing → price chart → "Base vs Fine-tuned" panel]

## Code

{% github Amitrawal1/ListSmart %}

```
backend/    FastAPI app: Gemma vision -> fine-tuned writer -> SerpApi pricing; serves the frontend
finetune/   data prep, Tinker LoRA training, evaluation (base vs fine-tuned), export to Ollama
frontend/   plain HTML/CSS/JS, no build step
scripts/    doctor.py (setup checker), run_v2.sh (data -> train -> eval), fill_submission.py
```

## How I Built It

**Open models at the core:**

| Job | Open model | Where it runs |
|---|---|---|
| Read the photos | **Gemma 3 4B** (local, Ollama) / Gemma 4 (hosted fallback) | laptop, or Google AI Studio |
| Write the listing | **Qwen3-8B + our LoRA fine-tune** | laptop (Ollama), or Tinker |
| Clean the training data | **Qwen3.6-35B** as a teacher | Tinker |
| Judge the eval | **Gemma 4 26B**, blind A/B | Google AI Studio |

**Pipeline:** photos → Gemma pulls out product details as JSON → the fine-tuned writer turns them into listing JSON → SerpApi pulls competitor prices → a simple, explainable pricing rule.

**Fine-tuning on Tinker:** a LoRA (rank 32) on `Qwen/Qwen3-8B` with "thinking" turned off, trained on product details → listing pairs built from **2,000+ good real product listings** (Amazon Reviews 2023 metadata: clothing, home and handmade categories, rated 4+ stars). Loss is counted only on the listing the model writes, not on the prompt. A fixed set of **120 test products** is never used in training.

**Honest part: my first fine-tune lost.** Version 1 trained for 3 epochs (~$2 on Tinker; val loss fell from 2.61 to 1.48). Then I measured it against the untrained base model on the 120 test products:

| Metric | Base Qwen3-8B | Fine-tune v1 |
|---|---|---|
| Valid JSON listing | 99.2% | 84.2% |
| Follows platform rules | 85.8% | 74.2% |
| Mentions the product's details | 89.7% | 59.5% |
| Blind judge win rate | **77.5%** | 0.8% (21.7% tie) |

Lower loss, worse model. I read the outputs and found two problems:
1. **Repetition loops.** 19 of 120 outputs looped, e.g. "10 Pack Door Stopper, 10 Pack Door Stopper, …". Real Amazon listings are keyword-stuffed: 38% of my training targets repeated 4-word phrases a lot.
2. **Made-up facts.** The training targets had sizes and details that my simple rule-based extractor never put into the input, so the model learned to **invent** facts like "10.5 x 10.5 inches" and "Kanchipuram silk".

**The fix (v2):** a bigger open model (**Qwen3.6-35B** on Tinker) re-read every original listing, pulled out the true details, and rewrote each target using **only** those details: no repetition, exactly 5 bullets, platform limits enforced. Every row is checked automatically, and weak rows are dropped. A 60-row sample cost **$0.11**, and the full rebuild about $4. Then: 2 epochs (v1's val loss was lowest around epoch 1.7), and the eval runs at the app's real temperature (0.7) for both models.

**v2 results (same 120 test products, same prompts):**

| Metric | Base Qwen3-8B | Fine-tune v2 |
|---|---|---|
| Valid JSON listing | [FILL] | [FILL] |
| Follows platform rules | [FILL] | [FILL] |
| Mentions the product's details | [FILL] | [FILL] |
| ROUGE-L vs reference | [FILL] | [FILL] |
| Blind judge win rate | [FILL] | [FILL] (tie [FILL]) |

[FILL: one honest sentence on where v2 helped most and where base is still as good or better]

**Running locally:** recent Ollama versions dropped support for LoRA adapters, so the export script bakes the LoRA into one ~5 GB Q4_K_M GGUF with llama.cpp and loads it with `ollama create listsmart-writer`. My Mac has 8 GB of RAM and only 12 GB of free disk, so I built the GGUF on free Google Colab (a notebook is in the repo) and downloaded only the final file.

**Built with Claude Code:** I worked with Claude Code as a team of parallel agents. Three agents wrote the backend, the fine-tune/eval pipeline and the frontend against one shared spec, and later agents did a bug review and built the data-cleaning step. I ran all the training and eval jobs myself.

## Why Does Open Innovation Matter?

- **I could fix the model, not just the prompt.** When v1 lost, I could read every training example, find the keyword stuffing and the missing facts, rebuild the data and retrain. With a closed API I could only have tweaked the prompt and hoped.
- **The seller's photos stay on their laptop.** In private mode Gemma and the fine-tuned Qwen run in Ollama on the seller's own machine. Unreleased product photos and cost prices never go to someone else's server.
- **It costs nothing per listing, so the business can work.** Local listings cost **₹0**. The same writer hosted on Tinker cost about **₹0.03** per listing in my live test. A ₹199/month plan stays profitable even for a seller who lists hundreds of products, with no API bill that grows with usage.
- **It works offline.** Private mode skips the only online step (the price check), so a seller can write listings with patchy internet.
- **No lock-in.** The base model, the LoRA and the GGUF are all files I own. If a provider changes its prices or retires a model (Tinker retired Qwen3-235B and Llama-3.3-70B during this build), I can swap models without rewriting the app.

## My Agent Session

[FILL: DevRelay embed using the agent_session tag, or a link to the saved session]

## My First User

[FILL: or delete this section if nobody has tried it yet. Be honest.]
- **Who:** [FILL: e.g. a Meesho kurti seller in my family / a local shop owner]
- **What they did:** [FILL: listed N products; posted M]
- **What they said:** "[FILL: a quote, used with their permission]"
- **Would they pay?** [FILL: yes / maybe / no, and how much per month]
- **Did they pay?** [FILL: "Not yet" is fine]
- **What they asked for next:** [FILL: e.g. Hindi descriptions, bulk upload]

## Prize Categories

[FILL: check the challenge page and list only the categories that apply, e.g. Gemma (vision + judge), Tinker (fine-tune + measured eval), SerpApi (competitor prices), Render (deploy, only once it's live)]

<!-- Thanks for participating! -->
