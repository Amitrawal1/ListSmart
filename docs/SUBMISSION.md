# ListSmart — photos in, listing out, on your own laptop

> **DRAFT.** Everything in `{{double_braces}}` is a placeholder. Fill it from real results only
> (see "How to fill the placeholders" at the bottom). Do not post until every placeholder is
> replaced or the sentence is removed. No invented numbers, users or quotes.

**One line:** ListSmart turns 2–3 product photos into a ready-to-post Meesho or Etsy listing
(title, description, 5 bullets, tags) and a suggested price — using open models that run on the
seller's own laptop, so photos never leave the device and each listing costs ₹0.

Links: {{demo_video_link}} · {{github_repo_link}} · {{render_demo_link}} · {{devrelay_session_link}}

---

## The problem

India has millions of small online sellers — Meesho resellers, Instagram shops, home-based
kurti, jewellery and handicraft makers. For **every** product they have to:

- write a title that fits the platform's rules (Meesho ≤ 100 characters, Etsy ≤ 140),
- write a description and bullet points in clean English,
- pick tags (Etsy: up to 13, each ≤ 20 characters),
- guess a price without knowing what competitors charge.

Many do this on a phone, late at night, for dozens of products. Listings end up as
"Beautiful Kurti Very Nice Quality Best Price", which hurts search ranking and sales.

General chat apps can help, but sellers must paste photos into a cloud service, pay per use or
per seat, and still fix the format by hand every time.

## Who pays, and why

**User:** a small seller listing 20–200 products a month.

**Why they would pay:** time saved per listing ({{minutes_saved_per_listing}} minutes in our tests —
placeholder, measure it with real sellers), better-formatted listings that follow platform rules, and a
price check against real competitors.

**Pricing idea (to test, not proven):**

| Plan | Price | What you get |
|---|---|---|
| Free | ₹0 | 20 listings, everything runs on your laptop |
| Seller | ₹199 / month | Unlimited listings, live competitor price checks, Hindi descriptions (planned) |
| Reseller / shop | {{price_tier_3}} | Bulk upload from a folder, CSV export for Meesho (planned) |

Why this can work as a side hustle: because the AI runs on the **seller's** laptop, our running cost
per listing is about ₹0. The only per-request cost we would pay is the competitor price search
(SerpApi), which users can turn off with Private mode. So almost all of ₹199 is margin, even at small scale.

What real sellers said about price: {{first_user_price_answers}} (see "First real users" below).

## How it works

```
  Seller's laptop (nothing below leaves it, except the dotted line)
  +--------------------------------------------------------------------------+
  |                                                                          |
  |  2-3 photos --> Gemma 3 4B (open vision model, Ollama)                    |
  |                    |  product attributes JSON                            |
  |                    v  (type, colours, fabric, pattern, features)         |
  |                 listsmart-writer = Qwen3-8B + our LoRA (Ollama, Q4 GGUF) |
  |                    |  listing JSON (title, description, 5 bullets, tags) |
  |                    v                                                     |
  |                 pricing rule (undercut median, keep 25% margin) -------> suggested price
  |                    :                                                     |
  +--------------------:-----------------------------------------------------+
                       :  search text only, e.g. "maroon cotton kurti"
                       :  (skipped in Private mode)
                       v
                 SerpApi Google Shopping (India) -> competitor prices
```

1. **Read the photos.** Open-weight **Gemma 3 (4B)** runs locally in Ollama and returns
   product attributes as JSON.
2. **Write the listing.** Our **fine-tuned Qwen3-8B** turns attributes + platform + seller notes into
   a listing that follows the platform rules. We trained a LoRA on Tinker on {{n_train}} real, well-rated
   product listings, merged it into the open base model, converted it to a 4-bit GGUF file and loaded it
   into Ollama as `listsmart-writer`. The app uses the exact same prompt format as training.
3. **Price it.** Optional competitor search via SerpApi (Google Shopping, India), outliers removed,
   suggested price = a little under the median, never below cost + 25%, with a plain-English reason.
4. **Show what happened.** After every result the app shows: *Ran locally ✓ · Data that left this device:
   none · Cost: ₹0.00* and the time per step.

## Why open matters here (concrete, not slogans)

- **Runs on a laptop.** Both models run on {{hardware}} with Ollama. No GPU server, no account.
  Median time per listing: {{speed_finetuned_median_latency_s}} s for the writer.
- **Keeps data private.** Seller photos (often taken at home, and they can show the home or family) never
  leave the device. In Private mode, *nothing* does — the app shows "Data that left this device: none".
- **Fine-tunable.** We could take an open 8B model and teach it the exact job (platform rules, JSON format,
  Indian product vocabulary) with a cheap LoRA. A closed model can only be prompted (or fine-tuned inside
  someone else's cloud, where we cannot take the weights home).
- **Costs nothing to run.** Cost per 1,000 listings: **₹{{cost_local}} locally** vs
  ₹{{cost_tinker_hosted}} on a hosted fine-tune vs ₹{{cost_closed_api}} with a closed model API
  (from `eval_summary.json`; remove the closed-API number if we did not measure it).
- **No lock-in.** The fine-tuned model is a file the seller (or we) own. If a provider changes prices,
  rate limits or retires a model, the app keeps working. It even works with Wi-Fi off.

## Where open beat closed (measured)

All numbers are from our held-out test set of **{{n_test}}** products the model never saw in training,
same prompts for both models. Source: `finetune/results/eval_summary.json`.

| Metric | Base Qwen3-8B (open, not tuned) | Our fine-tune (open) | {{closed_model_name}} (closed, optional) |
|---|---|---|---|
| Valid JSON listing | {{format_valid_base}} | {{format_valid_finetuned}} | {{format_valid_closed}} |
| Follows platform rules (title length, 5 bullets, tags) | {{constraint_compliance_base}} | {{constraint_compliance_finetuned}} | {{constraint_compliance_closed}} |
| Mentions the product's attributes | {{attribute_coverage_base}} | {{attribute_coverage_finetuned}} | {{attribute_coverage_closed}} |
| ROUGE-L vs real listing | {{rouge_l_base}} | {{rouge_l_finetuned}} | {{rouge_l_closed}} |
| Median seconds per listing | {{speed_base_median_latency_s}} | {{speed_finetuned_median_latency_s}} | {{speed_closed}} |
| Cost per 1,000 listings | ₹{{cost_local}} | ₹{{cost_local}} | ₹{{cost_closed_api}} |

Blind LLM judge (Gemma, A/B order randomised): fine-tuned wins **{{judge_finetuned_win}}**, base wins
{{judge_base_win}}, tie {{judge_tie}}.

What this means in plain words (rewrite after you see the numbers — keep it honest):
- {{one_sentence_where_finetune_helped_most}}
- {{one_sentence_where_closed_was_still_better_or_equal}} ← keep this line even if closed wins somewhere.
  If we did not run a closed baseline, say so and delete that column.

## First real users

We asked {{n_sellers_asked}} sellers; {{n_sellers_tried}} tried it on a real product.
(Playbook and raw notes: [FIRST_USER.md](FIRST_USER.md).)

| Seller | Sells on | Product | Posted the listing? | Would pay? | Fair price / month | Asked for next |
|---|---|---|---|---|---|---|
| {{seller_1}} | {{platform_1}} | {{product_1}} | {{posted_1}} | {{would_pay_1}} | ₹{{price_1}} | {{next_1}} |
| {{seller_2}} | {{platform_2}} | {{product_2}} | {{posted_2}} | {{would_pay_2}} | ₹{{price_2}} | {{next_2}} |

- **Did they pay?** {{did_anyone_pay}} (write "Not yet" if nobody paid — that is fine and honest.)
- **Would they?** {{would_pay_summary}}
- **What they asked for next:** {{top_requests}}
- Quote (with permission): "{{seller_quote}}" — {{seller_quote_name_or_anonymous}}

Feedback counter from the app at the time of writing (`GET /api/feedback/summary`): {{feedback_summary_json}}

## How we built it

- **6-day hackathon, {{team_size}}, plus a multi-agent Claude Code session.** One agent per area worked in
  parallel against a shared contract file (`SPEC.md`): backend (FastAPI, Ollama/hosted/mock backends),
  fine-tuning + evaluation (Tinker), and frontend + docs. A shared `MEMORY.md` logged every decision.
  Session: {{devrelay_embed_or_link}}
- **Data:** public Amazon product metadata (Clothing, Home, Handmade), filtered to well-rated listings and
  turned into (attributes → listing) pairs; split by product; test set fixed in git.
- **Training:** LoRA SFT on Tinker on Qwen3-8B with thinking turned off; the same prompt file is used by
  training, evaluation and the app, so they cannot drift.
- **Local serving:** LoRA merged into Qwen3-8B → GGUF Q4_K_M → `ollama create listsmart-writer`. The app
  sends the exact training chat template (raw mode), so the local model sees the same tokens it was trained on.
- **Every service has a mock path**, so the app (and a `?demo=1` mode) runs with zero keys.
- Sponsors / tools: Gemma (vision), Tinker (fine-tune), SerpApi (prices), Render (hosted demo), Ollama (local runtime).

## Limits (honest)

- Needs a laptop with ~8 GB RAM; many sellers work only on phones. First listing is slower while models load.
- 4-bit local models are slower than a cloud API: {{speed_finetuned_median_latency_s}} s per listing on {{hardware}}.
- English only for now; many sellers want Hindi/Hinglish.
- Training data is Amazon listings, not Meesho listings, so the style is a bit "Amazon".
- Gemma 3 4B can misread fabric or colour from poor photos; the seller must check the attribute chips.
- Competitor prices depend on Google Shopping results and can be noisy for niche handmade items.
- The hosted Render demo uses hosted models (Render has no Ollama), so it is **not** private; the
  privacy strip says so.
- Small user test ({{n_sellers_tried}} sellers) — not proof of demand yet.

## Next steps

1. {{next_step_from_users_1}} (from first-user requests)
2. Hindi / Hinglish descriptions (fine-tune on bilingual examples).
3. Bulk mode: a folder of photos → a CSV ready for Meesho's catalogue upload.
4. A smaller writer (e.g. 4B) for 8 GB laptops and, later, phones.
5. With permission, use sellers' edited listings (saved locally in `data/feedback.jsonl`) as new training data.

---

## How to fill the placeholders

From `finetune/results/eval_summary.json` (after running `finetune/evaluate.py`; **not** the
`eval_summary.example.json` file, which has placeholder numbers):

| Placeholder | JSON path | Format |
|---|---|---|
| `{{n_test}}` | `n_test` | integer |
| `{{format_valid_base}}` / `{{format_valid_finetuned}}` | `metrics.format_valid.base` / `.finetuned` | × 100, one decimal, add % |
| `{{constraint_compliance_*}}` | `metrics.constraint_compliance.*` | % |
| `{{attribute_coverage_*}}` | `metrics.attribute_coverage.*` | % |
| `{{rouge_l_*}}` | `metrics.rouge_l.*` | 2 decimals |
| `{{*_closed}}`, `{{closed_model_name}}` | closed-baseline entries, only if `--closed-baseline` was run | else delete the column |
| `{{judge_finetuned_win}}` / `{{judge_base_win}}` / `{{judge_tie}}` | `judge.finetuned_win` / `.base_win` / `.tie` | % |
| `{{speed_base_median_latency_s}}` / `{{speed_finetuned_median_latency_s}}` | `speed.base.median_latency_s` / `speed.finetuned.median_latency_s` | 1 decimal |
| `{{hardware}}` | `speed.hardware` | text |
| `{{cost_local}}` / `{{cost_tinker_hosted}}` / `{{cost_closed_api}}` | `cost_per_1000_listings_inr.local` / `.tinker_hosted` / `.closed_api` | ₹, `null` → "not measured" |
| `{{n_train}}` | line count of `finetune/data/train.jsonl` | integer |

A quick way to print them:

```bash
python3 -c "
import json; s = json.load(open('finetune/results/eval_summary.json'))
print('n_test', s.get('n_test'))
for k, v in s.get('metrics', {}).items(): print(k, v)
print('judge', s.get('judge')); print('speed', s.get('speed')); print('cost', s.get('cost_per_1000_listings_inr'))
"
```

First-user placeholders come from your table in [FIRST_USER.md](FIRST_USER.md) and from
`curl -s http://127.0.0.1:8000/api/feedback/summary`. Only quote sellers who said yes.
