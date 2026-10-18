# First real users: a one-week playbook

Goal: get **1–3 real sellers** to try ListSmart on a real product this week, and write down
honestly what happened: did they use the listing, would they pay, how much, and what they asked for next.

Keep it small and real. One seller who actually posts a listing is worth more than ten "looks nice!" replies.

---

## 1. Who to ask (in this order)

| Who | Where to find them | Why they are a good fit |
|---|---|---|
| Family / friends who sell | WhatsApp family groups, cousins, college friends | Easiest yes, honest feedback, can sit with you |
| Meesho resellers | Meesho seller WhatsApp/Telegram groups, local reseller aunties, people sharing Meesho catalogues | Exactly our user: many products, writes listings on a phone/laptop |
| Instagram sellers | Small handmade / kurti / jewellery pages (search hashtags like `#handmadeindia`, `#kurtiseller`, `#meeshoseller`) | Post many products, care about good captions and tags |
| Local shops | Boutique, saree shop, handicraft stall near you that wants to sell online | Has stock but no time to write listings |
| Etsy sellers in India | r/etsysellers, Etsy India seller groups | Need English listings with exact tag rules |

### Message you can send (edit it in your own words)

> Hi {{name}}! I built a small free tool for sellers: you add 2–3 photos of a product and it writes the
> Meesho/Etsy title, description, bullet points and tags, and suggests a price. It runs on your own laptop,
> so your photos are not uploaded anywhere. Can I show you for 10 minutes this week with one of your
> real products? I just want honest feedback.

Hindi version:

> Namaste {{name}}! Maine sellers ke liye ek chhota free tool banaya hai: product ki 2–3 photos daalo, aur
> yeh Meesho/Etsy ka title, description, points aur tags likh deta hai, aur price bhi suggest karta hai.
> Yeh aapke laptop par hi chalta hai, photos kahin upload nahi hoti. Kya is hafte 10 minute de sakte ho?
> Aapke ek asli product par try karenge. Bas sachcha feedback chahiye.

---

## 2. Setup options

### Option A — on their laptop (best: real privacy story, real speed)

Needs a Mac/Windows/Linux laptop with about 8 GB RAM and ~10 GB free disk.

1. Install Ollama from https://ollama.com and open it once.
2. In a terminal:
   ```bash
   ollama pull gemma3:4b
   ollama pull qwen3:8b      # or build listsmart-writer, see finetune/README.md
   ```
3. Get the code (zip or `git clone`), then follow "Run locally" in the main [README](../README.md).
   If `listsmart-writer` is not built yet, start with `OLLAMA_WRITER_MODEL=qwen3:8b`.
4. Open http://127.0.0.1:8000 and check the top-right badge says "Ollama on".

Do the model downloads **before** the meeting (they are a few GB).

### Option B — on your laptop, they bring the product (fastest)

They sit with you (or send photos on WhatsApp). You run it locally, they watch and edit.
Tell them clearly the photos are on your laptop only and you will delete them after.

### Option C — screen-share (Google Meet / WhatsApp video)

Share your screen with the app running on your laptop. Ask them to send 2–3 photos of a real product
first. Let them **tell you** what to type and edit, so you see what they care about.

---

## 3. The 10-minute test script

| Minute | You do / say | They do |
|---|---|---|
| 0–1 | "This writes a listing from photos. Please think out loud, there are no wrong answers." | – |
| 1–2 | Ask: "How do you write listings today? How long does one take?" Write the answer down. | Answer |
| 2–4 | Let them pick 2–3 photos of a **real product they sell**. Choose Meesho or Etsy. Add a note and cost price if they want. | Upload, fill form |
| 4–5 | Click **Generate**. Stay quiet while it runs. | Watch |
| 5–8 | "Would you post this? What would you change?" Let them edit the title, bullets and tags. | Edit, react |
| 8–9 | Show the price suggestion and competitor list. "Is this price right for you?" | React |
| 9–10 | Ask them to use the 👍/👎 and the "Would you pay for this?" form **themselves**. | Fill feedback |

If they say they will post it: ask them to click **Copy all** (this records "I used this listing") and,
if they agree, send you the link to the live listing later.

---

## 4. What to observe (write it down, do not explain or defend)

- Time from photos to a listing they are happy with (use the "Time" in the privacy strip + their editing time).
- What they **edited** first (title? tags? description language?).
- What they **ignored** (they never looked at the price? never scrolled to tags?).
- Where they got **stuck or confused** (buttons, words, waiting time).
- Their face when the result appeared: surprised, unsure, bored?
- Did they ask "Is my photo uploaded somewhere?" — note what reassured them (Private mode, the strip).
- Did they ask to try a **second product** without you suggesting it? (Strong signal.)

---

## 5. Questions to ask at the end

1. "Would you use this for your next 10 products?" Why / why not?
2. "Would you pay for this?" (yes / maybe / no) — then: **"How much per month would feel fair?"**
   Only *after* their answer, mention our idea: free for 20 listings, then about ₹199/month.
3. "What would make it a must-have?" (e.g. Hindi, bulk upload, direct Meesho upload, phone app, better photos)
4. "What would stop you from using it?" (laptop needed? speed? trust?)
5. "Do you know one other seller who should try it?" (Ask for an intro.)
6. Did it matter to you that photos stay on your laptop and it costs nothing to run? (yes / no / didn't think about it)

Tip: the strongest proof is behaviour, not words — did they post the listing, try another product,
or ask when they can get it?

---

## 6. Reading `data/feedback.jsonl`

Every feedback click in the app adds **one JSON line** to `data/feedback.jsonl` in the project folder
(this file is gitignored and stays on the laptop that ran the app). One action = one line:

- a 👍/👎 click → `"rating": "up"` or `"down"`
- "I used this listing" or **Copy all** → `"used": true`
- the "Would you pay?" form → `"would_pay"`, `"price_ok_inr"`, `"comment"`

Each line also has `request_id` (which result it was about) and `listing` (the listing as the seller edited it).
The backend may also add a timestamp.

Quick ways to read it:

```bash
# total counts (same as the footer counter in the app)
curl -s http://127.0.0.1:8000/api/feedback/summary

# every comment and price answer, one per line
python3 -c "
import json
for line in open('data/feedback.jsonl'):
    r = json.loads(line)
    if r.get('would_pay') or r.get('comment'):
        print(r.get('request_id'), r.get('would_pay'), r.get('price_ok_inr'), '-', r.get('comment'))
"

# what did the seller change? compare the final title to the one the model wrote
python3 -c "
import json
for line in open('data/feedback.jsonl'):
    r = json.loads(line)
    if r.get('used'):
        print(r['request_id'], '|', r['listing'].get('title'))
"
```

The edited listings are useful later: with the seller's permission they can become new training
examples for the next fine-tune.

---

## 7. Consent and privacy note (read this to the seller)

> "The app runs on this laptop. Your photos are not uploaded to any server. In private mode nothing
> goes online at all; otherwise only a short search text like 'maroon cotton kurti' is sent to check
> competitor prices. Your feedback and the edited listing are saved in a file on this laptop only.
> I would like to quote your feedback in my hackathon post — can I use your first name / shop name,
> or should I keep you anonymous? I will delete your photos after today."

- Ask before taking screenshots or quoting them. Write down their answer (name / anonymous / don't quote).
- Do not share their photos, prices or shop name without a clear yes.
- If you use Option B/C, delete their photos from your laptop after the session.
- If they want, delete their lines from `data/feedback.jsonl`.

---

## 8. Results table (copy into your notes and fill in)

| # | Date | Seller (or "anonymous") | Sells on | Product tried | Setup (A/B/C) | Time today per listing | Time with ListSmart | Posted it? (link) | 👍/👎 | Would pay? | Fair ₹/month | Asked for next | Quote (with consent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | | |

Honesty rule: if nobody paid, write "nobody paid yet". If someone said "maybe", write "maybe".
That is still a real and useful result for the submission ([SUBMISSION.md](SUBMISSION.md), "First user" section).
