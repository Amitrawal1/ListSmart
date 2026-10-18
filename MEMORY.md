# ListSmart — project memory

Hackathon project (6 days): Meesho/Etsy listing writer, fine-tuned.
Sponsors: Gemma (vision), Tinker (fine-tune + measured eval vs base — required), SerpApi (competitor prices), Render (deploy).

## Key decisions
- Pipeline split in 3: Gemma reads photos -> attributes JSON; writer model (fine-tuned on Tinker) turns attributes into listing JSON; SerpApi google_shopping gives competitor prices -> rule-based suggested price.
- Why split: the fine-tune can train on text-only (attributes -> listing) pairs made from real good listings, so no image dataset is needed, and the eval is cheap and repeatable.
- One shared prompt file `backend/app/prompts.py` used by both app and finetune, so training and serving prompts never drift.
- Every external service has a mock path so the app runs with zero keys (dev, tests, demo backup).
- FastAPI serves the static frontend -> one Render web service.
- Training/eval runs are done by the user locally (not by Claude).

## History log
- 2026-10-18: Project created. SPEC.md + prompts.py written; backend, finetune and frontend built in parallel by three agents.
- 2026-10-18: Tinker base model = Qwen/Qwen3-8B with renderer qwen3_disable_thinking (Gemma is not on Tinker; Llama-3.1-8B not on current price list). Backend reads TINKER_RENDERER so serving uses the same chat template as training. Data: Amazon-Reviews-2023 meta (Clothing, Home, Handmade), 1880/100/120 split; test.jsonl tracked in git so the eval set is fixed.
- 2026-10-18: Backend Part 2: Ollama is the default for vision (gemma3:4b, /api/chat, format json) and writer (listsmart-writer, /api/generate raw=true with chat_template.build_raw_prompt so serving matches training); fallbacks ollama -> hosted -> mock; OFFLINE_MODE/`offline` field skips SerpApi and hosted models; meta reports ran_locally, data_left_device, cost_inr (prices in app/costs.py), timings; feedback saved to data/feedback.jsonl. Render uses gemma_api + tinker (no Ollama there).
- 2026-10-18: Part 2 frontend + docs: Private mode toggle (offline=true, saved in localStorage), privacy/cost strip from meta, Ollama health hint, feedback (👍/👎, used, would-pay form -> /api/feedback, footer summary), eval speed/cost tables; README local-first quickstart; docs/FIRST_USER.md + docs/SUBMISSION.md (placeholders only, no invented results).
- 2026-10-18: Local fine-tuned writer: Ollama >=0.34.1 rejects LoRA ADAPTERs, so export_to_ollama.py bakes the Tinker LoRA into one Q4_K_M GGUF with llama.cpp (default: LoRA->GGUF + llama-export-lora into Qwen3-8B Q8_0, low RAM; --mode merge = cookbook bf16 merge); chat_template.py proven token-identical to qwen3_disable_thinking; train.py also saves save_weights_external (archive download is deprecated); evaluate.py --backend ollama adds speed/hardware/cost_per_1000_listings_inr. Laptop has ~13 GB free disk: export needs ~27 GB, so free space or export on Colab.
