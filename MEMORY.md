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
