#!/usr/bin/env bash
# Runs the whole v2 fine-tune in order: grounded data -> train -> eval -> fill the post.
# Every step can be re-run safely: the data build resumes where it stopped.
# Cost: about $4 data + $1-2 training + under $1 eval (Tinker), judge is free (Gemma).
set -euo pipefail

cd "$(dirname "$0")/../finetune"
source .venv/bin/activate

echo "== 1/4 Build grounded data with the teacher model (asks before spending) =="
python build_grounded_data.py --teacher-model Qwen/Qwen3.6-35B-A3B

echo "== 2/4 Train v2 (2 epochs) =="
python train.py

echo "== 3/4 Evaluate base vs fine-tuned (temperature 0.7, Gemma judge) =="
# If only the judge fails (Google 500s / rate limit), re-run: python evaluate.py --reuse-outputs
python evaluate.py

echo "== 4/4 Fill the submission post with the real numbers =="
cd ..
python3 scripts/fill_submission.py
