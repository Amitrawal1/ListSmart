"""LoRA fine-tune of the ListSmart listing writer on Tinker.

Each training example is the exact chat prompt the app uses
(prompts.build_writer_messages) followed by the reference listing JSON
(prompts.listing_to_target). The loss is computed ONLY on the assistant answer.

Built on the Tinker SDK + tinker-cookbook renderers (same pattern as the cookbook's
recipes/sl_loop.py): forward_backward -> optim_step, then save_weights_for_sampler.

Usage:
  python train.py --dry-run          # offline: tokenise the data, show the loss mask, no Tinker calls
  python train.py                    # real run (needs TINKER_API_KEY)
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

from common import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    RESULTS_DIR,
    choose_renderer_name,
    load_env,
    read_jsonl,
    require_env,
    training_conversation,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--renderer", default=None, help="cookbook renderer name (default: picked from the model)")
    p.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    p.add_argument("--val-file", default=str(DATA_DIR / "val.jsonl"))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="learning rate (default: cookbook's recommended LoRA LR)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048, help="max tokens per example (longer ones are cut)")
    p.add_argument("--eval-every", type=int, default=20, help="compute val loss every N steps (0 = only at the end)")
    p.add_argument("--max-val", type=int, default=100, help="max val examples used for val loss")
    p.add_argument("--max-steps", type=int, default=None, help="stop early after N steps (for quick tests)")
    p.add_argument("--name", default="listsmart-writer", help="name for the saved sampler weights")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="build datums locally and print stats, no Tinker calls")
    return p.parse_args()


def build_datums(rows, renderer, max_length):
    """Turn dataset rows into tinker.Datum objects with loss only on the assistant answer."""
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    return [
        conversation_to_datum(
            training_conversation(row),
            renderer,
            max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for row in rows
    ]


def mean_nll(datums, loss_fn_outputs):
    """Average loss per trained token (the number we watch go down)."""
    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [out["logprobs"] for out in loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in datums]
    return compute_mean_nll(logprobs, weights)


def val_loss(training_client, val_datums, batch_size):
    """Loss on held-out rows (forward pass only, no learning)."""
    total, count = 0.0, 0
    for i in range(0, len(val_datums), batch_size):
        batch = val_datums[i : i + batch_size]
        out = training_client.forward(batch, loss_fn="cross_entropy").result()
        total += mean_nll(batch, out.loss_fn_outputs) * len(batch)
        count += len(batch)
    return total / max(count, 1)


def log(entry, log_file):
    """Print a metrics line and append it to results/train_log.jsonl."""
    print(json.dumps(entry), flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def dry_run(args, renderer, tokenizer, train_rows, val_rows):
    """Offline check: tokenise everything and show which tokens get loss."""
    datums = build_datums(train_rows[:50], renderer, args.max_length)
    lengths = [d.model_input.length for d in datums]
    trained = [sum(1 for w in d.loss_fn_inputs["weights"].data if w > 0) for d in datums]
    print(f"train rows={len(train_rows)} val rows={len(val_rows)}")
    print(f"first 50 examples: tokens avg={sum(lengths) / len(lengths):.0f} max={max(lengths)}, "
          f"trained (assistant) tokens avg={sum(trained) / len(trained):.0f}")

    # Show the text of the trained tokens for one example: it should be ONLY the listing JSON
    d = datums[0]
    targets = d.loss_fn_inputs["target_tokens"].data
    weights = d.loss_fn_inputs["weights"].data
    trained_text = tokenizer.decode([t for t, w in zip(targets, weights) if w > 0])
    prompt_text = tokenizer.decode([t for t, w in zip(targets, weights) if w == 0])
    print("\n--- prompt tokens (no loss), last 300 chars ---\n" + prompt_text[-300:])
    print("\n--- trained tokens (loss), first 400 chars ---\n" + trained_text[:400])

    steps_per_epoch = len(train_rows) // args.batch_size
    est_tokens = sum(lengths) / len(lengths) * len(train_rows) * args.epochs
    print(f"\nplan: {args.epochs} epochs x {steps_per_epoch} steps (batch {args.batch_size}) "
          f"~= {est_tokens / 1e6:.2f}M training tokens")
    print("dry run OK (no Tinker calls made)")


def main():
    args = parse_args()
    load_env()

    from tinker_cookbook import hyperparam_utils, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer_name = args.renderer or choose_renderer_name(args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    lr = args.lr or hyperparam_utils.get_lr(args.base_model)
    print(f"base model={args.base_model} renderer={renderer_name} lora_rank={args.lora_rank} lr={lr:.2e}")

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)[: args.max_val]

    if args.dry_run:
        dry_run(args, renderer, tokenizer, train_rows, val_rows)
        return

    import tinker

    require_env("TINKER_API_KEY")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / "train_log.jsonl"
    log_file.write_text("")

    val_datums = build_datums(val_rows, renderer, args.max_length)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model, rank=args.lora_rank, seed=args.seed
    )

    steps_per_epoch = len(train_rows) // args.batch_size  # drop the last partial batch
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"training for {total_steps} steps ({steps_per_epoch} per epoch)")

    # Loss before any training, so we can show it going down
    first_val = val_loss(training_client, val_datums, args.batch_size)
    log({"step": 0, "val_loss": first_val}, log_file)

    step, train_loss, last_val = 0, float("nan"), first_val
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rows = train_rows[:]
        rng.shuffle(rows)  # new order every epoch
        for b in range(steps_per_epoch):
            if step >= total_steps:
                break
            start = time.time()
            batch = build_datums(rows[b * args.batch_size : (b + 1) * args.batch_size], renderer, args.max_length)

            # Learning rate goes down linearly to 0 over the run
            current_lr = lr * max(0.0, 1.0 - step / total_steps)
            adam = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            # Send both requests before waiting, so Tinker can run them back to back
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam)
            fwd_bwd = fwd_bwd_future.result()
            optim_future.result()

            train_loss = mean_nll(batch, fwd_bwd.loss_fn_outputs)
            step += 1
            entry = {"step": step, "epoch": epoch + 1, "train_loss": round(train_loss, 4),
                     "lr": current_lr, "seconds": round(time.time() - start, 1)}
            if args.eval_every and step % args.eval_every == 0:
                last_val = val_loss(training_client, val_datums, args.batch_size)
                entry["val_loss"] = round(last_val, 4)
            log(entry, log_file)

    # Final val loss + save LoRA weights in the format the sampler (and the app) can load
    last_val = val_loss(training_client, val_datums, args.batch_size)
    log({"step": step, "final_val_loss": last_val}, log_file)
    saved = training_client.save_weights_for_sampler(name=args.name).result()
    model_path = saved.path
    print(f"\nSaved fine-tuned sampler weights: {model_path}")

    (RESULTS_DIR / "finetuned_model.txt").write_text(model_path + "\n")
    info = {
        "finetuned_model": model_path,
        "base_model": args.base_model,
        "renderer": renderer_name,
        "lora_rank": args.lora_rank,
        "learning_rate": lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps": step,
        "n_train": len(train_rows),
        "val_loss_before": first_val,
        "val_loss_after": last_val,
        "last_train_loss": train_loss,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"val loss {first_val:.3f} -> {last_val:.3f}")
    print("Next: python evaluate.py   (and set TINKER_MODEL_PATH in .env to use it in the app)")
    if math.isnan(last_val):
        print("WARNING: val loss is NaN - check the data.")


if __name__ == "__main__":
    main()
