"""Turn our Tinker LoRA fine-tune into a LOCAL Ollama model called `listsmart-writer`.

Why this is needed: Tinker trains a LoRA adapter for Qwen/Qwen3-8B. To run it on the seller's
laptop we need ONE quantised GGUF file (Q4_K_M, ~5 GB) that already has the LoRA baked in.
Ollama 0.34.1+ no longer accepts LoRA adapters (`ADAPTER` in a Modelfile fails with
"LoRA adapters are no longer supported"), so the adapter must be merged before `ollama create`.

Two ways to bake it in (--mode):

  adapter (default, low RAM)                     merge (exact bf16 merge, big disk)
  --------------------------                     ----------------------------------
  download   Tinker LoRA (~0.2 GB)               download   Tinker LoRA
  peft       rename keys to PEFT format          merge_hf   cookbook build_hf_model: LoRA + bf16
             (no base weights needed)                       Qwen3-8B (downloads ~16 GB)
  lora_gguf  llama.cpp convert_lora_to_gguf.py   hf_gguf    llama.cpp convert_hf_to_gguf.py (bf16)
  base_gguf  Qwen/Qwen3-8B-GGUF Q8_0 (~8.7 GB)    quantize   llama-quantize -> Q4_K_M
  merge_gguf llama-export-lora -> f16 GGUF
  quantize   llama-quantize -> Q4_K_M
  then both: modelfile (export/Modelfile) and create (`ollama create listsmart-writer`)

Every step checks whether its output already exists and skips it, so you can stop and re-run.
`setup` (first) clones llama.cpp. Its convert scripts run with THIS venv's python (they load
their own gguf-py from the checkout; tested with torch/transformers from requirements.txt).

Usage:
  python export_to_ollama.py --dry-run              # show the plan, commands, disk/RAM needs; changes nothing
  python export_to_ollama.py                        # run every step (adapter mode)
  python export_to_ollama.py --mode merge           # exact merge path (needs ~55 GB free disk)
  python export_to_ollama.py --steps modelfile,create   # only some steps
  python export_to_ollama.py --base-gguf ollama     # reuse the qwen3:8b you already pulled (saves 8.7 GB, see README)
  python export_to_ollama.py --ollama-adapter       # OLD Ollama (<= 0.34.0) only: keep the LoRA separate
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import FINETUNE_DIR, RESULTS_DIR, chat_template, load_env

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
# Ollama release that dropped LoRA adapter support (found in ollama/ollama server/create.go)
OLLAMA_ADAPTERS_REMOVED = (0, 34, 1)

STEPS_BY_MODE = {
    "adapter": ["setup", "download", "peft", "lora_gguf", "base_gguf", "merge_gguf", "quantize", "modelfile", "create"],
    "legacy_adapter": ["setup", "download", "peft", "lora_gguf", "modelfile", "create"],
    "merge": ["setup", "download", "merge_hf", "hf_gguf", "quantize", "modelfile", "create"],
}

# Rough disk space (GB) each step ADDS, for an 8B model. Used for the free-space check.
DISK_GB = {
    "setup": 0.3,  # llama.cpp shallow clone (+ a small build if you have no llama.cpp binaries)
    "download": 0.5,
    "peft": 0.5,
    "lora_gguf": 0.5,
    "base_gguf": 8.7,  # Qwen3-8B-Q8_0.gguf
    "merge_gguf": 16.4,  # f16 GGUF (llama-export-lora always writes f16)
    "merge_hf": 33.0,  # 16.4 GB base download (HF cache) + 16.4 GB merged bf16 safetensors
    "hf_gguf": 16.4,  # bf16 GGUF
    "quantize": 5.0,  # Q4_K_M GGUF
    "modelfile": 0.0,
    "create": 5.0,  # Ollama copies the GGUF into ~/.ollama/models/blobs
}

RAM_NOTE = {
    "peft": "<1 GB (only the LoRA matrices)",
    "lora_gguf": "<2 GB",
    "merge_gguf": "~2-3 GB (one tensor at a time; quantised base tensors are dequantised to f32 one by one)",
    "merge_hf": "~5-6 GB with the cookbook's shard-by-shard merge (a plain transformers+peft merge needs ~18 GB)",
    "hf_gguf": "~2-4 GB (lazy conversion)",
    "quantize": "~2-3 GB (memory-mapped)",
    "create": "low (file copy)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["adapter", "merge"], default="adapter")
    p.add_argument("--ollama-adapter", action="store_true",
                   help="OLD Ollama (<= 0.34.0) only: Modelfile = FROM qwen3:8b + ADAPTER lora.gguf (no merge)")
    p.add_argument("--tinker-path", default=None,
                   help="tinker:// checkpoint (default: results/external_weights.txt, else results/finetuned_model.txt)")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B", help="HF base model the LoRA was trained on")
    p.add_argument("--base-hf-dir", default=None, help="local folder with the full HF base model, if you already have it")
    p.add_argument("--base-gguf", default=None,
                   help="adapter mode: path to a Qwen3-8B GGUF to merge into, or 'ollama' to reuse the qwen3:8b blob "
                        "(default: download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf)")
    p.add_argument("--base-gguf-repo", default="Qwen/Qwen3-8B-GGUF")
    p.add_argument("--base-gguf-file", default="Qwen3-8B-Q8_0.gguf")
    p.add_argument("--ollama-base", default="qwen3:8b", help="Ollama base model (for --ollama-adapter / --base-gguf ollama)")
    p.add_argument("--quant", default="Q4_K_M", help="final quantisation type for llama-quantize")
    p.add_argument("--model-name", default=os.environ.get("OLLAMA_WRITER_MODEL", "listsmart-writer"))
    p.add_argument("--temperature", type=float, default=0.2, help="default temperature written into the Modelfile")
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--export-dir", default=str(FINETUNE_DIR / "export"),
                   help="where all files go (use an external drive if your disk is small)")
    p.add_argument("--llama-cpp-dir", default=None, help="llama.cpp checkout (default: <export-dir>/llama.cpp)")
    p.add_argument("--llama-python", default=sys.executable,
                   help="python for llama.cpp convert scripts (default: this one). If it misses packages, make a venv "
                        "with llama.cpp/requirements/requirements-convert_hf_to_gguf.txt and pass its python here")
    p.add_argument("--steps", default=None, help="comma list of steps to run (default: all for the mode)")
    p.add_argument("--skip", default="", help="comma list of steps to skip")
    p.add_argument("--force", action="store_true", help="redo steps even if their output exists")
    p.add_argument("--cleanup", action="store_true",
                   help="delete big intermediate files as soon as the next step has used them")
    p.add_argument("--ignore-disk-check", action="store_true", help="run even if free disk looks too small")
    p.add_argument("--dry-run", action="store_true", help="print the plan and commands only; no downloads, no writes")
    return p.parse_args(argv)


# ---------------------------------------------------------------- small pure helpers (unit tested)
def plan_steps(mode, ollama_adapter, steps=None, skip=""):
    """Ordered list of steps to run for this mode, after --steps / --skip filters."""
    key = "legacy_adapter" if (mode == "adapter" and ollama_adapter) else mode
    all_steps = STEPS_BY_MODE[key]
    chosen = [s.strip() for s in steps.split(",") if s.strip()] if steps else list(all_steps)
    unknown = [s for s in chosen if s not in all_steps]
    if unknown:
        raise ValueError(f"steps {unknown} are not part of mode '{key}': {all_steps}")
    skipped = {s.strip() for s in skip.split(",") if s.strip()}
    return [s for s in all_steps if s in chosen and s not in skipped]


def export_paths(export_dir, quant="Q4_K_M", base_gguf_file="Qwen3-8B-Q8_0.gguf"):
    """Every file/folder the pipeline reads or writes, in one place."""
    d = Path(export_dir)
    return {
        "dir": d,
        "checkpoint": d / "tinker_checkpoint",
        "skeleton": d / "base_skeleton",
        "peft": d / "peft_adapter",
        "lora_gguf": d / "listsmart-lora-f16.gguf",
        "base_gguf": d / "base" / base_gguf_file,
        "merged_f16": d / "listsmart-writer-f16.gguf",
        "merged_hf": d / "merged",
        "merged_bf16": d / "listsmart-writer-bf16.gguf",
        "final": d / f"listsmart-writer-{quant}.gguf",
        "modelfile": d / "Modelfile",
    }


def resolve_tinker_path(explicit, results_dir):
    """Which checkpoint to download. Prefer the new 'external weights' save, else the sampler weights."""
    if explicit:
        return explicit
    for name in ("external_weights.txt", "finetuned_model.txt"):
        f = Path(results_dir) / name
        if f.exists() and f.read_text().strip():
            return f.read_text().strip()
    return None


def find_checkpoint_root(folder):
    """Find what was downloaded: ('lora', dir) for a LoRA adapter, ('hf_model', dir) for a full model, or (None, None)."""
    folder = Path(folder)
    if not folder.is_dir():
        return None, None
    for cfg in sorted(folder.rglob("adapter_config.json")):
        if (cfg.parent / "adapter_model.safetensors").exists():
            return "lora", cfg.parent
    for cfg in sorted(folder.rglob("config.json")):
        if list(cfg.parent.glob("*.safetensors")):
            return "hf_model", cfg.parent
    return None, None


def safe_relative_path(name):
    """Reject absolute paths and '..' in file names that come from a download listing."""
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe file name in checkpoint listing: {name!r}")
    return rel


def parse_ollama_version(text):
    """'ollama version is 0.34.4' -> (0, 34, 4); None if no version found."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def parse_modelfile_from(modelfile_text):
    """The blob path in the FROM line of `ollama show --modelfile` output."""
    for line in modelfile_text.splitlines():
        if line.startswith("FROM "):
            return line[5:].strip()
    return None


# Qwen3 chat format with thinking switched off (same as qwen3_disable_thinking), for `ollama run`.
# The app and evaluate.py send raw prompts (raw=true), which skip this template completely.
MODELFILE_TEMPLATE = '''{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{- end }}<|im_start|>assistant
<think>

</think>

'''


def render_modelfile(from_ref, adapter=None, stops=None, temperature=0.2, num_ctx=4096):
    """Text of the Ollama Modelfile. Paths are relative to the Modelfile's folder."""
    lines = ["# ListSmart listing writer: Qwen3-8B + our Tinker LoRA (made by export_to_ollama.py)", f"FROM {from_ref}"]
    if adapter:
        lines.append(f"ADAPTER {adapter}")
    lines.append(f'TEMPLATE """{MODELFILE_TEMPLATE}"""')
    for stop in stops if stops is not None else chat_template.STOP_STRINGS:
        lines.append(f'PARAMETER stop "{stop}"')
    lines.append(f"PARAMETER temperature {temperature}")
    lines.append(f"PARAMETER num_ctx {num_ctx}")
    return "\n".join(lines) + "\n"


# With --cleanup, these files are deleted after the step that uses them: step -> GB freed
FREED_BY_CLEANUP = {"merge_gguf": DISK_GB["base_gguf"], "quantize": 16.4, "hf_gguf": 16.4}


def peak_disk_gb(steps, cleanup=False, overrides=None):
    """Most extra disk in use at any moment while running these steps (GB)."""
    sizes = dict(DISK_GB, **(overrides or {}))
    used = peak = 0.0
    for step in steps:
        used += sizes.get(step, 0)
        peak = max(peak, used)
        if cleanup:
            # merge_gguf frees the base GGUF, but only one we downloaded (size 0 when reusing one)
            freed = sizes["base_gguf"] if step == "merge_gguf" else FREED_BY_CLEANUP.get(step, 0)
            used -= min(freed, used)
    return round(peak, 1)


# ---------------------------------------------------------------- the pipeline
class Exporter:
    def __init__(self, args):
        self.args = args
        self.paths = export_paths(args.export_dir, args.quant, args.base_gguf_file)
        self.llama_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else self.paths["dir"] / "llama.cpp"
        self.dry = args.dry_run
        self.tinker_path = resolve_tinker_path(args.tinker_path, RESULTS_DIR)

    # ---- helpers
    def say(self, msg):
        print(msg, flush=True)

    def run(self, cmd, cwd=None):
        """Run a command (or just print it in dry-run mode)."""
        printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        self.say(f"  $ {printable}")
        if not self.dry:
            subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

    def llama_python(self):
        return self.args.llama_python

    def tool(self, name):
        """A llama.cpp binary: from PATH (e.g. `brew install llama.cpp`) or our own build."""
        found = shutil.which(name)
        if found:
            return found
        built = self.llama_dir / "build" / "bin" / name
        if built.exists() or self.dry:
            return str(built)
        sys.exit(f"ERROR: {name} not found. Build it (needs `brew install cmake`):\n"
                 f"  cmake -S {self.llama_dir} -B {self.llama_dir}/build && "
                 f"cmake --build {self.llama_dir}/build --config Release -j --target llama-quantize llama-export-lora")

    def checkpoint(self):
        """(kind, folder) of the downloaded checkpoint."""
        return find_checkpoint_root(self.paths["checkpoint"])

    def remove(self, path):
        """Delete a big intermediate (only with --cleanup)."""
        path = Path(path)
        if not self.args.cleanup or self.dry or not path.exists():
            return
        self.say(f"  cleanup: deleting {path}")
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    # ---- step: setup
    def setup_done(self):
        need_bins = self.need_binaries()
        return ((self.llama_dir / "convert_lora_to_gguf.py").exists()
                and all(shutil.which(b) or (self.llama_dir / "build" / "bin" / b).exists() for b in need_bins))

    def need_binaries(self):
        steps = set(self.steps)
        return [b for b, used_by in (("llama-export-lora", {"merge_gguf"}), ("llama-quantize", {"quantize"}))
                if steps & used_by]

    def step_setup(self):
        if not (self.llama_dir / "convert_lora_to_gguf.py").exists():
            self.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, self.llama_dir])
        missing = [b for b in self.need_binaries()
                   if not shutil.which(b) and not (self.llama_dir / "build" / "bin" / b).exists()]
        if missing:
            if shutil.which("cmake"):
                self.run(["cmake", "-S", self.llama_dir, "-B", self.llama_dir / "build"])
                self.run(["cmake", "--build", self.llama_dir / "build", "--config", "Release", "-j",
                          "--target", *missing])
            else:
                msg = (f"{missing} not found and cmake is missing. Run `brew install cmake` and re-run: this step "
                       "then builds them from the llama.cpp checkout (~5 min). (`brew install llama.cpp` gives "
                       "llama-quantize, but its build may not include llama-export-lora.)")
                if self.dry:
                    self.say("  NOTE: " + msg)
                else:
                    sys.exit("ERROR: " + msg)

    # ---- step: download
    def download_done(self):
        return self.checkpoint()[0] is not None

    def step_download(self):
        if not self.tinker_path:
            msg = "no tinker:// path yet. Run train.py first or pass --tinker-path tinker://..."
            if not self.dry:
                sys.exit("ERROR: " + msg)
            self.say("  NOTE: " + msg)
            self.tinker_path = "tinker://<run-id>/sampler_weights/listsmart-writer"
        out = self.paths["checkpoint"]
        self.say(f"  tinker path: {self.tinker_path}")
        if self.dry:
            api = ("RestClient.get_external_weights_urls(path) -> one signed URL per file"
                   if "/external_weights/" in self.tinker_path else
                   "tinker_cookbook.weights.download(tinker_path=..., output_dir=...)  "
                   "[uses RestClient.get_checkpoint_archive_url_from_tinker_path -> signed .tar URL]")
            self.say(f"  would download with {api}\n  into {out}")
            return
        import tinker

        if not os.environ.get("TINKER_API_KEY"):
            sys.exit("ERROR: TINKER_API_KEY is not set (put it in .env).")
        partial = out.with_name(out.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        if "/external_weights/" in self.tinker_path:
            # New API: one signed URL per file of the external-format checkpoint
            import urllib.request

            rc = tinker.ServiceClient().create_rest_client()
            urls = rc.get_external_weights_urls(self.tinker_path).result().urls
            for name, url in urls.items():
                dest = partial / safe_relative_path(name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.say(f"  downloading {name}")
                urllib.request.urlretrieve(url, dest)
        else:
            # Documented sampler-weights download (tar archive with adapter_model.safetensors + adapter_config.json)
            from tinker_cookbook import weights

            weights.download(tinker_path=self.tinker_path, output_dir=str(partial))
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
        kind, root = self.checkpoint()
        if kind is None:
            sys.exit(f"ERROR: downloaded files in {out} are neither a LoRA adapter nor a HF model: "
                     f"{[str(p.relative_to(out)) for p in out.rglob('*')][:20]}")
        self.say(f"  got a {kind} checkpoint in {root}")

    # ---- step: peft (Tinker key names -> standard PEFT names, without downloading the 16 GB base)
    def peft_done(self):
        return (self.paths["peft"] / "adapter_model.safetensors").exists()

    def write_base_skeleton(self):
        """A tiny stand-in for the base model: real config.json + every weight NAME (1-number tensors).

        The cookbook converter only needs the base model's config and weight names to rename the
        LoRA keys, so we read them from the safetensors headers on HuggingFace (a few KB) instead of
        downloading 16 GB of weights.
        """
        import torch
        from huggingface_hub import get_safetensors_metadata, hf_hub_download
        from safetensors.torch import save_file

        skel = self.paths["skeleton"]
        if (skel / "model.safetensors").exists() and (skel / "config.json").exists():
            return skel
        skel.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(self.args.base_model, "config.json"), skel / "config.json")
        names = get_safetensors_metadata(self.args.base_model).weight_map.keys()
        save_file({name: torch.zeros(1) for name in names}, str(skel / "model.safetensors"))
        return skel

    def step_peft(self):
        kind, root = (None, None) if self.dry else self.checkpoint()
        base = self.args.base_hf_dir or str(self.paths["skeleton"])
        self.say(f"  tinker_cookbook.weights.build_lora_adapter(base_model={base!r}, "
                 f"adapter_path=<checkpoint>, output_path={str(self.paths['peft'])!r})")
        if self.dry:
            return
        if kind != "lora":
            sys.exit(f"ERROR: expected a LoRA adapter, got {kind}. A full model checkpoint needs --mode merge.")
        from tinker_cookbook.weights import build_lora_adapter

        if not self.args.base_hf_dir:
            self.write_base_skeleton()
        shutil.rmtree(self.paths["peft"], ignore_errors=True)
        build_lora_adapter(base_model=base, adapter_path=str(root), output_path=str(self.paths["peft"]))
        # Point the adapter at the real HF model id (not our skeleton folder)
        cfg_file = self.paths["peft"] / "adapter_config.json"
        cfg = json.loads(cfg_file.read_text())
        cfg["base_model_name_or_path"] = self.args.base_model
        cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
        self.say(f"  PEFT adapter: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} targets={cfg.get('target_modules')}")

    # ---- step: lora_gguf
    def lora_gguf_done(self):
        return self.paths["lora_gguf"].exists()

    def step_lora_gguf(self):
        # --base only needs config.json (no weights); the skeleton has the real one
        base_cfg_dir = self.args.base_hf_dir or self.paths["skeleton"]
        if not self.dry and not (Path(base_cfg_dir) / "config.json").exists():
            self.write_base_skeleton()
        tmp = self.paths["lora_gguf"].with_suffix(".partial.gguf")
        self.run([self.llama_python(), self.llama_dir / "convert_lora_to_gguf.py", "--outtype", "f16",
                  "--base", base_cfg_dir, "--outfile", tmp, self.paths["peft"]])
        if not self.dry:
            tmp.rename(self.paths["lora_gguf"])

    # ---- step: base_gguf
    def base_gguf_path(self):
        """Where the base GGUF to merge into lives (may be Ollama's own qwen3:8b blob)."""
        choice = self.args.base_gguf
        if choice and choice != "ollama":
            return Path(choice)
        if choice == "ollama":
            if self.dry and not shutil.which("ollama"):
                return Path(f"<blob of {self.args.ollama_base} from `ollama show --modelfile`>")
            text = subprocess.run(["ollama", "show", "--modelfile", self.args.ollama_base],
                                  capture_output=True, text=True).stdout
            blob = parse_modelfile_from(text)
            if not blob or not Path(blob).exists():
                if self.dry:
                    return Path(f"<blob of {self.args.ollama_base}: run `ollama pull {self.args.ollama_base}` first>")
                sys.exit(f"ERROR: could not find the GGUF blob of {self.args.ollama_base}. Run: ollama pull {self.args.ollama_base}")
            return Path(blob)
        return self.paths["base_gguf"]

    def base_gguf_done(self):
        return self.base_gguf_path().exists()

    def step_base_gguf(self):
        if self.args.base_gguf:
            self.say(f"  using existing base GGUF: {self.base_gguf_path()}")
            if not self.dry and not self.base_gguf_path().exists():
                sys.exit(f"ERROR: {self.base_gguf_path()} does not exist")
            return
        self.say(f"  huggingface_hub.hf_hub_download('{self.args.base_gguf_repo}', '{self.args.base_gguf_file}', "
                 f"local_dir='{self.paths['base_gguf'].parent}')   (~8.7 GB)")
        if self.dry:
            return
        from huggingface_hub import hf_hub_download

        hf_hub_download(self.args.base_gguf_repo, self.args.base_gguf_file, local_dir=str(self.paths["base_gguf"].parent))

    # ---- step: merge_gguf (bake the LoRA into the base GGUF)
    def merge_gguf_done(self):
        return self.paths["merged_f16"].exists() or self.paths["final"].exists()

    def step_merge_gguf(self):
        tmp = self.paths["merged_f16"].with_suffix(".partial.gguf")
        self.run([self.tool("llama-export-lora"), "-m", self.base_gguf_path(), "--lora", self.paths["lora_gguf"],
                  "-o", tmp])
        if not self.dry:
            tmp.rename(self.paths["merged_f16"])
            if not self.args.base_gguf:  # only delete a base file we downloaded ourselves
                self.remove(self.paths["base_gguf"])

    # ---- step: merge_hf (exact merge in bf16 with the cookbook)
    def merge_hf_done(self):
        return (self.paths["merged_hf"] / "config.json").exists() or self.paths["merged_bf16"].exists() \
            or self.paths["final"].exists()

    def step_merge_hf(self):
        base = self.args.base_hf_dir or self.args.base_model
        self.say(f"  tinker_cookbook.weights.build_hf_model(base_model={base!r}, adapter_path=<checkpoint>, "
                 f"output_path={str(self.paths['merged_hf'])!r}, dtype='bfloat16', merge_strategy='shard')")
        if self.dry:
            return
        kind, root = self.checkpoint()
        if kind == "hf_model":
            self.say(f"  checkpoint is already a full model, using it as is: {root}")
            return
        from huggingface_hub import hf_hub_download
        from tinker_cookbook.weights import build_hf_model

        shutil.rmtree(self.paths["merged_hf"], ignore_errors=True)
        build_hf_model(base_model=base, adapter_path=str(root), output_path=str(self.paths["merged_hf"]),
                       dtype="bfloat16", merge_strategy="shard")
        # convert_hf_to_gguf.py needs the tokenizer files next to the weights
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
            if not (self.paths["merged_hf"] / name).exists():
                shutil.copy(hf_hub_download(self.args.base_model, name), self.paths["merged_hf"] / name)

    # ---- step: hf_gguf
    def hf_gguf_done(self):
        return self.paths["merged_bf16"].exists() or self.paths["final"].exists()

    def step_hf_gguf(self):
        kind, root = (None, None) if self.dry else self.checkpoint()
        src = root if kind == "hf_model" else self.paths["merged_hf"]
        tmp = self.paths["merged_bf16"].with_suffix(".partial.gguf")
        self.run([self.llama_python(), self.llama_dir / "convert_hf_to_gguf.py", src, "--outtype", "bf16",
                  "--outfile", tmp])
        if not self.dry:
            tmp.rename(self.paths["merged_bf16"])
            self.remove(self.paths["merged_hf"])

    # ---- step: quantize
    def quantize_done(self):
        return self.paths["final"].exists()

    def step_quantize(self):
        src = self.paths["merged_bf16"] if self.args.mode == "merge" else self.paths["merged_f16"]
        tmp = self.paths["final"].with_suffix(".partial.gguf")
        self.run([self.tool("llama-quantize"), src, tmp, self.args.quant])
        if not self.dry:
            tmp.rename(self.paths["final"])
            self.remove(src)

    # ---- step: modelfile
    def modelfile_text(self):
        if self.args.ollama_adapter:
            return render_modelfile(self.args.ollama_base, adapter=f"./{self.paths['lora_gguf'].name}",
                                    temperature=self.args.temperature, num_ctx=self.args.num_ctx)
        return render_modelfile(f"./{self.paths['final'].name}", temperature=self.args.temperature,
                                num_ctx=self.args.num_ctx)

    def modelfile_done(self):
        mf = self.paths["modelfile"]
        return mf.exists() and mf.read_text() == self.modelfile_text()

    def step_modelfile(self):
        text = self.modelfile_text()
        self.say(f"  write {self.paths['modelfile']}:\n" + "\n".join("    | " + ln for ln in text.splitlines()))
        if not self.dry:
            self.paths["modelfile"].parent.mkdir(parents=True, exist_ok=True)
            self.paths["modelfile"].write_text(text)

    # ---- step: create
    def create_done(self):
        if self.dry or not shutil.which("ollama"):
            return False
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
        return any(line.split()[0] in (self.args.model_name, f"{self.args.model_name}:latest")
                   for line in out.splitlines()[1:] if line.strip())

    def step_create(self):
        if self.args.ollama_adapter and shutil.which("ollama"):
            out = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            version = parse_ollama_version(out.stdout + out.stderr)
            if version and version >= OLLAMA_ADAPTERS_REMOVED:
                msg = (f"Ollama {'.'.join(map(str, version))} rejects ADAPTER (removed in 0.34.1). "
                       "Use the default mode instead (bakes the LoRA into the GGUF).")
                if self.dry:
                    self.say("  WARNING: " + msg)
                else:
                    sys.exit("ERROR: " + msg)
        cmd = ["ollama", "create", self.args.model_name, "-f", self.paths["modelfile"]]
        if not shutil.which("ollama"):
            self.say("  ollama not found on PATH. Install it (https://ollama.com/download), then run:")
            self.dry = True
            self.run(cmd, cwd=self.paths["dir"])
            self.dry = self.args.dry_run
            return
        self.run(cmd, cwd=self.paths["dir"])
        if not self.dry:
            self.say(f"  done. Try: ollama run {self.args.model_name}   (Ollama keeps its own copy; you may delete "
                     f"{self.paths['final'].name} with --cleanup next time)")
            self.say(f"  Next: python evaluate.py --backend ollama --limit 20 --no-judge")

    # ---- driver
    def main(self):
        args = self.args
        self.steps = plan_steps(args.mode, args.ollama_adapter, args.steps, args.skip)
        todo = [s for s in self.steps if args.force or not getattr(self, f"{s}_done")()]
        self.say(f"mode={args.mode}{' (legacy ADAPTER)' if args.ollama_adapter else ''}  "
                 f"export dir={self.paths['dir']}  tinker path={self.tinker_path}")
        self.say(f"steps: {', '.join(s + ('' if s in todo else ' (done)') for s in self.steps)}")

        # Free-space check before anything big (reusing an existing base GGUF costs no extra disk)
        overrides = {"base_gguf": 0.0} if args.base_gguf else {}
        if args.ollama_adapter:
            overrides["create"] = 0.3  # Ollama only copies the small LoRA file
        need = peak_disk_gb(todo, args.cleanup, overrides)
        probe = self.paths["dir"] if self.paths["dir"].exists() else self.paths["dir"].parent
        free = shutil.disk_usage(probe).free / 1e9
        self.say(f"disk: ~{need:.1f} GB needed for the remaining steps, {free:.1f} GB free on {probe}")
        if need > free and not args.ignore_disk_check:
            msg = (f"not enough free disk ({free:.1f} GB < ~{need:.1f} GB). Free space, add --cleanup, use an external "
                   f"drive (--export-dir /Volumes/<drive>/listsmart-export), or run this script on Colab/Kaggle and copy "
                   f"only {self.paths['final'].name} (~5 GB) + Modelfile back (see README).")
            if args.dry_run:
                self.say("WARNING: " + msg)
            else:
                sys.exit("ERROR: " + msg + " Override with --ignore-disk-check.")

        for step in self.steps:
            if step not in todo:
                self.say(f"\n[{step}] already done, skipping")
                continue
            note = f"  (RAM {RAM_NOTE[step]})" if step in RAM_NOTE else ""
            self.say(f"\n[{step}] disk +{DISK_GB.get(step, 0)} GB{note}")
            getattr(self, f"step_{step}")()
        self.say("\nDRY RUN: nothing was downloaded or written." if args.dry_run else "\nexport finished.")


def main(argv=None):
    load_env()
    Exporter(parse_args(argv)).main()


if __name__ == "__main__":
    main()
