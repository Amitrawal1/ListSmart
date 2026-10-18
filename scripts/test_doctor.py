"""Tests for the pure helper functions in scripts/doctor.py.

Run:  backend/.venv/bin/python -m pytest scripts/test_doctor.py
"""

import json
import sys
from pathlib import Path

# Make `import doctor` work no matter where pytest is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

import doctor  # noqa: E402
from doctor import FAIL, OK, WARN, Check  # noqa: E402


def test_parse_version():
    assert doctor.parse_version("ollama version is 0.34.4") == (0, 34, 4)
    assert doctor.parse_version("Warning: client version is 0.34.4") == (0, 34, 4)
    assert doctor.parse_version("no numbers here") is None
    assert doctor.parse_version(None) is None


def test_version_at_least_adapter_cutoff():
    cutoff = doctor.OLLAMA_ADAPTERS_REMOVED
    assert doctor.version_at_least((0, 34, 1), cutoff)
    assert doctor.version_at_least((0, 35, 0), cutoff)
    assert not doctor.version_at_least((0, 34, 0), cutoff)
    assert not doctor.version_at_least(None, cutoff)


def test_parse_env_file():
    text = "\n".join([
        "# a comment",
        "",
        "GOOGLE_API_KEY=abc123",
        "SERPAPI_KEY=",
        "TINKER_MODEL_PATH=           # tinker://... sampler checkpoint",
        "OLLAMA_WRITER_MODEL=listsmart-writer   # use qwen3:8b to try the base model",
        'QUOTED="has # hash"',
        "export EXPORTED=yes",
        "not a key line",
    ])
    env = doctor.parse_env_file(text)
    assert env["GOOGLE_API_KEY"] == "abc123"
    assert env["SERPAPI_KEY"] == ""
    assert env["TINKER_MODEL_PATH"] == ""
    assert env["OLLAMA_WRITER_MODEL"] == "listsmart-writer"
    assert env["QUOTED"] == "has # hash"
    assert env["EXPORTED"] == "yes"
    assert "not a key line" not in env


def test_parse_env_file_reads_the_real_example():
    example = doctor.PROJECT_ROOT / ".env.example"
    env = doctor.parse_env_file(example.read_text())
    # The example ships with every key empty and the local defaults filled in
    assert env["TINKER_API_KEY"] == ""
    assert env["WRITER_BACKEND"] == "ollama"
    assert env["OLLAMA_VISION_MODEL"] == "gemma3:4b"


def test_merged_env_real_env_wins_but_empty_does_not():
    merged = doctor.merged_env({"A": "file", "B": "file"}, {"A": "real", "B": "  "})
    assert merged == {"A": "real", "B": "file"}


def test_key_is_set_and_gemini_alias():
    assert doctor.key_is_set({"SERPAPI_KEY": "x"}, "SERPAPI_KEY")
    assert not doctor.key_is_set({"SERPAPI_KEY": "  "}, "SERPAPI_KEY")
    assert doctor.key_is_set({"GEMINI_API_KEY": "x"}, "GOOGLE_API_KEY")
    assert not doctor.key_is_set({}, "GOOGLE_API_KEY")


def test_model_present():
    names = ["gemma3:4b", "listsmart-writer:latest", "qwen2.5:3b"]
    assert doctor.model_present("gemma3:4b", names)
    assert doctor.model_present("listsmart-writer", names)
    assert doctor.model_present("listsmart-writer:latest", names)
    assert not doctor.model_present("qwen3:8b", names)
    assert not doctor.model_present("gemma3", names)  # gemma3 means gemma3:latest, not gemma3:4b


def test_models_from_manifest_dir(tmp_path):
    lib = tmp_path / "registry.ollama.ai" / "library"
    (lib / "gemma3").mkdir(parents=True)
    (lib / "gemma3" / "4b").write_text("{}")
    (lib / "listsmart-writer").mkdir()
    (lib / "listsmart-writer" / "latest").write_text("{}")
    other = tmp_path / "hf.co" / "someone" / "model"
    other.mkdir(parents=True)
    (other / "q4").write_text("{}")
    names = doctor.models_from_manifest_dir(tmp_path)
    assert names == ["gemma3:4b", "hf.co/someone/model:q4", "listsmart-writer:latest"]
    assert doctor.models_from_manifest_dir(tmp_path / "missing") == []


def test_manifest_size_gb():
    manifest = {"config": {"size": 500_000_000}, "layers": [{"size": 2_000_000_000}, {"size": 1_000_000}]}
    assert doctor.manifest_size_gb(json.dumps(manifest)) == 2.5
    assert doctor.manifest_size_gb("not json") == 0.0


def test_disk_needed_gb():
    needs = doctor.disk_needed_gb([])
    assert [gb for _, gb in needs] == [3.3, doctor.WRITER_IMPORT_GB]
    assert doctor.disk_needed_gb(["gemma3:4b", "listsmart-writer:latest"]) == []
    only_writer = doctor.disk_needed_gb(["gemma3:4b"])
    assert len(only_writer) == 1 and "listsmart-writer" in only_writer[0][0]


def test_eval_summary_problem():
    assert doctor.eval_summary_problem(json.dumps({"available": True, "metrics": {}})) is None
    placeholder = {"available": True, "note": "example placeholder, run evaluate.py for real numbers"}
    assert "placeholder" in doctor.eval_summary_problem(json.dumps(placeholder))
    assert doctor.eval_summary_problem(json.dumps({"available": False})) is not None
    assert doctor.eval_summary_problem("{broken") == "is not valid JSON"


def test_eval_summary_problem_flags_the_shipped_example():
    example = doctor.RESULTS_DIR / "eval_summary.example.json"
    assert doctor.eval_summary_problem(example.read_text()) is not None


def test_next_steps_order_and_no_repeats():
    checks = [
        Check("later", WARN, fix="step C", order=30),
        Check("fine", OK, fix="never shown", order=1),
        Check("first", FAIL, fix="step A", order=5),
        Check("dup", WARN, fix="step A", order=40),
        Check("no fix", FAIL, order=2),
        Check("middle", WARN, fix="step B", order=10),
    ]
    assert doctor.next_steps(checks) == ["step A", "step B", "step C"]


def test_format_check():
    assert doctor.format_check(Check("Python", OK, "3.12.1", fix="x")) == "  ✓ Python: 3.12.1"
    text = doctor.format_check(Check("Ollama running", FAIL, "no answer", fix="ollama serve"))
    assert text.startswith("  ✗ Ollama running: no answer")
    assert "fix: ollama serve" in text
