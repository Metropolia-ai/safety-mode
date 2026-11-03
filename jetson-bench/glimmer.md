# Muse Glimmer 30B on a Jetson AGX Orin 64 GB as an offline rule compiler

Test on 2026-10-29. Question: can an open-weights Meta model run locally on this board and replace Muse Spark (API) for compiling Safety Mode rules into JSON per `schemas/rule.schema.json`? Answer: **yes, 10 of 10 rules pass `tests/test_compiler_live.py`**. The cost is 53 s per rule with `reasoning_effort=low` and 154 s with the default setting, and 18 GB of memory.

Paths under `/mnt/data/work/...` below are the working directory on the board's USB drive.

## 1. What the model is (facts from primary sources)
| | |
|---|---|
| Name | **Muse Glimmer 30B**, Meta Superintelligence Lab, released August 2026 (created on HF 2026-08-09, updated 2026-10-22) |
| Where | https://huggingface.co/meta-models/Muse-Glimmer-30B (BF16 weights, 55.5 GB in two safetensors); official GGUF: https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF; also `meta-models/Muse-Glimmer-30B-assistant`; conversions by ggml-org, unsloth, bartowski and others |
| License | **Apache 2.0** (LICENSE file in the repo, tag `license:apache-2.0`), plus `USAGE_POLICY.md`. No MAU limits, unlike the Llama Community License. |
| Gated | **No** (`gated: false`), downloads without accepting terms. An HF token was used but was not required. |
| Architecture | dense transformer with 29.6B parameters, of which ~1.8B is a perception encoder (ViT-G/14) for images; 52 layers, hidden 6656, GQA 32/2, SwiGLU 19 968, 2048 local window + global layers, context 131 072, vocabulary 202 048. Input: text + images, output: text. Controllable reasoning (`Reasoning strength: low/medium/high/xhigh` in the system prompt, `reasoning_effort` in the API). |
| Intended use per the model card | local agent tasks: tool calls, multi-step reasoning, "OpenClaw/Hermes compatibility"; benchmarks against Gemma4-31B and Qwen3.6-27B. There is a DFlash draft model for speculative decoding (5 layers, block 16). |
| GGUF sizes | official: `KQuant-17GB-Q4_K_M` **15.6 GiB** (1.0 % degradation per the card), `KQuant-Dynamic-Q4_K_XL` 18.3 GiB (0.2 %), mmproj 1.3 GiB (images only), dflash 1.5 GiB; ggml-org: Q4_K_M 17.9, Q8_0 27.6, BF16 51.9 GiB |
| Muse Spark Open | exists as **`meta-models/Muse-Spark-Open-816B-A42B`** (MoE 816B, 42B active; NVFP4 and GGUF from unsloth/ggml-org). Does not fit this board at any quantization (even 2-bit ≈ 200+ GB), not considered. |

News context (secondary sources, 2026-08-10): Meta's first open-weight model after Llama 4, a distillation of the closed Muse Spark 1.2, positioned for a single 24 GB consumer GPU.

## 2. How it was installed
- Runtime: llama.cpp from the bench (`/mnt/data/work/bench/build/llama.cpp`, commit 0c8c59e of 2026-10-27, CUDA 12.6, sm87). The build supports the `LLM_ARCH_MUSE_GLIMMER` architecture (needs build ≥ b10353, merged 2026-08-10). Nothing was built or installed system-wide.
- Model: `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (15.6 GiB) in `/mnt/data/work/glimmer/model/`, direct curl (~16 MB/s, 16 min). mmproj and dflash were not downloaded: the USB stick ran out of space (2.6 GB free after the download).
- Server: `jetson-bench/bench/glimmer_server.sh` → `llama-server -m … -a muse-glimmer-30B -ngl 99 -c 16384 -np 1 --jinja -fa on --temp 1.0 --top-p 0.95 --top-k 64 --port 8090` (sampling from the model card; the repo's compiler sends `temperature=0` itself). Load time 43 s. Thinking arrives separately in `reasoning_content` (default reasoning-format).
- Client: venv `/mnt/data/work/venv-glimmer` (Python 3.10, `pip install -e .` of this repo: openai 3.41, jsonschema 4.26, pytest 9.2). Variables: `SM_LLM_BASE_URL=http://127.0.0.1:8090/v1 SM_LLM_API_KEY=local SM_LLM_MODEL=muse-glimmer-30B SM_LLM_PROVIDER=llama.cpp-jetson SM_LLM_TIMEOUT=900 SM_LLM_MAX_RETRIES=0`; the second run with `SM_LLM_REASONING=low`. The `safety_mode/` code was not changed.
- Run: `jetson-bench/bench/glimmer_run_live.py` uses the same `example_rules()`/`compare()` from `tests/test_compiler_live.py`, one rule at a time with a timer, peak RAM from tegrastats; results in `logs/glimmer_*.json`, compiled rules in `logs/glimmer-rules/`.
- Structured output mode: llama-server accepted `json_schema` strict right away (grammar from the schema), there was no fallback to `json_object`. Smoke test: "17 * 23" → `391`.

## 3. Table: 10 rules from rules-examples.md against rules/r01–r10.json
The criterion is the same as in `test_compiler_live.py` (SAM prompts, zones, severity, ±1 s window, top-level operator, primitives, unsupported). One call per rule, no retries. "default" = without `reasoning_effort` (the model decides how much to think), "low" = `SM_LLM_REASONING=low`.

| Rule | default | s | response tokens | low | s | response tokens |
|---|---|---|---|---|---|---|
| r01 | PASS / condition | 184 | 1543 | PASS / condition | 45 | 377 |
| r02 | PASS / condition | 146 | 1144 | PASS / condition | 52 | 420 |
| r03 | PASS / condition | 121 | 868 | PASS / condition | 45 | 375 |
| r04 | PASS / unsupported | 271 | 2274 | PASS / unsupported | 97 | 814 |
| r05 | PASS / condition | 168 | 1351 | PASS / condition | 55 | 444 |
| r06 | PASS / condition | 131 | 1031 | PASS / condition | 52 | 415 |
| r07 | PASS / condition | 103 | 724 | PASS / condition | 48 | 380 |
| r08 | PASS / condition | 110 | 854 | PASS / condition | 58 | 468 |
| r09 | PASS / unsupported | 233 | 1890 | PASS / unsupported | 42 | 354 |
| r10 | PASS / condition | 79 | 656 | PASS / condition | 34 | 280 |
| **total** | **10/10** | **154 avg, 79–271** | 1234 avg | **10/10** | **53 avg, 34–97** | 433 avg |

- r04 and r09 correctly return `unsupported` (with Muse Spark 1.3 and gpt-5-mini, r04 failed on the object set; Glimmer gave the expected `forklift, person, raised load`). Rule 11 (not part of the test) also passes: 150 s with the default setting.
- The prompt per rule is ≈ 2 120 tokens (system prompt + zones + schema as grammar). Response tokens include the reasoning; the JSON alone is ≈ 200–300 tokens.
- At `temperature=0` the response length is not deterministic: r01 with the default setting took 1 076 tokens/136 s in the first call and 1 543/184 s in the second; both PASS.

## 4. Speed and memory
| Metric | Value |
|---|---|
| Prefill | 107 tok/s on 70 tokens, **271 tok/s** on 2 115 tokens (7.8 s per rule prompt) |
| Generation | **8.4 tok/s** (118 ms/token), stable across all calls, GPU 94–99 %, VDD_GPU_SOC 38 W |
| Latency per rule | default 154 s (79–271), low **53 s** (34–97); 10 rules: 26 min / 8.8 min |
| Memory | llama-server RSS **18.1 GB** (weights 15.6 + KV for 16k context + buffers); system peak RAM 33 GB, but ~20 GB of that belongs to an unrelated process); the increase from our server ≈ 18 GB. GPU memory is shared with RAM. |
| Conditions | nvpmodel MAXN + jetson_clocks; another process (TTS) ran on the board in parallel (GPU 0 % during our measurements, but RAM in use), so these are "busy machine" figures; on a clean board they may be slightly better |
| Not tested | DFlash draft model (×3.1 on an RTX 5090 per the model card, unknown on Orin), not downloaded for lack of space; `KQuant-Dynamic` 18.3 GiB; context above 16k; images (mmproj) |

## 5. What this means for an offline deployment
- The offline rule compiler works on AGX Orin with no API and no key: 10/10 on the test set, a response in ~1 minute on `low` against ~2.5 minutes with the default. For compiling a rule "once at setup" this is acceptable. For an interactive Rule Scout (dozens of calls with tools), 8 tok/s is slow; it needs DFlash or shorter prompts.
- 18 GB of memory leaves room for SAM 3.1 (+7.5 GB) or Embedl (+2.5 GB) on the same board at the same time. On an Orin NX 16 GB this model will not fit together with the detector.
- The model is under Apache 2.0, with no MAU limit.

## 6. What did not work / limitations
- Space: after the 15.6 GiB download the USB stick had 2.6 GB free (other bench jobs took ~47 GB over the week). mmproj and DFlash were not downloaded, and speculative decoding was not tested.
- Peak memory from tegrastats is polluted by an unrelated process. The clean increase is 18 GB by llama-server RSS.
- `tests/test_compiler_live.py` was not run as pytest (it expects exactly 10 rules, while `example_rules()` now returns 11 rows with r11; this does not matter when run via `python tests/test_compiler_live.py`). The run used the same logic through `glimmer_run_live.py`.

## 7. How to reproduce
```bash
jetson-bench/bench/glimmer_server.sh start      # llama-server :8090, model from the USB stick
. /mnt/data/work/venv-glimmer/bin/activate
export SM_LLM_BASE_URL=http://127.0.0.1:8090/v1 SM_LLM_API_KEY=local SM_LLM_MODEL=muse-glimmer-30B SM_LLM_REASONING=low SM_LLM_TIMEOUT=900
cd /mnt/data/work/aivm-safety-mode && python tests/test_compiler_live.py        # or jetson-bench/bench/glimmer_run_live.py --tag X
jetson-bench/bench/glimmer_server.sh stop
```
