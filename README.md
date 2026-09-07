# Qwen3.8-Flash-Next **FP8** on 2× DGX Spark (GB10)

Serving Qwen/Qwen3.8-Flash-Next-FP8 (125B total / **6B active**, hybrid Gated DeltaNet +
Qwen Sparse Attention) TP=2 across two DGX Spark (GB10, `sm_121`) nodes with expert
parallelism, and measuring it against two models already benchmarked on the same hardware
and the same harness.

**The journey — what failed, what bricked a GPU, and the five config traps:**
[docs/JOURNEY.md](docs/JOURNEY.md)

**Companion repos (same nodes, same 40-instance SWE suite):**
- https://github.com/tsarihan/glm-5.3-flash-nvfp4-2x-dgx-spark-playbook
- https://github.com/tsarihan/deepseek-v4-flash-0731-nvfp4-2x-dgx-spark-playbook
- https://github.com/tsarihan/qwen3.8-27b-dgx-spark-gb10-playbook

© 2026 Tom Sarihan, Desnet AI LLC. Apache-2.0 (see `LICENSE`, `NOTICE`).

*Author accounts: GitHub [@tsarihan](https://github.com/tsarihan), Hugging Face [@tomsarihan](https://huggingface.co/tomsarihan). The Hugging Face account `tsarihan` is a different person and is unrelated to this work.*

---

## The short version

It works, and on this hardware it is **the best of the three by a wide margin on
everything except accuracy — where it ties.**

| | Qwen3.8-Flash-Next-FP8 | GLM-5.3-Flash NVFP4 | Qwen3.8-27B Q4_K_M |
|---|---|---|---|
| hardware | 2× DGX Spark | 2× DGX Spark | 1× RTX 5090 |
| **SWE-bench Pro (40)** | **36/40 = 90.0%** | **36/40 = 90.0%** | 30/40 = 75.0% |
| wall-clock for those 40 | **3 h 21 m** (5 workers) | ~13 h (1 worker) | ~3 h 15 m (2 workers) |
| mean agent steps | **47** | 54 | 64 |
| single-stream decode | **26.5 tok/s** | 21.5 | 131 (smaller model) |
| peak aggregate | **177.5 tok/s** @64 | 61.3 @16 | — |
| KV capacity | **595,137 tok** | 389,766 | 524,288 |
| prefix cache hit (agent load) | **91.5%** | 49.9–83.9% | — |
| NIAH 5/5 verified to | **248,418 tok** | 131K | — |

Same accuracy as GLM, **~4× faster in practice**, ~2.9× the aggregate throughput, and 1.9×
the verified context — while carrying **bf16** KV rather than fp8, because the architecture
forbids fp8 KV. A 125B model with 6B active is simply a better fit for a 121 GB unified-memory
node than GLM's denser footprint.

**The single most useful line in this repo**, if you are bringing this model up yourself:

```
--kv-cache-dtype bfloat16     # fp8 is REJECTED: "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
--tool-call-parser qwen3_xml  # NOT qwen3_coder, which is what Qwen's own repo says
--entrypoint bash             # the image ENTRYPOINT is `vllm serve`, so a bare -c becomes --compilation-config
```

---

## Reproducibility — everything is pinned

| component | pin |
|---|---|
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` — 185.6 GB, 131 shards, block-128 FP8 |
| Architecture | `Qwen4ExpForConditionalGeneration`, 48 layers, 512 experts (10 routed + 1 shared) |
| Attention | hybrid `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))` |
| Image | `vllm/vllm-openai:qwen38-flash-next` (20.6 GB) |
| vLLM | `v0.1.dev20073+g8e685d198` |
| GPU arch | `sm_121` (GB10), driver `580.173.02`, CUDA 13.0 |
| MoE backend | **DEEPGEMM Fp8** (selected automatically; *not* MARLIN — see finding 2) |
| KV cache | **bfloat16** (forced by QSA) |
| CUDA graphs | `--enforce-eager` |
| Speculative decoding | MTP, `num_speculative_tokens=3` |
| Interconnect | RoCE over the 200 GbE fabric, `NCCL_MIN/MAX_NCHANNELS=4` |
| Weights resident | 88.06 GiB per node (TP=2) |
| Harness | mini-swe-agent 2.4.6, 250-step cap |
| Evaluation | SWE-bench Pro official `swe_bench_pro_eval.py`, local Docker |

---

## Findings

### 1. fp8 KV cache is impossible — QSA mandates bf16

```
NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache
```

Qwen Sparse Attention keeps its main KV in bf16, so this model pays **2× the KV bytes per
token** that GLM's `fp8_e4m3` cache did. It still ends up with **50% more KV capacity**
(595,137 vs 389,766 tokens), because 6B active parameters leave far more of the 121 GB
unified pool free than GLM's weights do.

### 2. The MARLIN fallback did NOT happen — this was the main risk going in

vLLM [issue #43906](https://github.com/vllm-project/vllm/issues/43906) reports that on
SM_121 the MoE backend selector gates on `family(100)` (datacenter Blackwell), excluding
consumer Blackwell, so MXFP8 MoE weights get dequantized to BF16 before compute. On a 125B
MoE with 121 GB of unified memory that would be a capacity problem, not merely a slow one.

It does not affect this checkpoint. The engine selects:

```
Using DEEPGEMM Fp8 MoE backend out of potential backends:
  ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', ...]
```

The FP8 experts stay FP8. The issue is scoped to MXFP8 (OCP block-32); this is fine-grained
block-128 FP8, tracked separately as #43507, and it works.

### 3. Two official Qwen sources disagree on the tool-call parser — the recipe is right

| source | says |
|---|---|
| [Qwen repo README](https://github.com/QwenLM/Qwen3.8-Flash-Next) | `--tool-call-parser qwen3_coder` |
| [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) | `--tool-call-parser qwen3_xml` |

**`qwen3_xml` is correct for this image**, verified with a real tool-calling request
returning `finish_reason: tool_calls` and a well-formed call. Getting this wrong does not
error — the agent simply never receives a parseable action, and every instance dies with
`RepeatedFormatError` after burning its whole step budget. Verify the parser with one request
before running any benchmark.

### 4. The image ENTRYPOINT is `vllm serve`, so `-c` is stolen

```
vllm serve: error: argument --compilation-config/-cc: 1 validation error
  Invalid JSON: input_value='vllm serve /model --served-model-name ...'
```

The whole shell command was being parsed as vLLM's `--compilation-config`. Launch with
`--entrypoint bash` so `-c` reaches the shell. This is the same collision class as the
SWE-bench Pro images (`ENTRYPOINT ["/bin/bash"]`), and it presents as a nonsense error about
a flag you never passed.

### 5. `VLLM_FLASHINFER_MOE_BACKEND` does not exist in this build

NVIDIA's DGX Spark guidance recommends setting it to latency mode. This build answers:

```
WARNING [envs.py:2224] Unknown vLLM environment variable detected: VLLM_FLASHINFER_MOE_BACKEND
```

It is silently ignored. Drop it.

### 6. The vLLM recipe's memory guidance is for discrete GPUs — do not copy it to GB10

Every configuration in the official recipe is a discrete datacenter card: 4×GB300, 8×H200,
4×H100, 4×MI355X. **None is unified memory.** Two of its recommendations do not transfer:

- **`--max-num-seqs 256`** is sized for 80 GB of dedicated HBM. Measured knee here is **16**.
- **`VLLM_PLE_CPU_OFFLOAD=1`** ("for 80GB GPUs") moves the 20-million-entry n-gram embedding
  out of HBM into separate host DRAM. On GB10 there is no second pool — CPU and GPU share the
  same 121 GB — so it relocates an allocation without freeing a byte. Left at 0 here.

### 7. Throughput — the knee is at 16

| conc | TTFT p50 | per-stream tok/s | aggregate tok/s | Δ aggregate |
|---|---|---|---|---|
| 1 | 0.71 s | 26.51 | 25.61 | — |
| 2 | 1.12 | 22.31 | 42.19 | +65% |
| 4 | 1.40 | 16.43 | 60.98 | +45% |
| 8 | 1.57 | 12.99 | 95.86 | +57% |
| **16** | 2.59 | 9.77 | **140.62** | **+47%** |
| 32 | 1.82 | 9.51 | 158.02 | +12% |
| 64 | 1.63 | 8.63 | 177.46 | +12% |

16 is the last doubling that buys a large gain. Past it you pay per-stream latency for ~12%
aggregate. Note TTFT stays **under 2.6 s at every rung** — this model does not fall over under
concurrency the way a denser one does.

Useful second-order effect: setting `--max-num-seqs` to the knee (16) while running **fewer**
concurrent agents still helps, because it gives the scheduler more room to batch.

### 8. Context — 5/5 needles at 248,418 tokens

| context | TTFT | prefill tok/s | decode tok/s | needles |
|---|---|---|---|---|
| 4,316 | 2.26 s | 1,914 | 35.7 | **5/5** |
| 33,371 | 15.0 s | 2,219 | 41.3 | **5/5** |
| 132,971 | 63.3 s | 2,100 | 44.5 | **5/5** |
| 202,817 | 95.7 s | 2,120 | 40.4 | **5/5** |
| **248,418** | 119.3 s | 2,083 | 33.3 | **5/5** |

Prefill *rises* with context (1,914 → 2,219 tok/s) and decode holds ~40 tok/s well past 200K.
248,418 is effectively the full 262,144 window with room reserved for output; a prompt of
exactly 262,144 is rejected because the output tokens no longer fit.

### 9. Page cache is the wedge on GB10 — and it is not only a load-time problem

GPU and page cache draw on the **same** unified pool. An 8-worker, 185 GB checkpoint download
filled it and wedged the node: still pinging, still accepting TCP on 22, but sshd never
completing a login and the running engine unresponsive. Not an OOM kill — the kernel can evict
page cache, so it grinds instead.

`scripts/cache-warden.py` bounds it with `posix_fadvise(POSIX_FADV_DONTNEED)` on files that
have stopped growing. No root, no engine patches. With it running, three subsequent large
transfers (185 GB download, 176 GB fabric copy, 322 GB offload) held `MemAvailable` steady at
**115–116 GB** and none wedged. Run it during **any** bulk read or write, not just model loads.

---

## SWE-bench Pro — enterprise-app subset

40 instances, 10 per language, selected from the full 731 by scoring `issue_categories`
(`api_knowledge`, `authentication_authorization_knowledge`, `security_knowledge`,
`web_knowledge`, `ml_ai_knowledge`) and problem text for web/API, multi-user auth, crypto and
compliance signal, with a per-repo cap (`scripts/select_enterprise.py`). A short, targeted
benchmark for **enterprise application development**, not a general capability ranking.

| language | qwen3.8-FN | glm-5.3 | qwen3.8-27B |
|---|---|---|---|
| go | **8/10** | 7/10 | 5/10 |
| js | 10/10 | 10/10 | 8/10 |
| ts | 10/10 | 10/10 | 8/10 |
| python | 8/10 | **9/10** | 9/10 |
| **total** | **36/40 (90.0%)** | **36/40 (90.0%)** | 30/40 (75.0%) |

| repo | qwen3.8-FN | glm-5.3 | qwen3.8-27B |
|---|---|---|---|
| tutao/tutanota | 10/10 | 10/10 | 8/10 |
| gravitational/teleport | 5/5 | 5/5 | 3/5 |
| element-hq/element-web | 5/5 | 5/5 | 4/5 |
| NodeBB/NodeBB | 3/3 | 3/3 | 2/3 |
| protonmail/webclients | 2/2 | 2/2 | 2/2 |
| navidrome/navidrome | 1/1 | 1/1 | 1/1 |
| ansible/ansible | 5/6 | 5/6 | 5/6 |
| internetarchive/openlibrary | 3/4 | **4/4** | 4/4 |
| **flipt-io/flipt** | **2/4** | 1/4 | 1/4 |

**Identical totals, reached differently.** Flash-Next is better on Go — the language both
others struggled with — and weaker on Python. Only **4 disagreements out of 40**, splitting
2–2. And **flipt-io/flipt**, which had beaten both previous models at 1/4 (CSRF, cookie-auth
middleware, Kubernetes auth, webhook audit sink), is the first repo where anyone improved:
**2/4**.

**Submission rate is not resolve rate.** All three models produced patches on 95–100% of
instances; the graded numbers are 90 / 90 / 75. Roughly one patch in five *looks* plausible
and fails the tests. Only `swe_bench_pro_eval.py` against `fail_to_pass` / `pass_to_pass`
produces a score.

---

## Setup

```bash
# both nodes, after every reboot -- vm.swappiness=0 does not survive one
./scripts/prelaunch-gb10.sh

# bound page cache during any bulk transfer or load
python3 scripts/cache-warden.py /data/models/qwen3.8-flash-next-fp8 20 &

# rank 0 on node 0, rank 1 on node 1
NODE_RANK=0 ./scripts/serve-qwen38fn-fp8-vllm.sh
NODE_RANK=1 ./scripts/serve-qwen38fn-fp8-vllm.sh

# poll /health, never /v1/models
curl http://${NODE0_MGMT}:8891/health
```

Tunables: `MAX_NUM_SEQS` (knee = 16), `MAX_MODEL_LEN`, `GPU_UTIL`, `KV_DTYPE` (must stay
bfloat16), `SPEC`/`MTP_K`, `EP`, `PLE_OFFLOAD`, `TOOL_PARSER`.

## Reproducing the measurements

```bash
# concurrency ladder / knee
python3 sweep.py --base-url http://${NODE0_MGMT}:8891/v1 \
  --model qwen3.8-flash-next-fp8 --concurrency 1,2,4,8,16,32,64 --max-tokens 512

# TTFT curve and NIAH
CONTEXTS=4096,32768,131072 ./ttft-curve.sh <tag> http://${NODE0_MGMT}:8891/v1 qwen3.8-flash-next-fp8
python3 needles.py --base-url http://${NODE0_MGMT}:8891/v1 \
  --model qwen3.8-flash-next-fp8 --contexts 4096,32768,131072,245000 --needles 5

# SWE-bench Pro (5 workers against a 16-stream server)
OVERRIDE=swe-env-override.yaml ./scripts/run-suite.sh <tag> openai/<model> <base_url> 5 10
```

Grade with the official harness — a completed run is not a score:

```bash
python3 swe_bench_pro_eval.py --raw_sample_path dataset/test.jsonl \
  --patch_path patches.json --output_dir eval-out \
  --dockerhub_username jefzda --scripts_dir run_scripts \
  --use_local_docker --docker_platform linux/amd64 --num_workers 3
```

## Credits

Qwen3.8-Flash-Next by [Qwen](https://qwen.ai) / Alibaba. SWE-bench Pro by ScaleAI.
Harness: mini-swe-agent. Serving: vLLM. GB10 field observations on page cache and NCCL
channel limits informed the memory handling here.
