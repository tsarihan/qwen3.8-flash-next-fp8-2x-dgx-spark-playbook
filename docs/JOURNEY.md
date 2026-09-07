# The journey: 931 GB of shuffling, one bricked GPU, and five config traps

Bringing Qwen3.8-Flash-Next-FP8 up on 2× DGX Spark. Unlike the GLM bring-up on the same
nodes — where two of three serving lanes were impossible and four benchmark runs silently
returned zero — this model came up on the **first architecture that was tried**. Everything
that went wrong was either disk, memory pressure, or a config disagreement between two
official sources.

---

## Act I — 185 GB does not fit, twice over

TP=2 needs the **full checkpoint on each node**, and the checkpoint is 185.6 GB. spark-1 had
202 GB free, spark-2 had 181 GB. So one node just fit and the other was 5 GB short.

Freeing it meant moving, not deleting. Two DeepSeek NVFP4 checkpoints (165 GB + 157 GB) went
to a 3 TB drive over the LAN at ~106 MB/s, and the resulting verification produced the first
real lesson — **about the verifier, not the data**.

The check compared two `sha256sum` lists with `diff`. It reported `VERIFY_FAIL` on both
transfers. The copies were perfect: 76 files each side, and the complete *set* of content
hashes was byte-identical. The lists merely sorted dotfiles differently on the two hosts, so
an order-sensitive `diff` called correct data corrupt.

The same script also produced the opposite error. A queued `spark1-backup` move fired while
spark-1 was mid-reboot, rsync died with `connection unexpectedly closed (0 bytes received)`,
and the script then hashed an unreachable source against an empty destination, got two empty
lists, and reported:

```
VERIFY_OK spark1-backup  files=0  all sha256 match
```

**A check that passes when it has nothing to compare.** Had deletion been wired to that
signal, 609 GB would have been removed against a successful-looking verification of nothing.
Both flaws were in the verification, not the copying; the fix is to compare hash *sets*
order-independently and treat a zero or mismatched count as fatal.

---

## Act II — the download wedges the node

With space arranged, the checkpoint downloaded at 8 parallel workers. Partway through,
spark-1 stopped answering: it still replied to ping, still accepted TCP on port 22, but sshd
never completed a banner exchange and the vLLM instance already running on it went silent.

No GPU job was involved. The only load was writing 185 GB to disk.

This is the failure mode a third-party GB10 field report describes as *"page cache is half
your memory budget — and it is the wedge"*: on unified memory the page cache and the GPU draw
on the same pool, so a large sequential write starves the engine. Their observation was of
*loading* 41.6 GB shards; this was *downloading* 185 GB. Same pool, same result.

Their mitigation — `posix_fadvise(POSIX_FADV_DONTNEED)` on files that have stopped growing —
was in notes already read, and applied only to model loading. It should have been applied to
the download.

With `cache-warden.py` running, the resumed download held `MemAvailable` at **116.5 GB** for
its entire duration while dropping 188 GB of page cache cumulatively. The two subsequent large
transfers (176 GB fabric copy, 322 GB offload) behaved the same. Three transfers, no wedges,
against two wedges before.

One self-inflicted detail worth recording: while a node was wedged, polling it with background
ssh loops piled up hundreds of half-open sessions and made recovery *slower*. It looked causal
for a while. It was not — the other node wedged identically with a clean session table.

---

## Act III — the GPU bricks

Two power cycles later the node booted clean, but every container reported
`torch.cuda.is_available() == False` while its sibling was fine. `nvidia-smi` explained it:

```
|   0  NVIDIA GB10   On | 0000000F:01:00.0 N/A |    N/A |
|ERR!  35C  P8   N/A / N/A | Not Supported | N/A  Default |
|                                          |       ERR! |

Pending : GPU requires reset
```

`ERR!` in the fan/perf and compute-mode fields, and the driver asking for a reset. Almost
certainly a power cycle landing while the GPU held a UVM allocation. GB10 is integrated, so
`nvidia-smi -r` does not clear it — a clean boot does. One more cycle and the GPU came back
at `4W, 0%` with no reset pending.

**The lesson is ordering.** The wedge had been caused by page cache, and the GPU fault was
caused by power-cycling *through* an outstanding allocation. Releasing the memory holder first
(`docker rm -f` on the serving container) is cheaper than a reset, and avoids the fault.

---

## Act IV — five config traps, all recoverable in minutes

None of these produced a wrong *number*. Each stopped the engine outright, which is the good
kind of failure.

**1. The image ENTRYPOINT is `vllm serve`.**

```
vllm serve: error: argument --compilation-config/-cc: 1 validation error
  Invalid JSON: input_value='vllm serve /model --served-model-name ...'
```

The entire shell command was being consumed as vLLM's `--compilation-config`, because `-c`
is its short form. Fixed with `--entrypoint bash`. This is the *identical* collision that
wasted hours on the SWE-bench Pro images, and it presents as a nonsense error about a flag
that was never passed.

**2. fp8 KV cache is rejected.**

```
NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache
```

Qwen Sparse Attention keeps its main KV in bf16. This costs 2× the KV bytes GLM paid — and
the model *still* ends up with 50% more KV capacity, because 6B active parameters leave far
more of the pool free.

**3. `VLLM_FLASHINFER_MOE_BACKEND` does not exist** in this build. NVIDIA's DGX Spark guidance
recommends setting it to latency mode; the engine answers `Unknown vLLM environment variable
detected` and ignores it.

**4. The two official Qwen sources disagree on the tool-call parser.** The repo README says
`qwen3_coder`; the vLLM recipe says `qwen3_xml`. **`qwen3_xml` is correct for this image.**
This one would not have stopped the engine — it would have produced a full, clean-looking
benchmark run of zeros, exactly as a parser mismatch did on GLM. It was settled with one
tool-calling request before any benchmark ran.

**5. The recipe's memory guidance is for discrete GPUs.** Every config in it is 4×GB300,
8×H200, 4×H100 or 4×MI355X — all dedicated HBM. Its `--max-num-seqs 256` and
`VLLM_PLE_CPU_OFFLOAD=1` ("for 80GB GPUs") assume a second memory pool that GB10 does not
have. Measured knee here is **16**, and PLE offload cannot free capacity when CPU and GPU
share the same 121 GB.

**And the risk that did not materialise.** vLLM issue #43906 reports the MoE backend selector
gating on `family(100)` and excluding SM_121, dequantizing FP8 experts to BF16 — which on a
125B MoE would be a capacity problem, not just a slow one. The engine selected `DEEPGEMM Fp8`
instead. The issue is scoped to MXFP8; this block-128 FP8 checkpoint is unaffected.

---

## Act V — the result

Same 40 enterprise-themed instances, same harness, same litellm front door, graded with
SWE-bench Pro's official evaluator.

| model | hardware | resolved |
|---|---|---|
| **Qwen3.8-Flash-Next-FP8** | 2× DGX Spark | **36/40 = 90.0%** |
| GLM-5.3-Flash NVFP4 | 2× DGX Spark | 36/40 = 90.0% |
| Qwen3.8-27B Q4_K_M | RTX 5090 | 30/40 = 75.0% |

**A dead tie with GLM on accuracy, reached differently** — better on Go (8/10 vs 7/10, the
language both others found hardest), weaker on Python (8/10 vs 9/10), with only 4
disagreements out of 40 splitting evenly. And `flipt-io/flipt`, which had held both previous
models to 1/4, went to **2/4** — the first improvement anyone made on that repo.

The tie is on accuracy alone. Everything else is decisive: **3 h 21 m against ~13 h** for the
same 40 instances, **47 mean agent steps against 54**, 2.9× the aggregate throughput, 1.5× the
KV capacity, 91.5% prefix-cache hit rate against 49.9–83.9%, and NIAH verified to 248,418
tokens against 131K.

A 125B model with 6B active is a better fit for a 121 GB unified-memory node than a denser
model of similar file size. That is the finding.

### Still to come

The NVFP4 variant of the same model (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, 135 GB — 50 GB
smaller) for a like-for-like FP8-vs-NVFP4 comparison across TPS, TTFT, NIAH, the concurrency
knee and SWE-bench Pro, in the same shape as the DeepSeek MXFP4-vs-NVFP4 study. Results will
be published here.
