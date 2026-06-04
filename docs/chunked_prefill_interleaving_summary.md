# Chunked Prefill and Decode Interleaving Summary

## Goal

The goal of this feature is to improve fairness between long prompt prefill and active decode requests.

Before this scheduler change, Nano-vLLM preferred prefill whenever there was waiting prompt work. If a long prompt was chunked into many prefill steps, already-running decode requests could stall behind those prefill chunks. This is bad for latency because users with active generations may wait even though each decode step is small.

The new policy keeps the existing chunked prefill behavior, but alternates work when both queues are active:

```text
prefill chunk -> decode batch -> prefill chunk -> decode batch
```

This is controlled by `Config.prefill_decode_interleave`, which defaults to `True`.

## Code Change

The change is intentionally scheduler-only.

### Configuration

`nanovllm/config.py` adds:

```python
prefill_decode_interleave: bool = True
```

This lets users compare the old and new behavior:

```python
LLM(model_path, prefill_decode_interleave=False)
LLM(model_path, prefill_decode_interleave=True)
```

### Scheduler Refactor

`nanovllm/engine/scheduler.py` was split into three scheduling methods:

- `schedule()`: decides whether the next engine step is prefill or decode.
- `schedule_prefill()`: schedules waiting prompt tokens up to `max_num_batched_tokens`.
- `schedule_decode()`: schedules one decode token per running sequence up to `max_num_seqs`.

The scheduler now tracks:

```python
self.last_schedule_was_prefill
```

When interleaving is enabled, the scheduler chooses decode first if all of these are true:

```python
self.prefill_decode_interleave
self.last_schedule_was_prefill
self.waiting
self.running
```

In plain language: if the previous step was prefill, and there is both waiting prompt work and active decode work, run decode next.

### Attention And Model Runner

No attention kernel change was required for this first stage.

Nano-vLLM already supports partial prefill. `schedule_prefill()` can set:

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

Then `ModelRunner.prepare_prefill()` starts at `seq.num_cached_tokens` and prepares only the next prompt chunk. Later chunks can attend to cached prefix blocks through `block_tables`.

This feature does not mix prefill and decode tokens in the same model forward. Each engine step is still either:

- a prefill forward, or
- a decode forward.

That keeps the implementation small and avoids changing the attention context, which currently uses one global `is_prefill` flag.

## Experiment Plan

The experiment compares:

- baseline: `prefill_decode_interleave=False`
- interleaved scheduler: `prefill_decode_interleave=True`

The benchmark script is:

```text
experiments/benchmark_latency_metrics.py
```

The measured workload is:

- 4 initial short requests
- 2 warmup scheduler steps, so the short requests enter decode
- 1 long prompt request of about 1024 prompt tokens
- 4 late short requests
- `max_num_batched_tokens=128`, forcing the long prompt to be chunked
- `max_num_seqs=8`
- Qwen3-0.6B on a rented RTX 4090 RunPod

The script records:

- per-step trace in `steps.csv`
- per-request latency in `requests.csv`
- aggregate metrics in `summary.json`
- cross-mode comparison in `compare_summary.json`

Metrics include:

- time to first token, p50/p90/p99
- inter-token latency, p50/p90/p99
- end-to-end request latency, p50/p90/p99
- prefill tokens per second
- decode tokens per second
- total output tokens per second
- number of preemptions
- peak KV-cache block usage
- peak CUDA allocated/reserved memory

## Experiment Result

Result directory:

```text
latency_metrics_20260603_023520/
```

### Schedule Trace

With `prefill_decode_interleave=False`, the long prompt's prefill chunks run back-to-back before decode resumes:

```text
prefill, decode, prefill, prefill, prefill, prefill, ...
```

With `prefill_decode_interleave=True`, prefill and decode alternate while both queues have work:

```text
prefill, decode, prefill, decode, prefill, decode, ...
```

This confirms that the scheduler policy is working.

### Metric Summary

| Metric | Interleave False | Interleave True | Result |
|---|---:|---:|---|
| Total time | 44.95 s | 42.57 s | 5.3% faster |
| TTFT average | 11.70 s | 10.76 s | Better |
| TTFT p50 | 0.519 s | 0.494 s | Better |
| TTFT p90 | 25.67 s | 23.60 s | Better |
| TTFT p99 | 25.67 s | 23.60 s | Better |
| Inter-token latency average | 0.0645 s | 0.0622 s | Slightly better |
| Inter-token latency p50 | 0.0277 s | 0.0280 s | Similar |
| Inter-token latency p90 | 0.0280 s | 0.0301 s | Similar |
| Inter-token latency p99 | 0.376 s | 0.0615 s | Better |
| End-to-end latency average | 15.42 s | 14.34 s | Better |
| End-to-end latency p50 | 19.23 s | 18.94 s | Slightly better |
| End-to-end latency p90 | 27.95 s | 25.63 s | Better |
| End-to-end latency p99 | 27.95 s | 25.63 s | Better |
| Prefill throughput | 43.98 tok/s | 48.27 tok/s | Better |
| Decode throughput | 27.69 tok/s | 27.76 tok/s | Similar |
| Output throughput | 11.75 tok/s | 12.40 tok/s | Better |
| Preemptions | 0 | 0 | No KV pressure |
| Peak KV blocks used | 13 | 13 | Same |
| Peak CUDA allocated | 22.41 GB | 20.99 GB | Lower in this run |

### Per-Request Observations

The long request improved slightly:

| Request | False TTFT | True TTFT | False E2E | True E2E |
|---|---:|---:|---:|---:|
| `long_0` | 0.492 s | 0.439 s | 0.940 s | 0.889 s |

The late short requests also improved slightly:

| Request | False TTFT | True TTFT | False E2E | True E2E |
|---|---:|---:|---:|---:|
| `late_short_0` | 0.519 s | 0.494 s | 2.262 s | 2.249 s |
| `late_short_1` | 0.519 s | 0.494 s | 2.262 s | 2.249 s |
| `late_short_2` | 0.519 s | 0.493 s | 2.262 s | 2.249 s |
| `late_short_3` | 0.519 s | 0.493 s | 19.233 s | 18.941 s |

`late_short_3` has much higher end-to-end latency in both modes because the experiment has 9 total requests and `max_num_seqs=8`. One request waits while the active decode batch is full. This is a benchmark-shape artifact, not the core interleaving behavior.

## What We Learned

The scheduler-level implementation achieved the expected behavior. When both queues have work, it prevents a long chunked prompt from monopolizing the engine.

The measured run shows:

- lower total runtime
- lower average TTFT
- better p90/p99 TTFT
- lower average end-to-end latency
- better p90/p99 end-to-end latency
- similar decode throughput
- no increase in KV pressure

The most important qualitative result is the trace pattern. With interleaving disabled, decode waits behind consecutive long-prompt prefill chunks. With interleaving enabled, active decoders continue making progress every other step.

## Caveats

The absolute TTFT numbers are affected by cold-start overhead. The first measured prefill step takes around 23-25 seconds, so initial short requests include model warmup/compilation/loading effects in their TTFT.

The experiment also uses `max_num_seqs=8` with 9 requests. This creates unrelated queueing for one late short request, visible as high end-to-end latency for `late_short_3` in both modes.

Because `SamplingParams` currently rejects `temperature=0`, this run uses stochastic sampling with `temperature=0.6`. It is good for latency measurement, but not ideal for exact output-token correctness comparison.

## Future Plan

1. Add a separate unmeasured warmup phase before starting benchmark requests.

   This will remove model cold-start overhead from TTFT and end-to-end latency.

2. Run with `max_num_seqs >= total active requests`.

   This will avoid unrelated starvation from the decode batch limit and isolate the interleaving effect.

3. Add scheduler unit tests.

   Tests should cover single long prompt chunking, running decoders plus a long waiting prompt, multiple waiting prompts, and KV-pressure preemption.

4. Add a deterministic correctness test.

   Once greedy sampling is supported, compare generated token IDs between chunked and non-chunked prefill for the same prompts.

5. Test heavier workloads.

   Useful workloads include one long prompt plus many short prompts, many active decoders plus one long prompt, and continuous mixed traffic with short and long prompts.

6. Measure under KV pressure.

   The current run had zero preemptions. A smaller KV cache or larger workload should be used to validate preemption behavior.

7. Consider true mixed prefill/decode batching later.

   This first stage alternates separate prefill and decode forwards. A future stage could support prefill and decode rows in the same model forward, but that requires attention-layer changes and should be treated as a separate project.
