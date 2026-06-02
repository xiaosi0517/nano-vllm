# Chunked Prefill and Decode Interleaving

This note describes the first-stage scheduler change for chunked prefill fairness in Nano-vLLM. The goal is to keep long prompts from monopolizing the engine while decode requests are already running.

## Summary

Nano-vLLM already supports partial prompt prefill at the model-runner and attention-metadata level. The scheduler can schedule fewer prompt tokens than the full prompt length:

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

`ModelRunner.prepare_prefill()` then starts from `seq.num_cached_tokens`, prepares only that prompt slice, and uses the cached KV blocks as the prefix context for later chunks.

The change in this patch is scheduler-level fairness:

```text
prefill chunk -> decode batch -> prefill chunk -> decode batch
```

This is controlled by `Config.prefill_decode_interleave`, which defaults to `True`.

## Current Design

The scheduler has two queues:

- `waiting`: sequences that still need prompt prefill or were preempted.
- `running`: sequences that have completed prefill and are generating decode tokens.

The scheduling logic is split into three methods:

- `schedule()`: chooses whether the next engine step is prefill or decode.
- `schedule_prefill()`: batches prompt tokens up to `max_num_batched_tokens`.
- `schedule_decode()`: schedules one decode token per running sequence, up to `max_num_seqs`.

The scheduler remembers whether the previous step was prefill:

```python
self.last_schedule_was_prefill
```

When prefill/decode interleaving is enabled, and both queues have work, the scheduler chooses decode immediately after a prefill step:

```text
if previous step was prefill and waiting and running:
    schedule decode
else:
    try prefill first, then decode
```

This preserves the original preference for prefill when decode has no work, while preventing a long prompt from blocking already-running decoders for many consecutive steps.

## Attention Behavior

No attention-kernel change is required for this first stage.

Chunked prefill uses the same prefill path as normal prompt processing. For each sequence, `prepare_prefill()` computes:

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
seqlen_k = end
```

For the first chunk, `start == 0`, so query length and key length are the same.

For later chunks, `start > 0`, so `seqlen_k > seqlen_q`. In that case, `prepare_prefill()` passes `block_tables` into the attention context, allowing the prefill attention call to attend to the existing KV cache plus the new chunk.

This patch does not mix prefill tokens and decode tokens in the same model forward. A given engine step is still either:

- prefill, using variable-length prefill attention metadata, or
- decode, using paged decode attention metadata.

That distinction matters because the current context object has one global `is_prefill` flag.

## Why Not True Mixed Batching Yet?

True same-forward mixed batching would require larger attention changes. A single model invocation would contain both:

- prefill rows with `cu_seqlens_q`, `cu_seqlens_k`, and prefill slot mappings, and
- decode rows with `context_lens` and decode slot mappings.

The current attention path chooses one mode from `context.is_prefill`, so supporting both in one forward would require either:

- two separate model forwards per scheduler step, or
- hybrid attention logic that splits prefill and decode tokens inside each layer and stitches outputs back together.

The first-stage interleaving policy gets the main latency benefit with less risk.

## Expected Behavior

With only waiting prompts:

```text
prefill -> prefill -> prefill
```

With only running decoders:

```text
decode -> decode -> decode
```

With a long waiting prompt and active decoders:

```text
prefill chunk -> decode -> prefill chunk -> decode
```

If a prefill chunk completes a sequence's prompt, that sequence moves to `running` and can participate in the next decode step.

## Experiment Plan

### Scheduler Tests

These can be tested without loading a model.

1. Single long prompt

   Configure `max_num_batched_tokens` smaller than the prompt length. Verify the scheduler emits multiple prefill steps and no decode steps until the prompt has completed prefill.

2. Running decode plus long waiting prompt

   Seed one sequence in `running` and one long sequence in `waiting`. Verify the schedule order alternates:

   ```text
   prefill, decode, prefill, decode
   ```

3. Multiple running decoders plus one long prompt

   Verify each decode step schedules up to `max_num_seqs` running sequences, and that prefill still progresses every other step.

4. Multiple waiting prompts

   Verify normal prompt batching still works, and only the first oversized prompt is chunked when the prefill token budget is exhausted.

5. KV pressure and preemption

   Use a small number of KV blocks. Verify preempted sequences return to `waiting`, their block tables are cleared, and the scheduler can continue making progress.

### Correctness Tests

Run deterministic generation with the same prompts in two modes:

- no chunking: `max_num_batched_tokens >= prompt_len`
- forced chunking: `max_num_batched_tokens < prompt_len`

Use deterministic sampling, for example `temperature = 0`, and compare generated token IDs exactly.

Recommended prompt lengths:

- short prompt
- exactly one KV block
- one KV block plus one token
- multiple KV blocks
- shared-prefix prompts to exercise prefix caching

The expected result is identical generated token IDs between chunked and non-chunked prefill.

### Performance Tests

Benchmark at least three workloads:

1. One long prompt plus many short prompts.
2. Many already-running short generations, then one long prompt.
3. Continuous mixed workload, for example 80 percent short prompts and 20 percent long prompts.

Compare:

- baseline with `prefill_decode_interleave=False`
- interleaved scheduler with `prefill_decode_interleave=True`

Useful metrics:

- time to first token, p50/p90/p99
- inter-token latency, p50/p90/p99
- end-to-end request latency
- prefill tokens per second
- decode tokens per second
- total output tokens per second
- number of preemptions
- peak KV-cache usage

The expected improvement is lower decode stall time and better latency for short/running requests when long prompts are present. Total throughput may move slightly because alternating decode and prefill can reduce some batching opportunities.

## Future Work

After validating this scheduler-only change, the next possible stage is true mixed prefill/decode batching in one forward pass. That should be treated as a separate attention-layer project, not as a small scheduler tweak.
