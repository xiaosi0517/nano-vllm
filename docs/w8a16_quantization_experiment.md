# W8A16 Quantization Experiment

This experiment adds a minimal weight-only int8 path for Qwen3 linear layers.
It is intended to answer whether W8A16 is worth deeper kernel work in this
project, especially together with chunked prefill/decode interleaving.

## Scope

`quantization="w8a16"` quantizes these decoder linear weights at load time:

- attention `qkv_proj`
- attention `o_proj`
- MLP `gate_up_proj`
- MLP `down_proj`

The first pass deliberately keeps these tensors unchanged:

- activations
- KV cache
- RMSNorm weights and computation
- RoPE and attention kernels
- embeddings
- LM head

The current implementation dequantizes weights inside each linear forward. This
is a correctness and memory experiment, not a final fast int8 kernel path.

## Run A Baseline

```bash
python experiments/benchmark_latency_metrics.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --quantization none \
  --mode both \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 8 \
  --out-dir results/baseline_no_quant
```

## Run W8A16

```bash
python experiments/benchmark_latency_metrics.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --quantization w8a16 \
  --mode both \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 8 \
  --out-dir results/w8a16_weight_only
```

## Metrics To Compare

Compare each run's `compare_summary.json`:

- `peak_cuda_allocated_bytes`
- `peak_cuda_reserved_bytes`
- `model_storage_bytes`
- `model_parameter_bytes`
- `model_buffer_bytes`
- `peak_kv_cache_total_blocks`
- `prefill_tokens_per_s`
- `decode_tokens_per_s`
- `time_to_first_token_s`
- `end_to_end_request_latency_s`
- `num_preemptions`

The main success signal for this first pass is lower model memory and more KV
cache room without severe latency or quality regression.

## Expected Interpretation

If W8A16 lowers peak memory and increases `peak_kv_cache_total_blocks`, it is
worth considering a real int8 matmul kernel later.

If memory improves but latency gets worse, the experiment is still useful: it
means the dequantized fallback is correct, but performance needs kernel work.

If memory barely changes, quantizing only decoder linear weights is not enough
for this model or measurement setup. The next target would be the LM head or a
more detailed memory breakdown before doing any W8A8 work.

## Quick API Example

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-0.6B",
    enforce_eager=True,
    tensor_parallel_size=1,
    quantization="w8a16",
)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
```
