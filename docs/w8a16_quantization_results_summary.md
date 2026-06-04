# W8A16 Quantization Results Summary

This report summarizes the RTX 4090 experiment for the first W8A16
weight-only quantization path.

## Experiment Goal

The goal was to test whether a simple W8A16 implementation is feasible in
Nano-vLLM and whether it is worth deeper optimization work.

The implementation quantizes decoder linear weights to int8 at load time and
keeps activations in fp16/bf16. It currently dequantizes weights during each
linear forward call, so this is a feasibility and measurement experiment rather
than a final optimized int8 inference path.

## Tested Runs

The main four runs were:

- `results/rtx4090_baseline_eager`
- `results/rtx4090_w8a16_eager`
- `results/rtx4090_baseline_cudagraph`
- `results/rtx4090_w8a16_cudagraph`

Each run tested both scheduler modes:

- `prefill_decode_interleave=False`
- `prefill_decode_interleave=True`

## Eager Mode Results

| Run | Interleave | Total Time | Model Storage | KV Blocks | Prefill tok/s | Decode tok/s | TTFT p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline eager | false | 96.36s | 1.42 GiB | 713 | 23.18 | 11.13 | 48.97s |
| baseline eager | true | 87.77s | 1.42 GiB | 673 | 21.95 | 14.72 | 52.19s |
| W8A16 eager | false | 75.54s | 1.01 GiB | 729 | 27.24 | 15.62 | 41.43s |
| W8A16 eager | true | 84.82s | 1.01 GiB | 703 | 21.71 | 16.35 | 52.69s |

In eager mode, W8A16 reduced persistent model storage and improved the
non-interleaved run. The interleaved W8A16 run was only slightly faster than the
interleaved baseline.

## CUDA Graph Mode Results

| Run | Interleave | Total Time | Model Storage | KV Blocks | Prefill tok/s | Decode tok/s | TTFT p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline cudagraph | false | 47.19s | 1.42 GiB | 713 | 24.54 | 2277.51 | 46.12s |
| baseline cudagraph | true | 40.42s | 1.42 GiB | 671 | 28.67 | 2215.04 | 39.89s |
| W8A16 cudagraph | false | 38.46s | 1.01 GiB | 729 | 30.33 | 1127.10 | 37.23s |
| W8A16 cudagraph | true | 45.57s | 1.01 GiB | 700 | 25.58 | 999.49 | 44.64s |

CUDA graph mode is much faster for decode-heavy steps. W8A16 improved the
non-interleaved CUDA graph run, but the interleaved W8A16 run was slower than
the interleaved baseline.

## Key Findings

The W8A16 path is functional. It loads, runs, and completes all benchmark
requests in both eager and CUDA graph modes.

Persistent model storage decreased from about 1.42 GiB to about 1.01 GiB,
roughly a 29% reduction.

Available KV cache blocks increased in the W8A16 runs:

- eager false: 713 to 729 blocks
- eager true: 673 to 703 blocks
- cudagraph false: 713 to 729 blocks
- cudagraph true: 671 to 700 blocks

Peak CUDA allocation did not clearly decrease. This is expected because the
current implementation dequantizes int8 weights back to floating-point tensors
during `forward`, creating temporary tensors that offset persistent storage
savings.

Latency did not show a consistent W8A16 win. Some W8A16 runs were faster, but
the best interleaved CUDA graph result remained the baseline run.

## Conclusion

This experiment is successful as a feasibility milestone.

It proves that W8A16 weight-only quantization can be integrated into this
codebase and benchmarked alongside chunked prefill/decode interleaving. It also
shows a real reduction in persistent model storage.

It should not yet be presented as a production-speed quantization path. The
current dequantize-on-forward implementation is useful for correctness and
measurement, but reliable speedup or peak-memory reduction would require a real
int8 matmul kernel or backend integration.

## Recommended Next Step

The next step should be one of:

- add an optimized int8 GEMM path for the quantized linear layers
- integrate an existing quantized linear backend
- add a memory breakdown benchmark that separates persistent model storage,
  temporary forward allocations, and KV cache allocation

W8A8 should remain out of scope until W8A16 has a real optimized matmul path.

