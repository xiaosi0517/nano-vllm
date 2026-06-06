#!/usr/bin/env python3
"""Compare baseline and W8A16 memory/latency behavior.

Optional NVIDIA profiler commands:
  nsys profile -o profile_quant python benchmark_quant_profile.py --model-path ~/huggingface/Qwen3-0.6B/
  ncu --set full python benchmark_quant_profile.py --model-path ~/huggingface/Qwen3-0.6B/
"""

import argparse
import atexit
import csv
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


@dataclass
class RequestRecord:
    label: str
    prompt_tokens: int
    add_time_s: float
    first_token_time_s: float | None = None
    finish_time_s: float | None = None
    output_tokens: int = 0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def cuda_memory() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_memory_breakdown(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters and buffers, including W8A16 scales stored either way."""
    result = {
        "total_parameter_bytes": 0,
        "total_buffer_bytes": 0,
        "int8_weight_bytes": 0,
        "fp16_weight_bytes": 0,
        "bf16_weight_bytes": 0,
        "fp32_weight_bytes": 0,
        "floating_point_weight_bytes": 0,
        "scale_bytes": 0,
    }

    tensors: list[tuple[str, torch.Tensor, str]] = []
    tensors.extend((name, tensor, "parameter") for name, tensor in model.named_parameters())
    tensors.extend((name, tensor, "buffer") for name, tensor in model.named_buffers())

    for name, tensor, storage_kind in tensors:
        size = tensor_bytes(tensor)
        result[f"total_{storage_kind}_bytes"] += size
        leaf_name = name.rsplit(".", 1)[-1].lower()
        is_scale = "scale" in leaf_name
        is_weight = "weight" in leaf_name or "qweight" in leaf_name

        if is_scale:
            result["scale_bytes"] += size
        if not is_weight or is_scale:
            continue
        if tensor.dtype in (torch.int8, torch.uint8):
            result["int8_weight_bytes"] += size
        elif tensor.dtype == torch.float16:
            result["fp16_weight_bytes"] += size
            result["floating_point_weight_bytes"] += size
        elif tensor.dtype == torch.bfloat16:
            result["bf16_weight_bytes"] += size
            result["floating_point_weight_bytes"] += size
        elif tensor.dtype == torch.float32:
            result["fp32_weight_bytes"] += size
            result["floating_point_weight_bytes"] += size

    result["total_model_storage_bytes"] = (
        result["total_parameter_bytes"] + result["total_buffer_bytes"]
    )
    return result


def kv_cache_breakdown(llm: LLM) -> dict[str, int | None]:
    """Best-effort introspection so future internal layout changes do not crash."""
    runner = getattr(llm, "model_runner", None)
    config = getattr(runner, "config", None)
    scheduler = getattr(llm, "scheduler", None)
    block_manager = getattr(scheduler, "block_manager", None)
    kv_cache = getattr(runner, "kv_cache", None)

    num_blocks = getattr(config, "num_kvcache_blocks", None)
    if not isinstance(num_blocks, int) or num_blocks < 0:
        free_ids = getattr(block_manager, "free_block_ids", None)
        used_ids = getattr(block_manager, "used_block_ids", None)
        if free_ids is not None and used_ids is not None:
            num_blocks = len(free_ids) + len(used_ids)
        else:
            num_blocks = None

    block_size = getattr(config, "kvcache_block_size", None)
    if block_size is None:
        block_size = getattr(runner, "block_size", None)

    total_bytes = tensor_bytes(kv_cache) if isinstance(kv_cache, torch.Tensor) else None
    bytes_per_block = (
        total_bytes // num_blocks
        if total_bytes is not None and isinstance(num_blocks, int) and num_blocks > 0
        else None
    )
    return {
        "num_kvcache_blocks": num_blocks,
        "block_size": block_size,
        "estimated_bytes_per_kv_block": bytes_per_block,
        "estimated_total_kv_cache_bytes": total_bytes,
    }


def repeated_token_ids(tokenizer: Any, text: str, target_tokens: int) -> list[int]:
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        seed = [tokenizer.eos_token_id]
    return (seed * ((target_tokens + len(seed) - 1) // len(seed)))[:target_tokens]


def synthetic_workload(tokenizer: Any, max_model_len: int, max_tokens: int) -> list[tuple[str, list[int]]]:
    longest = max(1, min(1024, max_model_len - max_tokens))
    medium = max(1, min(256, longest))
    short = max(1, min(32, medium))
    specs = [
        ("short", short, 6, "Summarize this short note."),
        ("medium", medium, 6, "Explain the memory behavior of transformer inference."),
        ("long", longest, 4, "Analyze latency and scheduling for language model serving."),
    ]
    workload = []
    for label, length, count, text in specs:
        for index in range(count):
            workload.append((f"{label}_{index}", repeated_token_ids(tokenizer, text, length)))
    return workload


def safe_exit(llm: LLM | None) -> None:
    if llm is None:
        return
    try:
        llm.exit()
    except Exception as exc:
        print(f"warning: LLM cleanup failed: {exc}")
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass


def run_workload(
    llm: LLM,
    workload: list[tuple[str, list[int]]],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    records: dict[int, RequestRecord] = {}
    for label, prompt_token_ids in workload:
        seq = Sequence(prompt_token_ids, sampling_params)
        add_time = time.perf_counter()
        records[seq.seq_id] = RequestRecord(label, len(prompt_token_ids), add_time)
        llm.scheduler.add(seq)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    prefill_tokens = 0
    decode_tokens = 0
    prefill_time_s = 0.0
    decode_time_s = 0.0

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        before = {seq.seq_id: len(seq.completion_token_ids) for seq in seqs}
        scheduled_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        now = time.perf_counter()

        if is_prefill:
            prefill_tokens += scheduled_tokens
            prefill_time_s += step_time
        else:
            decode_tokens += scheduled_tokens
            decode_time_s += step_time

        for seq in seqs:
            record = records[seq.seq_id]
            if len(seq.completion_token_ids) > before[seq.seq_id]:
                record.output_tokens = len(seq.completion_token_ids)
                if record.first_token_time_s is None:
                    record.first_token_time_s = now
            if seq.is_finished and record.finish_time_s is None:
                record.finish_time_s = now

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    finish = time.perf_counter()
    ttft = [
        record.first_token_time_s - record.add_time_s
        for record in records.values()
        if record.first_token_time_s is not None
    ]
    e2e = [
        record.finish_time_s - record.add_time_s
        for record in records.values()
        if record.finish_time_s is not None
    ]
    return {
        "total_wall_time_s": finish - start,
        "ttft_s": distribution(ttft),
        "e2e_s": distribution(e2e),
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_execution_time_s": prefill_time_s,
        "decode_execution_time_s": decode_time_s,
        "prefill_tokens_per_s": prefill_tokens / prefill_time_s if prefill_time_s else None,
        "decode_tokens_per_s": decode_tokens / decode_time_s if decode_time_s else None,
        "completed_requests": sum(record.finish_time_s is not None for record in records.values()),
        "request_metrics": [asdict(record) for record in records.values()],
    }


def flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    weights = result.get("model_weight_memory", {})
    kv = result.get("kv_cache", {})
    load = result.get("cuda_memory_after_model_load", {})
    after = result.get("cuda_memory_after_benchmark", {})
    row = {
        "quantization": result.get("quantization"),
        "prefill_decode_interleave": result.get("prefill_decode_interleave"),
        "error": result.get("error"),
        "total_wall_time_s": metrics.get("total_wall_time_s"),
        "prefill_tokens_per_s": metrics.get("prefill_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
        "cuda_max_memory_allocated_bytes": result.get("cuda_max_memory_allocated_bytes"),
        "cuda_max_memory_reserved_bytes": result.get("cuda_max_memory_reserved_bytes"),
        "cuda_allocated_after_model_load_bytes": load.get("allocated_bytes"),
        "cuda_reserved_after_model_load_bytes": load.get("reserved_bytes"),
        "cuda_allocated_after_benchmark_bytes": after.get("allocated_bytes"),
        "cuda_reserved_after_benchmark_bytes": after.get("reserved_bytes"),
    }
    for prefix in ("ttft_s", "e2e_s"):
        values = metrics.get(prefix, {})
        for stat in ("avg", "p50", "p90", "p99"):
            row[f"{prefix}_{stat}"] = values.get(stat)
    row.update(weights)
    row.update(kv)
    return row


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def print_comparison(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("quant", "quantization"),
        ("interleave", "prefill_decode_interleave"),
        ("wall(s)", "total_wall_time_s"),
        ("TTFT p50", "ttft_s_p50"),
        ("E2E p50", "e2e_s_p50"),
        ("prefill tok/s", "prefill_tokens_per_s"),
        ("decode tok/s", "decode_tokens_per_s"),
        ("model MiB", "total_model_storage_bytes"),
        ("peak alloc MiB", "cuda_max_memory_allocated_bytes"),
    ]
    display_rows = []
    for row in rows:
        display = {}
        for heading, key in columns:
            value = row.get(key)
            if key.endswith("_bytes") and value is not None:
                value = value / (1024**2)
            display[heading] = format_value(value)
        display_rows.append(display)

    widths = {
        heading: max(len(heading), *(len(row[heading]) for row in display_rows))
        for heading, _ in columns
    }
    print("\nComparison (memory columns are MiB)")
    print("  ".join(heading.ljust(widths[heading]) for heading, _ in columns))
    print("  ".join("-" * widths[heading] for heading, _ in columns))
    for row in display_rows:
        print("  ".join(row[heading].ljust(widths[heading]) for heading, _ in columns))


def run_config(args: argparse.Namespace, quantization: str | None, interleave: bool) -> dict[str, Any]:
    label = quantization or "none"
    print(f"\n== quantization={label}, prefill_decode_interleave={interleave} ==")
    result: dict[str, Any] = {
        "quantization": label,
        "prefill_decode_interleave": interleave,
        "llm_config": {
            "model_path": os.path.expanduser(args.model_path),
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "prefill_decode_interleave": interleave,
            "quantization": quantization,
        },
        "sampling_config": {
            "temperature": 0.6,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "seed": args.seed,
        },
    }
    llm = None
    try:
        llm = LLM(
            os.path.expanduser(args.model_path),
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prefill_decode_interleave=interleave,
            quantization=quantization,
        )
        result["cuda_memory_after_model_load"] = cuda_memory()
        result["model_weight_memory"] = model_memory_breakdown(llm.model_runner.model)
        result["kv_cache"] = kv_cache_breakdown(llm)

        warmup_prompt = repeated_token_ids(llm.tokenizer, "Warm up inference.", 16)
        warmup_params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
        llm.generate([warmup_prompt], warmup_params, use_tqdm=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        workload = synthetic_workload(llm.tokenizer, args.max_model_len, args.max_tokens)
        params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens, ignore_eos=True)
        result["workload"] = [
            {"label": label, "prompt_tokens": len(prompt)} for label, prompt in workload
        ]
        result["metrics"] = run_workload(llm, workload, params)
        result["cuda_memory_after_benchmark"] = cuda_memory()
        result["cuda_max_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        result["cuda_max_memory_reserved_bytes"] = (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"configuration failed: {result['error']}")
    finally:
        safe_exit(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_model_len <= args.max_tokens:
        raise SystemExit("--max-model-len must be greater than --max-tokens")
    if min(args.max_num_batched_tokens, args.max_num_seqs, args.max_tokens) <= 0:
        raise SystemExit("batching and token limits must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.expanduser(args.output_dir), f"quant_profile_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    configurations = [
        (None, False),
        (None, True),
        ("w8a16", False),
        ("w8a16", True),
    ]
    results = [run_config(args, quantization, interleave) for quantization, interleave in configurations]
    rows = [flatten_summary(result) for result in results]

    with open(os.path.join(run_dir, "summary.json"), "w") as file:
        json.dump({"config": vars(args), "runs": results}, file, indent=2)
    write_csv(os.path.join(run_dir, "summary.csv"), rows)
    print_comparison(rows)
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
