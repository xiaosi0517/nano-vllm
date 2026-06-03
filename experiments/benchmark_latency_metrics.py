import argparse
import atexit
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


@dataclass
class RequestRecord:
    seq: Sequence
    label: str
    prompt_tokens: int
    target_output_tokens: int
    add_time_s: float
    first_token_time_s: float | None = None
    finish_time_s: float | None = None
    last_token_time_s: float | None = None
    output_tokens: int = 0
    preemptions: int = 0
    token_times_s: list[float] | None = None

    def __post_init__(self):
        if self.token_times_s is None:
            self.token_times_s = []


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def csv_write(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_exit(llm: LLM):
    try:
        llm.exit()
    except Exception:
        pass
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass


def make_prompt(tokenizer, word: str, approx_tokens: int) -> str:
    text = " ".join([word] * max(1, approx_tokens))
    while len(tokenizer.encode(text)) < approx_tokens:
        text = text + " " + text
    return text


def add_tracked_request(
    llm: LLM,
    records: dict[int, RequestRecord],
    prompt: str,
    sampling_params: SamplingParams,
    label: str,
    now_s: float,
) -> RequestRecord:
    prompt_token_ids = llm.tokenizer.encode(prompt)
    seq = Sequence(prompt_token_ids, sampling_params)
    llm.scheduler.add(seq)
    record = RequestRecord(
        seq=seq,
        label=label,
        prompt_tokens=len(prompt_token_ids),
        target_output_tokens=sampling_params.max_tokens,
        add_time_s=now_s,
    )
    records[seq.seq_id] = record
    return record


def patch_preemption_counter(llm: LLM, records: dict[int, RequestRecord]):
    scheduler = llm.scheduler
    original_preempt = scheduler.preempt
    counter = {"count": 0}

    def counted_preempt(seq):
        counter["count"] += 1
        record = records.get(seq.seq_id)
        if record is not None:
            record.preemptions += 1
        return original_preempt(seq)

    scheduler.preempt = counted_preempt
    return counter


def run_engine_step(llm: LLM, records: dict[int, RequestRecord], step_index: int, start_s: float) -> dict:
    scheduled_seqs, is_prefill = llm.scheduler.schedule()
    scheduled_ids = [seq.seq_id for seq in scheduled_seqs]
    scheduled_labels = [records[seq.seq_id].label for seq in scheduled_seqs if seq.seq_id in records]
    scheduled_prompt_tokens = sum(seq.num_scheduled_tokens for seq in scheduled_seqs) if is_prefill else 0
    scheduled_decode_tokens = 0 if is_prefill else len(scheduled_seqs)

    before_output_tokens = {
        seq.seq_id: len(seq.completion_token_ids)
        for seq in scheduled_seqs
        if seq.seq_id in records
    }

    t0 = time.perf_counter()
    token_ids = llm.model_runner.call("run", scheduled_seqs, is_prefill)
    model_time_s = time.perf_counter() - t0

    llm.scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)
    now_s = time.perf_counter() - start_s

    new_output_tokens = 0
    finished_ids = []
    for seq in scheduled_seqs:
        record = records.get(seq.seq_id)
        if record is None:
            continue
        before = before_output_tokens.get(seq.seq_id, 0)
        after = len(seq.completion_token_ids)
        delta = after - before
        if delta > 0:
            new_output_tokens += delta
            record.output_tokens = after
            for _ in range(delta):
                record.token_times_s.append(now_s)
            if record.first_token_time_s is None:
                record.first_token_time_s = now_s
            record.last_token_time_s = now_s
        if seq.is_finished and record.finish_time_s is None:
            record.finish_time_s = now_s
            finished_ids.append(seq.seq_id)

    used_kv_blocks = len(llm.scheduler.block_manager.used_block_ids)
    free_kv_blocks = len(llm.scheduler.block_manager.free_block_ids)

    return {
        "step": step_index,
        "kind": "prefill" if is_prefill else "decode",
        "scheduled_seq_ids": " ".join(map(str, scheduled_ids)),
        "scheduled_labels": " ".join(scheduled_labels),
        "num_scheduled_seqs": len(scheduled_seqs),
        "prefill_tokens": scheduled_prompt_tokens,
        "decode_tokens": scheduled_decode_tokens,
        "new_output_tokens": new_output_tokens,
        "finished_seq_ids": " ".join(map(str, finished_ids)),
        "step_latency_s": model_time_s,
        "elapsed_s": now_s,
        "waiting": len(llm.scheduler.waiting),
        "running": len(llm.scheduler.running),
        "used_kv_blocks": used_kv_blocks,
        "free_kv_blocks": free_kv_blocks,
    }


def build_summary(
    interleave: bool,
    step_rows: list[dict],
    records: dict[int, RequestRecord],
    total_time_s: float,
    preemption_count: int,
    peak_cuda_allocated_bytes: int | None,
    peak_cuda_reserved_bytes: int | None,
) -> dict:
    request_rows = []
    ttft = []
    e2e_latency = []
    inter_token_latency = []

    for record in records.values():
        token_times = record.token_times_s or []
        for left, right in zip(token_times, token_times[1:]):
            inter_token_latency.append(right - left)
        if record.first_token_time_s is not None:
            ttft.append(record.first_token_time_s - record.add_time_s)
        if record.finish_time_s is not None:
            e2e_latency.append(record.finish_time_s - record.add_time_s)

        request_rows.append({
            "seq_id": record.seq.seq_id,
            "label": record.label,
            "prompt_tokens": record.prompt_tokens,
            "target_output_tokens": record.target_output_tokens,
            "actual_output_tokens": record.output_tokens,
            "add_time_s": record.add_time_s,
            "first_token_time_s": record.first_token_time_s,
            "finish_time_s": record.finish_time_s,
            "time_to_first_token_s": (
                record.first_token_time_s - record.add_time_s
                if record.first_token_time_s is not None else None
            ),
            "end_to_end_latency_s": (
                record.finish_time_s - record.add_time_s
                if record.finish_time_s is not None else None
            ),
            "preemptions": record.preemptions,
        })

    total_prefill_tokens = sum(row["prefill_tokens"] for row in step_rows)
    total_decode_tokens = sum(row["decode_tokens"] for row in step_rows)
    total_output_tokens = sum(row["new_output_tokens"] for row in step_rows)
    prefill_time_s = sum(row["step_latency_s"] for row in step_rows if row["kind"] == "prefill")
    decode_time_s = sum(row["step_latency_s"] for row in step_rows if row["kind"] == "decode")
    peak_used_kv_blocks = max((row["used_kv_blocks"] for row in step_rows), default=0)
    peak_total_kv_blocks = max(
        (row["used_kv_blocks"] + row["free_kv_blocks"] for row in step_rows),
        default=0,
    )

    summary = {
        "interleave": interleave,
        "num_requests": len(records),
        "completed_requests": sum(1 for record in records.values() if record.finish_time_s is not None),
        "total_time_s": total_time_s,
        "time_to_first_token_s": summarize_distribution(ttft),
        "inter_token_latency_s": summarize_distribution(inter_token_latency),
        "end_to_end_request_latency_s": summarize_distribution(e2e_latency),
        "prefill_tokens": total_prefill_tokens,
        "decode_tokens": total_decode_tokens,
        "total_output_tokens": total_output_tokens,
        "prefill_tokens_per_s": total_prefill_tokens / prefill_time_s if prefill_time_s > 0 else None,
        "decode_tokens_per_s": total_decode_tokens / decode_time_s if decode_time_s > 0 else None,
        "total_output_tokens_per_s": total_output_tokens / total_time_s if total_time_s > 0 else None,
        "num_preemptions": preemption_count,
        "peak_kv_cache_used_blocks": peak_used_kv_blocks,
        "peak_kv_cache_total_blocks": peak_total_kv_blocks,
        "peak_kv_cache_usage_ratio": (
            peak_used_kv_blocks / peak_total_kv_blocks if peak_total_kv_blocks else None
        ),
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "num_prefill_steps": sum(1 for row in step_rows if row["kind"] == "prefill"),
        "num_decode_steps": sum(1 for row in step_rows if row["kind"] == "decode"),
        "step_trace": [row["kind"] for row in step_rows],
    }
    return summary, request_rows


def run_once(args, interleave: bool, out_dir: str) -> dict:
    tag = "true" if interleave else "false"
    run_dir = os.path.join(out_dir, f"interleave_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n========== Running prefill_decode_interleave={interleave} ==========\n")
    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prefill_decode_interleave=interleave,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    records: dict[int, RequestRecord] = {}
    preemption_counter = patch_preemption_counter(llm, records)
    start_s = time.perf_counter()
    step_rows = []

    short_prompt = make_prompt(llm.tokenizer, "hello", args.short_prompt_tokens)
    long_prompt = make_prompt(llm.tokenizer, "hello", args.long_prompt_tokens)
    sp_short = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.short_output_tokens,
        ignore_eos=True,
    )
    sp_long = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.long_output_tokens,
        ignore_eos=True,
    )

    try:
        for i in range(args.initial_short_requests):
            now_s = time.perf_counter() - start_s
            add_tracked_request(llm, records, short_prompt, sp_short, f"initial_short_{i}", now_s)

        for step_index in range(args.warmup_steps):
            if llm.is_finished():
                break
            row = run_engine_step(llm, records, step_index, start_s)
            row["phase"] = "warmup"
            step_rows.append(row)
            print(
                f"warmup {step_index:03d}: {row['kind']:7s} "
                f"prefill={row['prefill_tokens']:5d} decode={row['decode_tokens']:3d} "
                f"latency={row['step_latency_s']:.4f}s"
            )

        for i in range(args.long_requests):
            now_s = time.perf_counter() - start_s
            add_tracked_request(llm, records, long_prompt, sp_long, f"long_{i}", now_s)

        for i in range(args.late_short_requests):
            now_s = time.perf_counter() - start_s
            add_tracked_request(llm, records, short_prompt, sp_short, f"late_short_{i}", now_s)

        main_step = 0
        while not llm.is_finished() and main_step < args.max_steps:
            row = run_engine_step(llm, records, len(step_rows), start_s)
            row["phase"] = "main"
            step_rows.append(row)
            print(
                f"step {main_step:03d}: {row['kind']:7s} "
                f"prefill={row['prefill_tokens']:5d} decode={row['decode_tokens']:3d} "
                f"out={row['new_output_tokens']:3d} kv={row['used_kv_blocks']:4d} "
                f"latency={row['step_latency_s']:.4f}s"
            )
            main_step += 1

        total_time_s = time.perf_counter() - start_s
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        else:
            peak_allocated = None
            peak_reserved = None

        summary, request_rows = build_summary(
            interleave=interleave,
            step_rows=step_rows,
            records=records,
            total_time_s=total_time_s,
            preemption_count=preemption_counter["count"],
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
        )

        csv_write(os.path.join(run_dir, "steps.csv"), step_rows)
        csv_write(os.path.join(run_dir, "requests.csv"), request_rows)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        print("\nsummary:")
        print(json.dumps(summary, indent=2))
        print(f"\nsaved run results to: {run_dir}")
        return summary
    finally:
        safe_exit(llm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--mode", choices=["true", "false", "both"], default="both")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--initial-short-requests", type=int, default=4)
    parser.add_argument("--late-short-requests", type=int, default=4)
    parser.add_argument("--long-requests", type=int, default=1)
    parser.add_argument("--short-prompt-tokens", type=int, default=16)
    parser.add_argument("--long-prompt-tokens", type=int, default=1024)
    parser.add_argument("--short-output-tokens", type=int, default=64)
    parser.add_argument("--long-output-tokens", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/latency_metrics_{ts}"

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    if args.mode in ["false", "both"]:
        summaries.append(run_once(args, False, args.out_dir))
    if args.mode in ["true", "both"]:
        summaries.append(run_once(args, True, args.out_dir))

    compare_path = os.path.join(args.out_dir, "compare_summary.json")
    with open(compare_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nsaved comparison summary to: {compare_path}")


if __name__ == "__main__":
    main()
