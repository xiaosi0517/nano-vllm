import os
import csv
import json
import time
import argparse
from datetime import datetime

from nanovllm import LLM, SamplingParams


def run_once(interleave: bool, out_dir: str):
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        prefill_decode_interleave=interleave,
    )

    sp_short = SamplingParams(
        temperature=0.6,
        max_tokens=32,
        ignore_eos=True,
    )

    sp_long = SamplingParams(
        temperature=0.6,
        max_tokens=8,
        ignore_eos=True,
    )

    rows = []

    # 1. Start short request first.
    llm.add_request("hello", sp_short)

    # 2. Warmup: let short request enter decode phase.
    for i in range(2):
        t0 = time.perf_counter()
        outputs, num_tokens = llm.step()
        dt = time.perf_counter() - t0
        kind = "prefill" if num_tokens > 0 else "decode"

        rows.append({
            "phase": "warmup",
            "step": i,
            "kind": kind,
            "num_tokens": num_tokens,
            "latency_s": dt,
            "finished_outputs": len(outputs),
            "interleave": interleave,
        })

        print(f"warmup {i}: {kind:7s} tokens={num_tokens:5d} time={dt:.4f}s finished={len(outputs)}")

    # 3. Add long prompt after decode already exists.
    long_prompt = " ".join(["hello"] * 1000)
    llm.add_request(long_prompt, sp_long)

    # 4. Main trace.
    trace = []
    total_start = time.perf_counter()

    for i in range(20):
        if llm.is_finished():
            break

        t0 = time.perf_counter()
        outputs, num_tokens = llm.step()
        dt = time.perf_counter() - t0

        kind = "prefill" if num_tokens > 0 else "decode"
        trace.append(kind)

        rows.append({
            "phase": "main",
            "step": i,
            "kind": kind,
            "num_tokens": num_tokens,
            "latency_s": dt,
            "finished_outputs": len(outputs),
            "interleave": interleave,
        })

        print(f"step {i:02d}: {kind:7s} tokens={num_tokens:5d} time={dt:.4f}s finished={len(outputs)}")

    total_time = time.perf_counter() - total_start

    llm.exit()

    # 5. Compute summary.
    main_rows = [r for r in rows if r["phase"] == "main"]
    prefill_rows = [r for r in main_rows if r["kind"] == "prefill"]
    decode_rows = [r for r in main_rows if r["kind"] == "decode"]

    total_prefill_tokens = sum(max(0, r["num_tokens"]) for r in prefill_rows)
    total_decode_steps = len(decode_rows)

    first_decode_idx = None
    for idx, k in enumerate(trace):
        if k == "decode":
            first_decode_idx = idx
            break

    # Detect simple interleaving pattern:
    # whether decode appears before all prefill chunks are done.
    has_prefill_after_decode = False
    seen_decode = False
    for k in trace:
        if k == "decode":
            seen_decode = True
        if seen_decode and k == "prefill":
            has_prefill_after_decode = True
            break

    summary = {
        "interleave": interleave,
        "trace": trace,
        "num_main_steps": len(main_rows),
        "num_prefill_steps": len(prefill_rows),
        "num_decode_steps": len(decode_rows),
        "first_decode_step_index": first_decode_idx,
        "has_prefill_after_decode": has_prefill_after_decode,
        "total_prefill_tokens": total_prefill_tokens,
        "main_total_time_s": total_time,
        "prefill_tokens_per_s": total_prefill_tokens / total_time if total_time > 0 else None,
        "avg_prefill_step_latency_s": (
            sum(r["latency_s"] for r in prefill_rows) / len(prefill_rows)
            if prefill_rows else None
        ),
        "avg_decode_step_latency_s": (
            sum(r["latency_s"] for r in decode_rows) / len(decode_rows)
            if decode_rows else None
        ),
    }

    tag = "true" if interleave else "false"

    csv_path = os.path.join(out_dir, f"interleave_{tag}_steps.csv")
    json_path = os.path.join(out_dir, f"interleave_{tag}_summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nsummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved step trace to: {csv_path}")
    print(f"saved summary to:    {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["true", "false", "both"], default="both")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/interleave_{ts}"

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []

    if args.mode in ["false", "both"]:
        print("\n========== Running interleave=False ==========\n")
        summaries.append(run_once(False, args.out_dir))

    if args.mode in ["true", "both"]:
        print("\n========== Running interleave=True ==========\n")
        summaries.append(run_once(True, args.out_dir))

    compare_path = os.path.join(args.out_dir, "compare_summary.json")
    with open(compare_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nsaved comparison summary to: {compare_path}")


if __name__ == "__main__":
    main()
