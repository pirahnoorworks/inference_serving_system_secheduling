#!/usr/bin/env python3
"""Multi-pipeline scheduler benchmark: SHISHA manager vs baselines.

This demo mirrors ideas from the dump's multipipe runtime modes:
- mode-like baselines that either don't shrink or greedily shrink.
- SHISHA-style manager that can reclaim EPs from low-priority pipelines.
- SLO and window-aware admission outcomes.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass
class Job:
    job_id: int
    model: str
    arrival: float
    batch: int
    priority: int
    slo_target: int


@dataclass
class Running:
    job: Job
    width: int
    start_time: float
    est_total_time: float
    end_time: float


MODEL_DATA: Dict[str, Dict[str, List[float]]] = {
    "resnet50": {
        "hints": [1.2, 0.8, 1.1, 0.9, 1.5, 0.7, 1.0, 1.4],
        "layer_ms": [2.4, 2.2, 2.3, 2.0, 2.8, 2.0, 2.1, 2.5],
    },
    "vgg16": {
        "hints": [1.0, 1.3, 1.2, 1.4, 1.1, 1.5, 1.2, 1.3],
        "layer_ms": [3.0, 3.3, 3.2, 3.1, 2.9, 3.5, 3.1, 3.0],
    },
    "resnet152": {
        "hints": [1.6, 1.4, 1.5, 1.3, 1.7, 1.6, 1.2, 1.8],
        "layer_ms": [4.3, 3.9, 4.2, 4.0, 4.5, 4.1, 3.8, 4.4],
    },
}


def shisha_seed(hints: List[float], width: int) -> List[int]:
    if width <= 0:
        raise ValueError("width must be positive")
    if width >= len(hints):
        return [1] * len(hints)

    conf = [1] * len(hints)
    weights = hints[:]
    while len(conf) > width:
        idx = 0
        best = float("inf")
        for i in range(len(weights) - 1):
            merged = weights[i] + weights[i + 1]
            if merged < best:
                best = merged
                idx = i
        conf[idx] += conf[idx + 1]
        del conf[idx + 1]
        weights[idx] += weights[idx + 1]
        del weights[idx + 1]
    return conf


def stage_times(layer_ms: List[float], partition: List[int]) -> List[float]:
    out: List[float] = []
    p = 0
    for c in partition:
        out.append(sum(layer_ms[p : p + c]))
        p += c
    return out


def pipeline_time(model: str, width: int, batch: int) -> float:
    part = shisha_seed(MODEL_DATA[model]["hints"], width)
    st = stage_times(MODEL_DATA[model]["layer_ms"], part)
    return sum(st) + max(st) * max(0, batch - 1)


def best_width(model: str, max_width: int) -> int:
    best_w = 1
    best_t = float("inf")
    for w in range(1, max_width + 1):
        t = pipeline_time(model, w, batch=2)
        if t < best_t:
            best_t = t
            best_w = w
    return best_w


def acceptable_width(job: Job, max_width: int) -> int:
    best_t = min(pipeline_time(job.model, w, job.batch) for w in range(1, max_width + 1))
    slack = (100 - job.slo_target) / 100.0
    allowed = best_t * (1.0 + slack)
    valid = [w for w in range(1, max_width + 1) if pipeline_time(job.model, w, job.batch) <= allowed]
    return min(valid) if valid else best_width(job.model, max_width)


def create_stream(n: int, seed: int, arrival_rate: float) -> List[Job]:
    rnd = random.Random(seed)
    t = 0.0
    models = list(MODEL_DATA.keys())
    jobs: List[Job] = []
    for i in range(n):
        t += rnd.expovariate(arrival_rate)
        jobs.append(
            Job(
                job_id=i,
                model=rnd.choice(models),
                arrival=t,
                batch=rnd.randint(1, 5),
                priority=rnd.randint(1, 5),
                slo_target=rnd.randint(70, 95),
            )
        )
    return jobs


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def shrink_running_job(r: Running, now: float) -> Optional[Running]:
    if r.width <= 1:
        return None
    new_width = r.width - 1
    elapsed = max(0.0, now - r.start_time)
    progress = min(0.98, elapsed / max(1e-6, r.est_total_time))
    new_total = pipeline_time(r.job.model, new_width, r.job.batch)
    new_remaining = (1.0 - progress) * new_total
    return Running(
        job=r.job,
        width=new_width,
        start_time=now,
        est_total_time=new_total,
        end_time=now + new_remaining,
    )


def simulate(policy: str, jobs: List[Job], total_eps: int, max_width: int, allowed_window: float) -> Dict[str, float]:
    now = 0.0
    waiting: List[Job] = []
    running: List[Running] = []
    completed: List[Tuple[Job, float, float]] = []
    dropped = 0
    shrink_events = 0

    i = 0
    busy_area = 0.0

    def free_eps() -> int:
        return total_eps - sum(r.width for r in running)

    def maybe_drop_stale() -> None:
        nonlocal dropped
        keep: List[Job] = []
        for w in waiting:
            if now - w.arrival > allowed_window and w.priority >= 3:
                dropped += 1
            else:
                keep.append(w)
        waiting[:] = keep

    while i < len(jobs) or waiting or running:
        next_arrival = jobs[i].arrival if i < len(jobs) else float("inf")
        next_finish = min((r.end_time for r in running), default=float("inf"))
        t_next = min(next_arrival, next_finish)
        if t_next == float("inf"):
            break

        busy_area += sum(r.width for r in running) * max(0.0, t_next - now)
        now = t_next

        finished = [r for r in running if abs(r.end_time - now) < 1e-9]
        if finished:
            for r in finished:
                completed.append((r.job, r.start_time, r.end_time))
            running = [r for r in running if r not in finished]

        while i < len(jobs) and abs(jobs[i].arrival - now) < 1e-9:
            waiting.append(jobs[i])
            i += 1

        waiting.sort(key=lambda j: (j.priority, j.arrival))
        maybe_drop_stale()

        made_progress = True
        while made_progress:
            made_progress = False
            if not waiting:
                break

            for idx, job in enumerate(waiting):
                if policy == "no_shrink":
                    width = best_width(job.model, max_width)
                elif policy == "greedy_shrink":
                    width = acceptable_width(job, max_width)
                else:  # shisha_manager
                    # Choose width with balanced latency and footprint.
                    choices = list(range(acceptable_width(job, max_width), max_width + 1))
                    width = min(choices, key=lambda w: pipeline_time(job.model, w, job.batch) + 0.8 * w)

                if free_eps() >= width:
                    t = pipeline_time(job.model, width, job.batch)
                    running.append(
                        Running(
                            job=job,
                            width=width,
                            start_time=now,
                            est_total_time=t,
                            end_time=now + t,
                        )
                    )
                    waiting.pop(idx)
                    made_progress = True
                    break

                if policy == "shisha_manager" and job.priority <= 2:
                    needed = width - free_eps()
                    if needed > 0:
                        # Reclaim EPs by shrinking low-priority jobs first.
                        candidates = sorted(running, key=lambda r: (-r.job.priority, r.end_time), reverse=True)
                        replaced: Dict[int, Running] = {}
                        gained = 0
                        for r in candidates:
                            if r.job.priority <= job.priority:
                                continue
                            shrunk = shrink_running_job(r, now)
                            if shrunk is None:
                                continue
                            allowed = pipeline_time(r.job.model, best_width(r.job.model, max_width), r.job.batch)
                            allowed *= 1.0 + (100 - r.job.slo_target) / 100.0
                            projected_latency = (now - r.job.arrival) + (shrunk.end_time - now)
                            if projected_latency <= allowed:
                                replaced[r.job.job_id] = shrunk
                                gained += 1
                                if gained >= needed:
                                    break

                        if gained >= needed:
                            new_running = []
                            for r in running:
                                if r.job.job_id in replaced:
                                    new_running.append(replaced[r.job.job_id])
                                    shrink_events += 1
                                else:
                                    new_running.append(r)
                            running = new_running

                            t = pipeline_time(job.model, width, job.batch)
                            running.append(
                                Running(
                                    job=job,
                                    width=width,
                                    start_time=now,
                                    est_total_time=t,
                                    end_time=now + t,
                                )
                            )
                            waiting.pop(idx)
                            made_progress = True
                            break

    makespan = max((end for _, _, end in completed), default=1.0)
    latencies = [end - job.arrival for job, _, end in completed]
    waits = [start - job.arrival for job, start, _ in completed]

    slo_violations = 0
    for job, _, end in completed:
        best_t = min(pipeline_time(job.model, w, job.batch) for w in range(1, max_width + 1))
        allowed = best_t * (1.0 + (100 - job.slo_target) / 100.0)
        if end - job.arrival > allowed:
            slo_violations += 1

    return {
        "completed": float(len(completed)),
        "dropped": float(dropped),
        "throughput_jobs_per_sec": len(completed) / makespan,
        "avg_wait_sec": mean(waits) if waits else 0.0,
        "p95_latency_sec": percentile(latencies, 0.95),
        "slo_violation_rate": (slo_violations / len(completed)) if completed else 0.0,
        "utilization": busy_area / (total_eps * makespan),
        "shrink_events": float(shrink_events),
    }


def save_csv(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for scheduler, metrics in results.items():
        row = {"scheduler": scheduler}
        row.update(metrics)
        rows.append(row)

    fields = list(rows[0].keys()) if rows else ["scheduler"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_plot(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    if plt is None:
        return

    sched = list(results.keys())
    throughput = [results[s]["throughput_jobs_per_sec"] for s in sched]
    violation = [results[s]["slo_violation_rate"] * 100.0 for s in sched]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = range(len(sched))
    ax1.plot(x, throughput, marker="o", linewidth=2.5, color="#2a9d8f", label="Throughput")
    ax2.bar(x, violation, alpha=0.4, color="#e76f51", label="SLO Violations (%)")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(sched)
    ax1.set_ylabel("Throughput (jobs/s)")
    ax2.set_ylabel("SLO Violations (%)")
    ax1.set_title("Multi-Pipeline Orchestration: SHISHA Manager vs Baselines")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-pipeline management benchmark")
    parser.add_argument("--jobs", type=int, default=400, help="Number of stream arrivals")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--arrival-rate", type=float, default=1.6, help="Poisson arrival rate")
    parser.add_argument("--eps", type=int, default=18, help="Total execution processors")
    parser.add_argument("--max-width", type=int, default=8, help="Max pipeline width")
    parser.add_argument("--allowed-window", type=float, default=120.0, help="Max queueing window for lower priority")
    parser.add_argument("--plot", action="store_true", help="Render chart if matplotlib is available")
    args = parser.parse_args()

    stream = create_stream(args.jobs, args.seed, args.arrival_rate)
    results: Dict[str, Dict[str, float]] = {}
    for policy in ["no_shrink", "greedy_shrink", "shisha_manager"]:
        results[policy] = simulate(policy, stream, args.eps, args.max_width, args.allowed_window)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "multipipeline_results.csv"
    save_csv(csv_path, results)

    print("\n=== Multi-Pipeline Management Results ===")
    for policy, metrics in results.items():
        print(f"\n[{policy}]")
        for k, v in metrics.items():
            if k in {"completed", "dropped", "shrink_events"}:
                print(f"  {k:24s}: {int(v)}")
            else:
                print(f"  {k:24s}: {v:.4f}")

    print(f"\nSaved: {csv_path}")

    if args.plot:
        png_path = out_dir / "multipipeline_benchmark.png"
        render_plot(png_path, results)
        if png_path.exists():
            print(f"Saved: {png_path}")
        elif plt is None:
            print("Plot skipped: matplotlib is not installed.")


if __name__ == "__main__":
    main()
