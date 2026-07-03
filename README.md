# SHISHA Multi-Pipeline Manager

A clean portfolio project that demonstrates **dynamic resource management across concurrent inference pipelines**.

This project maps to the ideas in your fourth research direction from the dump (multi-pipeline runtime with SHISHA seeds, SLO-aware pipeline sizing, and baseline modes), rewritten as a reproducible benchmark.

## Why this stands out in interviews

- Demonstrates systems-level ML thinking: not just model accuracy, but **scheduler design under constrained accelerators**.
- Includes adaptive orchestration behavior (resource reclaim + shrink decisions).
- Produces concrete benchmark artifacts and visual comparisons.

## Compared schedulers

1. `no_shrink`: baseline that launches only best-throughput pipelines and never adapts.
2. `greedy_shrink`: baseline that uses minimum acceptable width from free EPs only.
3. `shisha_manager`: SHISHA-inspired manager that can reclaim EPs by shrinking lower-priority pipelines while preserving SLO slack.

## Reported metrics

- Completed jobs
- Dropped jobs
- Throughput (jobs/sec)
- Average waiting time
- P95 latency
- SLO violation rate
- EP utilization
- Number of shrink/reclaim events

## How to run

```bash
python demo_multipipeline.py --jobs 400 --seed 7 --plot
```

PowerShell shortcut:

```powershell
./run_demo.ps1
```

Outputs:

- `results/multipipeline_results.csv`
- `results/multipipeline_benchmark.png` (if `matplotlib` installed and `--plot` used)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Interview talking points

- Why static assignments underperform with bursty mixed-model traffic.
- How reclaiming EPs from low-priority pipelines increases admitted high-priority work.
- How to reason about SLO-safe shrinking and workload admission trade-offs.

## Optional upgrades

- Replace synthetic timing with your original `execution_times.txt` tables.
- Add policy-gradient or contextual bandit tuning for reclaim thresholds.
- Add a Streamlit dashboard for recruiter-friendly demos.
