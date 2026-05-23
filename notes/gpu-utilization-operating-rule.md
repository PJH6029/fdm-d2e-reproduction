# GPU Utilization Operating Rule

This note is a durable goal-wide instruction for the FDM-D2E ultragoal. It applies to every MLXP/cluster run that uses reserved GPUs.

## Rule

Treat sustained 4×H200 GPU idleness as a blocker/risk, not as harmless background overhead. The ultragoal requires meaningful full-corpus training, so GPU wall-clock must be protected just like metric quality and reproducibility.

## Required practice

- Before each GPU launch, record the expected GPU-active phases, expected CPU/IO-only phases, PID files, log paths, and GPU monitor output path.
- During active training/inference/prediction phases, monitor all allocated GPUs with `nvidia-smi`, run-specific GPU monitor CSVs, and process topology evidence.
- If GPUs remain near-idle during a phase that should be GPU-active, inspect parent/child PIDs, torchrun ranks, dataloader/cache status, disk IO, logs, and barriers before assuming the run is healthy.
- Prefer implementations that keep GPUs fed while preserving audits: sharded readers, rank-disjoint shard assignment, parallel materialization, tensor-cache reuse, multi-worker prediction, resume/recovery scripts, and fail-closed artifact validation.
- CPU/IO materialization can be necessary, but it must be labeled explicitly and shortened or moved before GPU reservation when practical.
- Do not restart or kill an active long run blindly. First capture a progress snapshot, determine whether the current phase is expected CPU/IO or an unintended bottleneck, and preserve recoverable artifacts.
- Commit or otherwise persist utilization evidence for major milestones: GPU monitor summaries, progress snapshots, bottleneck diagnosis, and optimization changes.

## Current G004 implication

For `G004-d2e-only-fdm-4xh200`, continue monitoring the `bfe61db` relaunch until all epochs, checkpointing, prediction, finalization, and audit pass. Rank-0-only coordination/evaluation may cause temporary imbalance, but sustained all-GPU 0% during DDP training or prediction should trigger diagnosis and throughput hardening.
