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

## G004 result and forward implication

`G004-d2e-only-fdm-4xh200` is now complete and checkpointed in OMX. Its evidence includes 4×H200 monitor logs and a rank-imbalance diagnosis showing that path-modulo cache shard assignment could idle GPU0 near epoch tails. The repo default has been hardened to deterministic `greedy_rows` cache-shard assignment for future/recovery runs.

For `G005` and later GPU work, keep this rule active: rank-local imbalance or sustained all-GPU 0% during DDP training/inference/prediction should trigger immediate diagnosis and throughput hardening. Expected CPU/IO-only phases are allowed only when labeled and bounded.
