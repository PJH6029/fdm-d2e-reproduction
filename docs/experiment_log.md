# Experiment Log

## Smoke run

The smoke run should record:

- command sequence
- config files
- data manifest path/hash
- pseudo-label artifact path/hash
- FDM checkpoint metadata path/hash
- metrics JSON path/hash
- rollout smoke path/hash
- known gaps and skipped categories

Initial implementation uses deterministic synthetic D2E-shaped fixtures for local reproducibility; real D2E sample paths can replace the fixture after source-contract validation.
