# G003 progress — reset split pack hardening

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Extended packed action-slot split outputs beyond legacy `target_*` files to include reset ROADMAP/G002 roles:
  - `recording_val`
  - `recording_test`
  - `heldout_game`
  - `pseudo_idm_labeled_a`
  - `pseudo_pseudo_b`
  - `pseudo_fdm_gt_eval`
- Preserved `fdm1_recording_split`, `fdm1_heldout_game_split`, `fdm1_pseudo_label_split`, `fdm1_scale_memberships`, and split-fingerprint metadata in materialized action-slot rows.
- Tightened the G003 completion audit config so full-corpus completion requires non-empty reset split outputs, not just a broad `target_all_eval` file.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/data/fdm1_action_dataset.py src/fdm_d2e/reporting/fdm1_g003_completion.py
uv run pytest tests/test_fdm1_action_dataset_materializer.py tests/test_fdm1_g003_completion_audit.py tests/test_fdm1_mouse_bins.py -q
python3 -m json.tool configs/eval/fdm1_g003_action_dataset_completion.yaml
git diff --check
```

Result: targeted checks passed (`11 passed`).

## Remaining G003 work

Run the full D2E-480p pipeline and require all reset split outputs plus the completion audit to pass before checkpointing G003 complete.
