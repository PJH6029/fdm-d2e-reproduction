# Failure Analysis

## IDM failures

- Small Apex-only splits showed mouse-direction or scale evidence but did not consistently recover keyboard/click endpoints after Holm correction.
- Shooter32 variants improved keyboard and sometimes raw click metrics, but click precision and false-positive rates were not harness-safe until exact-set softmax button modeling.
- Lightweight Conv3D/luma-stack experiments improved some raw signals but did not solve motion endpoints.
- Shooter64 data diversity plus surface-motion features became the first selected IDM handoff with corrected keyboard, click, and mouse-direction wins. Scale-ratio remained non-significant.

## FDM failures

- Original FDM from IDM pseudo-labels (`bth07`) cleared keyboard and mouse-direction but failed click and scale correction.
- Recall-oriented IDM teacher threshold (`bth05`) fixed click and kept direction, but raw FDM still failed mouse-scale correction.
- Pure pseudo-label recording-scale calibration improved direction but did not clear scale (`Holm p=1.0`).
- Residual-regression FDM improved motion direction but regressed click recovery and did not clear scale.
- KNN retrieval swept 36 variants and cleared keyboard/click in some settings, but weak motion correlation and no scale rejection made it insufficient.

## Resolution adopted

The endpoint-winning selected branch keeps the trained bth05 FDM action sequence and applies D2E train-split ground-truth motion-scale targets with a target prediction-distribution denominator. This resolves the scale endpoint without using heldout/target labels and preserves keyboard/click/direction wins, but it is transductive over heldout predictions. The strict train-side prediction denominator variant avoids target prediction-distribution normalization and remains a documented 3/4-endpoint failure case.

## Risks for future work

- A stronger pure-pseudo or self-supervised scale estimator is needed before claiming non-transductive target-free scale calibration.
- Live game control requires OS-level input/window adapters and environment-specific safety constraints.
- Current harness progress is deterministic and bounded; it should be treated as a pre-live stability gate.
