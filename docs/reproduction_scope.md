# Reproduction Scope

The target is a recipe-faithful scaled reproduction of FDM-1's public recipe shape:

1. Train/use an IDM-style labeler.
2. Produce action labels from the labeler.
3. Train an FDM-style autoregressive next-action model on interleaved video/action tokens.
4. Evaluate on held-out action metrics and a bounded rollout harness.

This project intentionally avoids claiming parity with the private FDM-1 system, avoids massive corpus collection, and avoids real-world robot/car demos in the first pass.
