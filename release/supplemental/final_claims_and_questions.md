# Final Claims and Open Questions (transition_run_v2 + pilot sweep)

## What is supported by data

1. A stable three-regime pattern appears across both runs:
   - 8-bit (uniform and mixed) tracks FP16.
   - 3-bit collapses.
   - 4-bit is the transition regime where precision allocation matters.

2. At 4-bit, encoder-preserving mixed precision is consistently better than uniform:
   - Budget bA: mixed INT4 = 0.267 vs uniform INT4 = 0.067.
   - Budget bB: mixed INT4 = 0.500 vs uniform INT4 = 0.200.
   - Paired deltas: +0.20 (bA, 95% CI [0.00, 0.40], n=30), +0.30 (bB, 95% CI [0.00, 0.55], n=20).

3. Increasing planning budget helps both INT4 variants but does not remove the gap:
   - uniform INT4 improves by +0.133 from bA->bB.
   - mixed INT4 improves by +0.233 from bA->bB.
   - mixed remains ahead at both budgets.

4. Mechanistic diagnostics align with a representation-degradation hypothesis:
   - Spearman(success, mean_state_dist) = -0.91
   - Spearman(success, mean_div_visual_emb) = -0.78
   - Spearman(success, mean_div_proprio_emb) = -0.71

5. Layerwise INT4 encoder-retention suggests encoder sensitivity is real but not uniformly distributed:
   - Best mean success at 100% encoder FP16.
   - Intermediate retention points are non-monotonic.

## Questions worth follow-up (workshop discussion value)

1. Representation vs optimization interaction:
   - How much of uniform-INT4 failure is recoverable by stronger planning alone?
   - Is there a budget level where uniform INT4 catches mixed INT4?

2. Which encoder blocks are precision-critical?
   - The non-monotonic retention curve suggests block-level structure.
   - Test per-block and per-stage bit allocation rather than coarse percentages.

3. Can calibration close the 4-bit gap?
   - Try lightweight post-training scale calibration before changing architecture/module precision.

4. Is the 4-bit transition universal across tasks?
   - Validate on PointMaze/PushT and compare whether transition location and mixed-precision gains persist.

5. Mechanistic target:
   - If preserving latent geometry is key, can explicit geometry-preserving objectives or constraints make uniform low-bit viable?
