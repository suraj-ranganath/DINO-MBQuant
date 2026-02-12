# Mixed-Bit Story Notes

Run: `my_mixedbit_run`

## Key Paired Effects

- bA mixed_int4 vs uniform_int4: delta=+0.200, 95% CI [+0.000, +0.400], n=30, sign-test p=0.1094
- bB mixed_int4 vs uniform_int4: delta=+0.300, 95% CI [+0.000, +0.550], n=20, sign-test p=0.1094
- bA mixed_int3 vs uniform_int3: delta=+0.000, 95% CI [+0.000, +0.000], n=30, sign-test p=1.0000
- bB mixed_int3 vs uniform_int3: delta=+0.000, 95% CI [+0.000, +0.000], n=20, sign-test p=1.0000
- bA enc8_pred4 vs uniform_int4: delta=+0.167, 95% CI [-0.033, +0.367], n=30, sign-test p=0.1797
- bA enc6_pred4 vs uniform_int4: delta=+0.233, 95% CI [+0.033, +0.433], n=30, sign-test p=0.0654
- bA enc4_pred8 vs mixed_int4: delta=-0.133, 95% CI [-0.333, +0.067], n=30, sign-test p=0.3438
- bA enc4_pred6 vs mixed_int4: delta=-0.067, 95% CI [-0.300, +0.167], n=30, sign-test p=0.7744

## Strongest Observed Mixed-Bit Gain

- bB mixed_int4 over uniform_int4: +0.300 success delta.

## Suggested Framing

- Near the 4-bit region, allocation policy (encoder vs predictor precision) changes outcomes more than average bitwidth alone.
- 3-bit variants should be framed as a collapse regime where both uniform and mixed settings fail.
