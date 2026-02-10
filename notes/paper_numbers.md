# Paper Numbers (Auto-generated)

Summary source: `results/summary.csv`

## Headline (highest opt_steps)

- opt_steps used for headline: `30`
- FP16 success rate: `1.000`
- Uniform INT8 success rate: `1.000`
- Mixed INT8 success rate: `1.000`
- Mixed - Uniform gap: `+0.000`

## Per-setting results

| Variant | opt_steps | Success mean | Success std | Avg plan time (s) | Model size (MB) | Peak mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 30 | 1.000 | 0.000 | 68.595 | 204.99 | 0.00 |
| Mixed INT8 (encoder FP16) | 30 | 1.000 | 0.000 | 69.621 | 148.31 | 0.00 |
| Uniform INT8 | 30 | 1.000 | 0.000 | 85.784 | 87.72 | 0.00 |

## Suggested abstract sentence

- On the Wall task at opt_steps=30 (Mac-only preliminary setting), mixed INT8 improved success over uniform INT8 by +0.000 absolute while retaining similar local efficiency footprint.
