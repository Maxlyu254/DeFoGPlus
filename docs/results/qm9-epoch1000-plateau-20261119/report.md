# Are the QM9 experiments stable by epoch1000?

Assessed 2026-11-19 using all eighteen completed runs. B1 continuation12355212
and sync/email12355214 both completed; continuation allocation1.4025 A100-hours.
The B1 active history contains exactly1000 unique epochs and96,000 steps.
No additional training or generation was requested for this analysis.

## Loss

Mean logged edge cross-entropy over epochs801–900 versus901–1000, averaging
within each of three training seeds and then across seeds:

| Model | Epochs801–900 | Epochs901–1000 |
| --- | ---: | ---: |
| B0 | 0.463728 | 0.463693 |
| B1 | 0.463708 | 0.463675 |
| A0 | 0.434215 | 0.434167 |
| A1 | 0.430408 | 0.430343 |
| M0 | 0.459350 | 0.459210 |
| M1 | 0.454850 | 0.454706 |

The relative decreases are all below0.04%. Node cross-entropy, where applicable,
also changes very little (e.g. B0 0.513657→0.513535; M0 0.436956→0.436784).
A0/A1 do not train node-type denoising. Thus training loss has reached a
practical plateau, with small remaining improvements. This does not prove an
optimization optimum or the result of extending training beyond1000 epochs.
Fixed-corruption validation edge CE is broadly stable, with small residual
changes, including some improvement at high t. Absolute losses across different
conditioning/noising constructions are not controlled performance comparisons.

## Validity, uniqueness and novelty

Use one10k-attempt curve pool per checkpoint, averaged over confirmation seeds1
and2. Seed0 used2k pools and is excluded from this table: uniqueness/novelty
depend on pool size through deduplication. These single curve pools are distinct
from the three fresh final-report pools. All values are percentages.

| Model | N at600 | N at900 | N at1000 | N change900→1000 (pp) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 34.00 | 32.45 | 32.41 | -0.04 |
| B1 | 34.24 | 32.38 | 31.50 | -0.88 |
| A0 | 33.07 | 31.55 | 30.75 | -0.81 |
| A1 | 32.37 | 30.93 | 31.02 | +0.09 |
| M0 | 34.61 | 33.53 | 33.81 | +0.27 |
| M1 | 34.48 | 33.82 | 33.90 | +0.08 |

Validity is near saturation around99%. Across epochs800→1000 its model means
change by -0.15 to+0.18 pp, with almost no net movement for B0/A0/M0/M1.
Uniqueness is broadly stable around96%, with small residual movement: means
change -0.27 to+0.20 pp across800→1000. Neither is literally constant.

Novelty has fallen across600→1000 for every model. The drop is largest for
B1/A0 (about2.73/2.33 pp) and smaller for M0/M1 (about0.80/0.58 pp). In the
last100 epochs alone, B1 and A0 still decline by about0.88 and0.81 pp. Other
models look flatter, but one near-flat interval from two seeds cannot establish
an asymptotic plateau. Small reversals may be sampling or checkpoint variation.
The data do not establish that all novelty curves have stabilized at1000.

VUN reflects the same pattern: seeds1,2 mean VUN600→1000 is32.63→30.025% for
B1 and31.455→29.35% for A0. M0/M1 change more slowly (33.085→32.37% and
32.93→32.305%). Flat denoising loss can coexist with changing sampled outputs;
the flat loss therefore does not establish stability of the graph distribution.
Novelty decline alone is also insufficient to establish overfitting.

## Practical interpretation

1000 epochs is a reasonable fixed horizon for comparing mature validity and
nearly flat training loss. It should not be described as a horizon at which
V, U, N and VUN have all demonstrably converged. No automated drift checks,
stopping rule or extra epochs were added. The numerical source and per-model
loss/node-loss and equal-pool curve estimates are in assessment.json.
