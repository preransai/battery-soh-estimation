# 5. Exclude the Mendeley forklift dataset from v2

Date: 2026-10-12

## Status

Accepted

## Context

v1 combined the NASA cells with a Mendeley dataset of forklift battery cycling. Its loader
([`legacy/src/phase3/mendeley_loader.py`](../../legacy/src/phase3/mendeley_loader.py)) treated
every contiguous stretch of negative current with at least 50 samples as a discharge "cycle"
(L124-L147), integrated the current over that stretch as the cycle's capacity (L156-L158),
took the largest of the first five stretches as the initial capacity (L188-L190) and defined
`soh = min(100, 100 * capacity / initial_capacity)` (L214-L222).

Under a forklift duty cycle, each such stretch is one load pulse, not a full discharge. The
resulting "SOH" measures the charge drawn by one pulse relative to the largest early pulse; it
carries no ageing information, and it is computed from an input feature. The data also comes
from a different chemistry, format and operating regime than the NASA cells.

A valid SOH label for that data would need identified reference capacity measurements under
controlled conditions. The v1 code did not establish them.

## Decision

v2 uses the NASA PCoE Battery Data Set only. The Mendeley loader stays frozen under `legacy/`
as evidence for the audit; nothing in `battery_soh` reads Mendeley data, and the processed
table, the benchmark and the models contain none.

## Consequences

- No invalid labels enter training or evaluation.
- The data comes from one source: 4 core cells, 12 in the multi-condition set. Claims about
  generalisation are limited to cells and protocols like NASA's.
- Re-introducing Mendeley, or any other dataset, requires a documented label derivation from
  real reference capacity measurements and passing the same quality control and inclusion
  rule ([`data/protocols.py`](../../src/battery_soh/data/protocols.py)) as the NASA cells.
