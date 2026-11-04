# `legacy/` — the frozen v1 pipeline

> **Status: archived, not maintained.** Nothing in this folder is imported by the
> `battery_soh` package, tested in CI, or shipped in the Docker image. It is kept
> because the v2 [leakage audit](../docs/LEAKAGE_AUDIT.md) cites it as evidence.

## What v1 was

v1 was an end-to-end SOH pipeline on the NASA PCoE (and Mendeley forklift)
datasets, built in six phases:

| Phase | Contents | Where |
|---|---|---|
| 1 | `.mat` loader, cycle filtering, ICA / DCIR / thermal features, CNN-LSTM | `src/*.py`, `main_pipeline.py` |
| 2 | Harmonised features, LSTM / Transformer baselines, SBERT "multimodal" model, ablations | `src/phase2/`, `notebooks/phase2_*` |
| 3 | NASA + Mendeley corpus, cross-attention model, first FastAPI service | `src/phase3/`, `notebooks/phase3_*` |
| 4 | Conformal / CQR intervals, PINN loss, Weibull RUL head, retrieval-augmented diagnosis | `src/phase4/`, `scripts/fit_phase4.py` |
| 5 | "Rigor suite": source-classifier probe, calibration, weighted conformal, SHAP, counterfactuals | `src/phase5/`, `reports/PHASE5_RIGOR_REPORT.md` |
| 6 | LOBO CV, PINN sweep, TENT, curriculum, contrastive SSL | `src/phase6/`, `reports/PHASE6_DEFERRED_REPORT.md` |

`api/` and `frontend/` are the v1 web service and single-file UI.

## Why it was replaced

A systematic re-audit of v1 (reproducible with `soh audit`) found that its headline numbers came from the
evaluation setup, not from the models. The full write-up, with reproducible
before/after numbers, is in [`docs/LEAKAGE_AUDIT.md`](../docs/LEAKAGE_AUDIT.md).
In short:

1. **Target leakage.** SOH was defined as `100 * capacity / C0`, and `capacity`
   was a model input (as were full-record statistics that rebuild it). The
   "prediction" was a rescaling of the label.
2. **Contaminated test set.** Most rows of the "held-out" table behind the
   reported R² ≈ 0.96 had been used for training.
3. **Invalid Mendeley labels.** Each forklift load step was treated as a cycle,
   so "SOH" was the size of a discharge pulse (median 0.6 %).
4. **Label in the text modality.** The Sentence-BERT descriptions contained the
   ground-truth SOH value verbatim.
5. **Training-loop bugs.** Early stopping kept the last epoch, not the best one
   (`state_dict().copy()` aliases live tensors), and the test set drove model
   selection.

v2 (`src/battery_soh/`) reframes the task as SOH estimation from **partial**
charge/discharge windows, evaluates it **leave-one-battery-out**, and calibrates
intervals at the **battery** level.

## Using this folder

The code is a snapshot: imports assume the old `src/` layout and the pinned
environment in `requirements-legacy-*.txt`, and trained artifacts are not
included. Notebook outputs have been stripped, and per-run compute logs that
recorded machine details were removed. The documents in `reports/` contain
claims that the audit later withdrew; read them as history, not as results.
References to "the paper" in v1 files point to an unpublished draft that is not
part of this repository.

The v1 code is also outside the v2 security posture and must not be deployed:
its API allowed any origin with credentials, loaded models with pickle /
`torch.load(weights_only=False)`, proxied an LLM with the server's own key on an
unauthenticated route, and rendered server strings with `innerHTML`. Each of
these is fixed by design in v2 (see [SECURITY.md](../SECURITY.md)).
