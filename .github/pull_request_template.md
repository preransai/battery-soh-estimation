## What and why

<!-- One or two sentences: the problem and how this change solves it. Link the issue. -->

Closes #

## Type of change

- [ ] `feat`: new capability (feature, model, endpoint, UI)
- [ ] `fix`: bug fix
- [ ] `docs` / `test` / `refactor` / `perf` / `build` / `ci` / `chore`
- [ ] Breaking change (API schema, bundle JSON schema, CLI flags or feature names)

## Checklist

- [ ] `ruff check src tests`, `ruff format --check src tests` and `mypy src/battery_soh` pass
- [ ] `pytest -m "not raw_data"` passes; new behaviour has tests that assert ground truth
- [ ] No new model input with `leakage_risk != "none"` in a production feature set, and
      no feature computed from the full discharge or from capacity
- [ ] Evaluation stays leave-one-battery-out; no information from the held-out battery
      reaches fitting, scaling, imputation or conformal calibration
- [ ] No pickle/joblib artifacts; models remain JSON bundles listed in `MANIFEST.json`
- [ ] If results changed: `soh benchmark` / `soh train` / `soh report` were re-run and the
      regenerated files are committed, with the before/after numbers below
- [ ] Docs and `CHANGELOG.md` updated where user-visible behaviour changed

## Results (if numbers changed)

| metric (LOBO, core cells) | before | after |
|---|---|---|
| MAE (SOH %) | | |
| 90 % interval coverage | | |

## Notes for reviewers

<!-- Trade-offs, follow-ups, anything you are unsure about. -->
