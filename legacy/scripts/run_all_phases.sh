#!/usr/bin/env bash
# Execute every phase notebook in order. Records pass/fail per notebook.
set -u
JUPYTER='./.venv/Scripts/jupyter.exe'
LOG='results/phase_runs.log'
mkdir -p results
: > "$LOG"

NB=(
  notebooks/PHASE2_RUN_ALL.ipynb
  notebooks/phase2_step5_ablation.ipynb
  notebooks/phase2_step6_generalization.ipynb
  notebooks/phase2_step7_robustness.ipynb
  notebooks/phase2_step8_explainability.ipynb
  notebooks/phase2_step9_statistical.ipynb
  notebooks/phase2_step10_final_report.ipynb
  notebooks/phase3_step1_mendeley.ipynb
  notebooks/phase3_step2_cross_attention.ipynb
  notebooks/phase3_step3_deployment.ipynb
  notebooks/phase4_tier_a_novelty.ipynb
)

pass=0; fail=0
for nb in "${NB[@]}"; do
  printf '\n========== %s ==========\n' "$nb" | tee -a "$LOG"
  start=$(date +%s)
  if "$JUPYTER" nbconvert --to notebook --execute --inplace "$nb" \
       --ExecutePreprocessor.timeout=2400 \
       --ExecutePreprocessor.kernel_name=battery_venv >>"$LOG" 2>&1; then
    end=$(date +%s); printf 'OK   %ss  %s\n' "$((end-start))" "$nb" | tee -a "$LOG"
    pass=$((pass+1))
  else
    end=$(date +%s); printf 'FAIL %ss  %s\n' "$((end-start))" "$nb" | tee -a "$LOG"
    fail=$((fail+1))
  fi
done

printf '\n=== SUMMARY: %d passed, %d failed ===\n' "$pass" "$fail" | tee -a "$LOG"
