#!/bin/bash
# Always run from this script's directory so globs find run_*.sh
# even if invoked as ../scripts_HD88986_gp_emcee/submit_all_emcee.sh
cd "$(dirname "$0")" || exit 1

echo "Submitting ALL HD88986 GP emcee jobs (sophie_gp)..."
echo "==================================================="
echo "Working directory: $(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_*_emcee.sh; do
  echo "Submitting: $script"
  bsub < "$script"
  job_count=$((job_count + 1))
  sleep 0.5
done

if [ "$job_count" -eq 0 ]; then
  echo "ERROR: no run_HD88986_*_emcee.sh scripts found in $(pwd)" >&2
  exit 1
fi
echo "Submitted $job_count jobs."
