#!/bin/bash
# Always run from this script's directory so globs find run_*.sh
# even if invoked as ../scripts_HD88986_dynesty/submit_all_dynesty.sh
cd "$(dirname "$0")" || exit 1

echo "Submitting ALL HD88986 dynesty jobs (no GP)..."
echo "=============================================="
echo "Working directory: $(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_*_dynesty.sh; do
  echo "Submitting: $script"
  bsub < "$script"
  job_count=$((job_count + 1))
  sleep 0.5
done

if [ "$job_count" -eq 0 ]; then
  echo "ERROR: no run_HD88986_*_dynesty.sh scripts found in $(pwd)" >&2
  exit 1
fi
echo "Submitted $job_count jobs."
