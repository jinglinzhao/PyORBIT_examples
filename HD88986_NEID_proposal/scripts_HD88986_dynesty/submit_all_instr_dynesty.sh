#!/bin/bash
cd "$(dirname "$0")" || exit 1

echo "Submitting HD88986 all_instr dynesty jobs..."
echo "============================================"
echo "Working directory: $(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_all_instr_*_dynesty.sh; do
  echo "Submitting: $script"
  bsub < "$script"
  job_count=$((job_count + 1))
  sleep 0.5
done

if [ "$job_count" -eq 0 ]; then
  echo "ERROR: no matching run scripts found in $(pwd)" >&2
  exit 1
fi
echo "Submitted $job_count all_instr jobs."
