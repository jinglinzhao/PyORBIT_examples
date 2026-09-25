#!/bin/bash

echo "Submitting ALL HD88986 emcee jobs (no GP)..."
echo "============================================"

job_count=0
for script in run_HD88986_*_emcee.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count jobs."
