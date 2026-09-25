#!/bin/bash

echo "Submitting HD88986 3p emcee jobs..."
echo "==============================================="

job_count=0
for script in run_HD88986_*_3p_emcee.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count 3p jobs."
