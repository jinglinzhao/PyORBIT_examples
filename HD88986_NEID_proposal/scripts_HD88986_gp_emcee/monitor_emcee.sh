#!/bin/bash

# Use -w so LSF does not truncate Job Name.

echo "HD88986 GP emcee Job Monitor"
echo "============================"
jobs=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_sophie_gp_.*_emcee" || true)
if [ -z "$jobs" ]; then
  echo "No GP emcee jobs found."
else
  echo "$jobs"
fi
echo ""
echo "Job counts by planet configuration:"
for planets in 1p 2p 3p; do
  count=$(bjobs -w 2>/dev/null | grep -cE "HD88986_.*_sophie_gp_${planets}_emcee" || true)
  echo "  ${planets}: ${count}"
done
