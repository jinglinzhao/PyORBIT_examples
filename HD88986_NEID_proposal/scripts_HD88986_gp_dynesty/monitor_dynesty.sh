#!/bin/bash

# Use -w so LSF does not truncate Job Name.

echo "HD88986 GP dynesty Job Monitor"
echo "=============================="
jobs=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_sophie_gp_.*_dynesty" || true)
if [ -z "$jobs" ]; then
  echo "No GP dynesty jobs found."
else
  echo "$jobs"
fi
echo ""
echo "Job counts by planet configuration:"
for planets in 1p 2p 3p; do
  count=$(bjobs -w 2>/dev/null | grep -cE "HD88986_.*_sophie_gp_${planets}_dynesty" || true)
  echo "  ${planets}: ${count}"
done
