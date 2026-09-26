#!/bin/bash

# Use -w so LSF does not truncate Job Name (default bjobs
# shortens HD88986_all_instr_no_gp_1p_dynesty -> *_1p_dynesty).

echo "HD88986 dynesty Job Monitor"
echo "==========================="
jobs=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_dynesty" || true)
if [ -z "$jobs" ]; then
  echo "No dynesty jobs found."
else
  echo "$jobs"
fi
echo ""
echo "Job counts by planet configuration:"
for planets in 1p 2p 3p; do
  count=$(bjobs -w 2>/dev/null | grep -cE "HD88986_.*_${planets}_dynesty" || true)
  echo "  ${planets}: ${count}"
done
