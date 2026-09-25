#!/bin/bash

echo "HD88986 emcee Job Monitor"
echo "========================="
bjobs | grep "HD88986_.*_emcee" || echo "No emcee jobs found."
echo ""
echo "Job counts by planet configuration:"
for planets in 0p 1p 2p 3p; do
  count=$(bjobs 2>/dev/null | grep -c "HD88986_.*_${planets}_emcee" || true)
  echo "  ${planets}: ${count}"
done
