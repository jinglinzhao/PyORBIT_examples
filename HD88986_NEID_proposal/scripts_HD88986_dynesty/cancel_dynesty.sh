#!/bin/bash

echo "Canceling all HD88986 dynesty jobs..."
echo "====================================="

job_ids=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_dynesty" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No dynesty jobs to cancel."
  exit 0
fi

echo "Jobs to cancel:"
echo "$job_ids"
echo ""
read -p "Proceed? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for id in $job_ids; do
    echo "Canceling job $id"
    bkill "$id"
  done
  echo "Done."
else
  echo "Aborted."
fi
