#!/bin/bash

echo "HD88986 dynesty Job Status Summary"
echo "=================================="
echo ""

results_dir="/work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal/results_HD88986_dynesty"

for config in all_instr; do
  for gp in no_gp; do
    echo "Configuration: ${config} / ${gp}"
    for planets in 1p 2p 3p; do
      job_name="HD88986_${config}_${gp}_${planets}_dynesty"
      job_dir="${results_dir}/${config}/${gp}/${planets}/${job_name}"
      if [ -d "${job_dir}" ]; then
        if [ -f "${job_dir}/dynesty_results.pkl" ] || ls "${job_dir}"/*dynesty*results* >/dev/null 2>&1; then
          echo "  ${planets}: COMPLETED (or results present)"
        elif [ -f "${job_dir}/configuration_file_dynesty_run_${job_name}.log" ]; then
          echo "  ${planets}: RUNNING / LOG PRESENT"
        else
          echo "  ${planets}: NOT STARTED"
        fi
      else
        echo "  ${planets}: NOT CREATED"
      fi
    done
    echo ""
  done
done
